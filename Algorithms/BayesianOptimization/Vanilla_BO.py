"""A standard BO implementation."""

from typing import Union, Callable, Optional
import os
from time import perf_counter
import numpy as np

import torch
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition.analytic import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    AnalyticAcquisitionFunction
)
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll

from ioh.iohcpp.problem import RealSingleObjective

from Algorithms.utils.tqdm_write_stream import redirect_stdout_to_tqdm, restore_stdout
from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import AbstractBayesianOptimizer

# Constants for acquisition function names
ALLOWED_ACQUISITION_FUNCTION_STRINGS = (
    "expected_improvement",
    "probability_of_improvement",
    "upper_confidence_bound"
)

ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS = {
    "EI": "expected_improvement",
    "PI": "probability_of_improvement",
    "UCB": "upper_confidence_bound"
}


class Vanilla_BO(AbstractBayesianOptimizer):
    """Vanilla Bayesian Optimization implementation."""

    TIME_PROFILES = ["SingleTaskGP", "optimize_acqf"]

    def __init__(
            self,
            budget: int,
            n_DoE: int = 0,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 43,
            **kwargs
    ):
        """Initialize the Vanilla BO optimizer with the given parameters."""
        # Call the superclass
        super().__init__(budget, n_DoE, **kwargs)

        self.random_seed = random_seed

        # Check the defaults
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float
        smoke_test = os.environ.get("SMOKE_TEST")

        # Set up the main configuration
        self.__torch_config = {
            "device": device,
            "dtype": dtype,
            "SMOKE_TEST": smoke_test,
            "BATCH_SIZE": 3 if not smoke_test else 2,
            "NUM_RESTARTS": 10 if not smoke_test else 2,
            "RAW_SAMPLES": 512 if not smoke_test else 32
        }

        # Set up the acquisition function
        self.__acq_func_class = None
        self.__acq_func = None
        self.acquisition_function_name = acquisition_function

    def __str__(self):
        return "This is an instance of a Vanilla BO Optimizer"

    def __call__(
            self,
            problem: Union[RealSingleObjective, Callable],
            dim: Optional[int] = -1,
            bounds: Optional[np.ndarray] = None,
            **kwargs
    ) -> None:
        """Execute the optimization algorithm."""
        if self._pbar is not None:
            original_stdout = redirect_stdout_to_tqdm(self._pbar)

        # Save current randomness states and impose seed
        self.impose_random_seed()

        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)

        if self._pbar is not None:
            self._pbar.update(self.n_DoE)

        # Start the optimisation loop
        for cur_iteration in range(self.budget - self.n_DoE):
            if self.number_of_function_evaluations >= self.budget:
                break

            # Initialize and fit the GPR model
            self._initialise_model(**kwargs)

            # Set up the acquisition function
            self.acquisition_function = self.acquisition_function_class(
                model=self.__model_obj,
                best_f=self.current_best,
                maximize=self.maximization
            )

            new_x = self.optimize_acqf_and_get_observation()

            # Append the new values
            for _, new_x_arr in enumerate(new_x):
                if self.number_of_function_evaluations >= self.budget:
                    break

                new_x_arr_numpy = new_x_arr.detach().numpy().ravel()

                # Append the new value
                self.x_evals.append(new_x_arr_numpy)

                # Evaluate the function
                new_f_eval = problem(new_x_arr_numpy)

                if self._pbar is not None:
                    self._pbar.update(1)

                # Append the function evaluation
                self.f_evals.append(new_f_eval)

                # Increment the number of function evaluations
                self.number_of_function_evaluations += 1

            # Assign the new best
            self.assign_new_best()

            # Print best to screen if verbose
            if self.verbose:
                print(
                    # f"Iteration: {cur_iteration + 1}",
                    f"Evaluations: {self.number_of_function_evaluations}/{self.budget}",
                    f"Best: x:{self.x_evals[self.current_best_index]} y:{self.current_best}",
                    flush=True
                )

        if self.verbose:
            print("Optimisation Process finalized!")

        # Restore initial randomness states
        self.restore_random_states()

        if self._pbar is not None:
            restore_stdout(original_stdout)

    def assign_new_best(self):
        """Assign the new best solution."""
        # Call the super class
        super().assign_new_best()

    def _initialise_model(self, **kwargs):
        """This function initializes and fits the Gaussian Process Regression.

        Args:
            **kwargs: Left these keyword arguments for upcoming developments
        """
        # Convert bounds array to Torch
        bounds_torch = torch.from_numpy(self.bounds.transpose()).detach()

        # Convert the initial values to Torch Tensors
        train_x = np.array(self.x_evals).reshape((-1, self.dimension))
        train_x = torch.from_numpy(train_x).detach()

        train_obj = np.array(self.f_evals).reshape((-1, 1))
        train_obj = torch.from_numpy(train_obj).detach()

        # Initialize and fit GP
        start_time = perf_counter()
        self.__model_obj = SingleTaskGP(
            train_x,
            train_obj,
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=train_x.shape[-1],
                transform_on_eval=False,
                transform_on_train=False,
                transform_on_fantasize=False,
                bounds=bounds_torch
            )
        )
        mll = ExactMarginalLogLikelihood(self.__model_obj.likelihood, self.__model_obj)
        fit_gpytorch_mll(mll)
        self.timing_logs["SingleTaskGP"].append(perf_counter() - start_time)

    def optimize_acqf_and_get_observation(self) -> Tensor:
        """Optimizes the acquisition function, and returns a new candidate."""
        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=torch.from_numpy(self.bounds.transpose()).detach(),
            q=1,  # self.__torch_config['BATCH_SIZE'],
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],  # used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        # Observe new values
        new_x = candidates.detach()
        new_x = new_x.reshape(shape=(1, -1)).detach()

        return new_x

    def __repr__(self):
        return super().__repr__()

    def reset(self):
        """Reset the optimizer state."""
        super().reset()

    @property
    def torch_config(self) -> dict:
        """Get the torch configuration."""
        return self.__torch_config

    @property
    def acquisition_function_name(self) -> str:
        """Get the acquisition function name."""
        return self.__acquisition_function_name

    @acquisition_function_name.setter
    def acquisition_function_name(self, new_name: str) -> None:
        """Set the acquisition function name.

        Args:
            new_name (str): Name of the acquisition function.

        Raises:
            ValueError: If the acquisition function name is not recognized.
        """
        # Remove some spaces
        new_name = new_name.strip()

        # Check in the reduced
        if new_name in ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS:
            self.__acquisition_function_name = ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS[new_name]
        else:
            if new_name.lower() in ALLOWED_ACQUISITION_FUNCTION_STRINGS:
                self.__acquisition_function_name = new_name
            else:
                raise ValueError(f"Oddly defined name {new_name}")

        # Run to set up the acquisition function subclass
        self.set_acquisition_function_subclass()

    def set_acquisition_function_subclass(self) -> None:
        """Set the acquisition function subclass based on the name."""
        if self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[0]:
            self.__acq_func_class = LogExpectedImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[1]:
            self.__acq_func_class = ProbabilityOfImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[2]:
            self.__acq_func_class = UpperConfidenceBound

    @property
    def acquisition_function_class(self) -> Callable:
        """Get the acquisition function class."""
        return self.__acq_func_class

    @property
    def acquisition_function(self) -> AnalyticAcquisitionFunction:
        """Get the stored acquisition function defined at some point of the loop.

        Returns:
            AnalyticAcquisitionFunction: The acquisition function.
        """
        return self.__acq_func

    @acquisition_function.setter
    def acquisition_function(self, new_acquisition_function: AnalyticAcquisitionFunction) -> None:
        """Set the acquisition function.

        Args:
            new_acquisition_function (AnalyticAcquisitionFunction): The acquisition function.

        Raises:
            AttributeError: If the new acquisition function is not a subclass of AnalyticAcquisitionFunction.
        """
        if issubclass(type(new_acquisition_function), AnalyticAcquisitionFunction):
            # Assign in this case
            self.__acq_func = new_acquisition_function
        else:
            raise AttributeError(
                "Acquisition function does not inherit from 'AnalyticAcquisitionFunction'",
                name="acquisition_function",
                obj=self.__acq_func
            )
