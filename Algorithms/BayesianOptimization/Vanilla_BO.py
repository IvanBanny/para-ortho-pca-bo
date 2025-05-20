"""A standard BO implementation."""

from typing import Union, Callable, Optional, Dict, Any
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
from Algorithms.utils.vis_utils import PCABOVisualizer

import warnings
from botorch.exceptions.warnings import NumericsWarning, OptimizationWarning

warnings.filterwarnings("ignore", category=NumericsWarning)  # Filter warnings from EI
warnings.filterwarnings("ignore", category=OptimizationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

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

    TIME_PROFILES = ["fit_gpytorch_mll", "optimize_acqf"]

    def __init__(
            self,
            budget: int,
            n_DoE: int = 10,
            q: int = 1,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 69,
            torch_config: Optional[Dict[str, Any]] = None,
            visualize: bool = False,
            vis_output_dir: str = "./visualizations",
            **kwargs
    ):
        """Initialize the Vanilla BO optimizer with the given parameters.

        Args:
            budget (int): Maximum number of function evaluations.
            n_DoE (int, optional): Number of initial design of experiments samples. Defaults to 10.
            q (int): Number of candidates to sample in parallel. Defaults to 1.
            acquisition_function (str): Acquisition function name. Defaults to "expected_improvement".
            random_seed (int, optional): Random seed for reproducibility. Defaults to 69.
            torch_config (Dict[str, Any], optional): gpu configuration.
            visualize (bool, optional): Whether to make 2d visualizations. Defaults to False.
            vis_output_dir (str, optional): Directory to save visualizations. Defaults to "./visualizations".
            **kwargs: Additional keyword arguments for the parent class.
        """
        # Call the superclass
        super().__init__(budget, n_DoE, **kwargs)

        self.random_seed = random_seed
        self.q = q

        # Acquisition function attributes
        self.__acqf_class = None
        self.__acqf = None
        self.acquisition_function_name = acquisition_function

        self.__torch_config = {
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float,
            "NUM_RESTARTS": 20,
            "RAW_SAMPLES": 1024,
            "OPTIMIZE_ACQF_OPTIONS": {"batch_limit": 5, "maxiter": 200, "method": "L-BFGS-B"},
            **(torch_config or {})
        }

        # Set device and dtype from torch_config
        self.device = self.__torch_config["device"]
        self.dtype = self.__torch_config["dtype"]

        # Ensure CUDA is available if device is CUDA
        if self.device.type == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA specified but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.__torch_config["device"] = self.device

        if self.verbose:
            print(f"\nUsing device: {self.device}")

        # Initialize visualizer if requested
        self.visualize = visualize
        self.visualizer = PCABOVisualizer(output_dir=vis_output_dir) if self.visualize else None

    def __str__(self):
        return "This is an instance of a Vanilla BO Optimizer"

    def __call__(
            self,
            problem: Union[RealSingleObjective, Callable],
            dim: Optional[int] = -1,
            bounds: Optional[np.ndarray] = None,
            **kwargs
    ) -> None:
        """Execute the optimization algorithm.

        Args:
            problem (Union[RealSingleObjective, Callable]): The optimization problem.
            dim (Optional[int], optional): Dimension of the problem. Defaults to -1.
            bounds (Optional[np.ndarray], optional): Bounds for the decision variables. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        # If used in the experiment setting - redirect stdout to the progress bar printing
        original_stdout = None
        if self._pbar is not None:
            original_stdout = redirect_stdout_to_tqdm(self._pbar)

        # Save current randomness states and impose seed
        self.impose_random_seed()

        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)

        # Update the progress bar with the doe points
        if self._pbar is not None:
            self._pbar.update(self.n_DoE)

        # Start the optimization loop
        while self.number_of_function_evaluations < self.budget:
            # Initialize and fit the GPR model
            self._initialize_model(**kwargs)

            new_x = self.optimize_acqf_and_get_candidates()
            evals = min(len(new_x), self.budget - self.number_of_function_evaluations)
            new_x = new_x[: evals]

            new_f = self.problem(new_x)

            self.x_evals = np.concatenate([self.x_evals, new_x])
            self.f_evals = np.concatenate([self.f_evals, new_f])

            print(f"\nSampled:\n"
                  f"    x: {new_x}\n"
                  f"    f: {new_f}")

            if self._pbar is not None:
                self._pbar.update(evals)

            self.number_of_function_evaluations += evals

            # Assign the new best
            self.assign_new_best()

            # Print best to screen if verbose
            if self.verbose:
                print(
                    f"\nEvals: {self.number_of_function_evaluations}/{self.budget}",
                    f"Best:\n"
                    f"    x: {self.x_evals[self.current_best_index]}\n"
                    f"    y: {self.current_best}",
                    flush=True
                )

            # Create visualizations
            if self.visualize and self.visualizer is not None:
                self.visualizer.visualize(
                    "bo",
                    torch.tensor(self.x_evals, device=self.device, dtype=self.dtype),
                    torch.tensor(self.f_evals, device=self.device, dtype=self.dtype),
                    torch.tensor(self.x_evals[self.current_best_index], device=self.device, dtype=self.dtype),
                    torch.tensor(self.bounds, device=self.device, dtype=self.dtype),
                    new_x, self.problem, self.__acqf, margin=0.1
                )

        # Save the visualization gifs if it was enabled
        if self.visualize and self.visualizer is not None:
            self.visualizer.save_gifs(
                postfix=f"{problem.meta_data.problem_id}_{problem.meta_data.instance}",
                duration=1000
            )

        if self.verbose:
            print("\nOptimization Process finalized!\n")

        # Restore initial randomness states
        self.restore_random_states()

        if self._pbar is not None:
            restore_stdout(original_stdout)

    def assign_new_best(self):
        """Assign the new best solution found so far."""
        super().assign_new_best()

    def _initialize_model(self, **kwargs):
        """This function initializes and fits the Gaussian Process Regression.

        Args:
            **kwargs: Left these keyword arguments for upcoming developments
        """
        # Convert train data to Tensors
        bounds_torch = torch.tensor(self.bounds, device=self.device, dtype=self.dtype).T
        train_x = torch.tensor(self.x_evals, device=self.device, dtype=self.dtype)
        train_obj = torch.tensor(self.f_evals, device=self.device, dtype=self.dtype)

        # Initialize and fit GP
        start_time = perf_counter()
        self.__model_obj = SingleTaskGP(
            train_x,
            train_obj,
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=train_x.shape[-1],
                bounds=bounds_torch
            )
        )
        mll = ExactMarginalLogLikelihood(self.__model_obj.likelihood, self.__model_obj)
        fit_gpytorch_mll(mll)
        self.timing_logs["fit_gpytorch_mll"].append(perf_counter() - start_time)

    def optimize_acqf_and_get_candidates(self) -> Tensor:
        """Optimizes the acquisition function, and returns new candidates.

        Returns:
            Tensor: `batch_shape x r`-dim Tensor - new candidate points in the reduced space.
        """
        # Set up the acquisition function
        self.acquisition_function = self.acquisition_function_class(
            model=self.__model_obj,
            best_f=self.current_best,
            maximize=self.maximization
        )

        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=torch.tensor(self.bounds, device=self.device, dtype=self.dtype).T,
            q=self.q,
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],
            options=self.__torch_config['OPTIMIZE_ACQF_OPTIONS'],
        )
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        return candidates

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
            self.__acqf_class = LogExpectedImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[1]:
            self.__acqf_class = ProbabilityOfImprovement
        elif self.__acquisition_function_name == ALLOWED_ACQUISITION_FUNCTION_STRINGS[2]:
            self.__acqf_class = UpperConfidenceBound

    @property
    def acquisition_function_class(self) -> Callable:
        """Get the acquisition function class."""
        return self.__acqf_class

    @property
    def acquisition_function(self) -> AnalyticAcquisitionFunction:
        """Get the stored acquisition function defined at some point of the loop.

        Returns:
            AnalyticAcquisitionFunction: The acquisition function.
        """
        return self.__acqf

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
            self.__acqf = new_acquisition_function
        else:
            raise AttributeError(
                "Acquisition function does not inherit from 'AnalyticAcquisitionFunction'",
                name="acquisition_function",
                obj=self.__acqf
            )
