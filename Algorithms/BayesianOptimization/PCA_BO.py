"""A standard PCA-BO implementation."""

import os
from math import ceil
from typing import Union, Callable, Optional, Dict, Any
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from gpytorch.kernels import MaternKernel

from botorch.models.transforms.outcome import Standardize
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Normalize
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    AnalyticAcquisitionFunction
)
from botorch.optim import optimize_acqf
from botorch.optim.initializers import sample_q_batches_from_polytope

from ioh.iohcpp.problem import RealSingleObjective

from Algorithms.utils.tqdm_write_stream import redirect_stdout_to_tqdm, restore_stdout
from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import AbstractBayesianOptimizer
from Algorithms.BayesianOptimization.PenalizedAcqf import PenalizedAcqf
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


class PCA_BO(AbstractBayesianOptimizer):
    """PCA-assisted Bayesian Optimization implementation with GPU parallelization support.

    This class implements Bayesian Optimization with dimensionality reduction
    using Principal Component Analysis (PCA). It reduces the search space
    dimensionality by transforming the data to a lower-dimensional space
    where the optimization is performed, and then maps the solutions back to
    the original space.
    """

    TIME_PROFILES = ["SingleTaskGP", "optimize_acqf"]

    def __init__(
            self,
            budget: int,
            n_DoE: int = 0,
            n_components: int = 0,
            var_threshold: float = 0.95,
            ortho_samples: int = 0,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 69,
            torch_config: Optional[Dict[str, Any]] = None,
            visualize: bool = False,
            vis_output_dir: str = "./visualizations",
            save_logs: bool = False,
            log_dir: str = "./logs",
            **kwargs
    ):
        """Initialize the PCA-BO optimizer with the given parameters.

        Args:
            budget (int): Maximum number of function evaluations.
            n_DoE (int, optional): Number of initial design of experiments samples. Defaults to 0.
            n_components (int, optional): Number of principal components to use. If 0, determined
                                          by var_threshold. Defaults to 0.
            var_threshold (float, optional): Variance threshold for selecting components. Defaults to 0.95.
            ortho_samples (int, optional): Number or additional samples in the orthogonal per iteration.
                                           If 0, only samples the original candidate. Defaults to 0.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 43.
            torch_config (Dict[str, Any], optional): gpu configuration.
            visualize (bool, optional): Whether to visualize the acquisition function. Defaults to False.
            vis_output_dir (str, optional): Directory to save visualizations. Defaults to "./visualizations".
            save_logs (bool, optional): Whether to log and save detailed data per iteration. Defaults to False.
            log_dir (str, optional): Directory to save logs. Defaults to "./logs".
            **kwargs: Additional keyword arguments for the parent class.
        """
        # Call the superclass
        super().__init__(budget, n_DoE, **kwargs)

        self.random_seed = random_seed
        self.ortho_samples = ortho_samples

        # Acquisition function attributes
        self.__acq_func_class = None
        self.__acq_func = None
        self.__pacqf = None
        self.acquisition_function_name = acquisition_function

        self.__torch_config = {
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float,
            "BATCH_SIZE": 1,
            "NUM_RESTARTS": 20,
            "RAW_SAMPLES": 1024,
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

        # Set PCA parameters
        self.n_components = n_components
        self.var_threshold = var_threshold

        # PCA attributes
        self.data_mean = None
        self.component_matrix = None
        self.pca_mean = None
        self.explained_variance_ratio = None
        self.reduced_space_dim = None
        self.__z_evals = torch.tensor(0, device=self.device, dtype=self.dtype)  # Points in the reduced space

        # Initialize visualizer if requested
        self.visualize = visualize
        self.visualizer = PCABOVisualizer(output_dir=vis_output_dir) if self.visualize else None

        self.save_logs = save_logs
        self.log_dir = log_dir
        if self.save_logs:
            os.makedirs(self.log_dir, exist_ok=True)

    def __str__(self):
        return "This is an instance of a PCA-assisted BO Optimizer"

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
            # Initialize __z_evals with transformed points
            self._transform_points_to_reduced_space()

            # Initialize and fit the GPR model
            self._initialize_model(**kwargs)

            new_z = self.optimize_acqf_and_get_observation()
            evals = min(
                len(new_z),
                ceil((self.budget - self.number_of_function_evaluations) / (self.ortho_samples + 1))
            )
            new_z = new_z[: evals]

            new_x = self._transform_points_to_original_space(new_z)

            if self.ortho_samples:
                quxt = perf_counter()
                new_x = self.get_orthogonal_samples(new_x)
                evals = min(len(new_x), self.budget - self.number_of_function_evaluations)
                new_x = new_x[: evals]
                new_z = new_z.repeat_interleave(self.ortho_samples + 1, dim=0)[: evals]
                print(f"quxt: {perf_counter() - quxt}")

            bounds_torch = torch.tensor(self.bounds, device=self.device, dtype=self.dtype).T

            outside_bounds = ~(new_x >= bounds_torch[0]).all(dim=1) | ~(new_x <= bounds_torch[1]).all(dim=1)

            if not (~outside_bounds).all().item() and self.verbose:
                print(f"Warning: transformed candidates are out of bounds: {new_x[outside_bounds]}")

            new_f_eval = self.problem(new_x)

            self.x_evals = np.concatenate([self.x_evals, new_x])
            self.__z_evals = torch.cat([self.__z_evals, new_z])
            self.f_evals = np.concatenate([self.f_evals, new_f_eval])

            print(f"\nSampled:\n"
                  f"    x: {new_x}\n"
                  f"    f: {new_f_eval}")

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

            # Save logs
            if self.save_logs:
                acqf_values_log, penalty_log, pei_values_log = self.__pacqf.log_forward(
                    self.__z_evals.unsqueeze(1)
                )
                arrays = [
                    self.x_evals, self.__z_evals.detach().numpy(), self.f_evals,
                    acqf_values_log.detach().numpy(),
                    penalty_log.detach().numpy(),
                    pei_values_log.detach().numpy()
                ]
                names = ["x", "z", "p", "acqf", "penalty", "pacqf"]
                df = pd.DataFrame({name: list(array) for name, array in zip(names, arrays)})

                run_log_dir = os.path.join(
                    self.log_dir,
                    f"log_{problem.meta_data.problem_id}_{problem.meta_data.instance}"
                )
                os.makedirs(run_log_dir, exist_ok=True)
                df.to_csv(os.path.join(run_log_dir, f"{self.number_of_function_evaluations}.csv"), index=False)

            # Create visualizations
            if self.visualize and self.visualizer is not None:
                self.visualizer.visualize_pcabo(
                    torch.tensor(self.x_evals, device=self.device, dtype=self.dtype),
                    torch.tensor(self.f_evals, device=self.device, dtype=self.dtype),
                    torch.tensor(self.x_evals[self.current_best_index], device=self.device, dtype=self.dtype),
                    torch.tensor(self.bounds, device=self.device, dtype=self.dtype),
                    new_x, self.component_matrix, self.reduced_space_dim, self.data_mean + self.pca_mean,
                    self.problem, self.__pacqf, margin=0.1
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

    def _calculate_weights(self) -> Tensor:
        """Calculate rank-based weights for PCA transformation.

        This implements the rank-based weighting scheme described in the original PCA-BO paper,
        where better points (with lower function values for minimization) are assigned higher weights.

        Returns:
            Tensor: Weights for each data point.
        """
        n = len(self.f_evals)

        # Flatten the array to handle 2D case
        f_vals = np.array(self.f_evals).flatten()

        # Get the ranking of points (1 = best, n = worst)
        if self.maximization:
            ranks = np.argsort(np.argsort(-f_vals)) + 1
        else:
            ranks = np.argsort(np.argsort(f_vals)) + 1

        # Calculate pre-weights
        pre_weights = np.log(n) - np.log(ranks)

        # Normalize weights
        weights = pre_weights / pre_weights.sum()

        return torch.from_numpy(weights).unsqueeze(1).to(device=self.device, dtype=self.dtype)

    def _transform_points_to_reduced_space(self) -> None:
        """Transform evaluated points to the reduced space using weighted PCA.

        This method performs the following steps:
        1. Calculate weights based on function values
        2. Fit PCA on the original points with the weights
        3. Transform all points to the reduced space
        """
        if len(self.x_evals) < 2:
            # Not enough points for PCA yet
            if len(self.x_evals) == 1:
                self.__z_evals = torch.zeros(1, 1)
            return

        x_torch = torch.tensor(self.x_evals, device=self.device, dtype=self.dtype)

        # Center the data first
        self.data_mean = x_torch.mean(dim=0)
        x_centered = x_torch - self.data_mean

        # Calculate weights
        weights = self._calculate_weights()

        # Apply the weights
        # Note: applying a square root here, which makes more sense, but isn't mentioned in the original paper
        x_weighted = x_centered * torch.sqrt(weights)

        # Add a small amount of noise to avoid numerical issues
        noise = torch.normal(0, 1e-8, size=x_weighted.shape)
        x_weighted += noise

        # Center weighted
        self.pca_mean = x_weighted.mean(dim=0)
        x_weighted_centered = x_weighted - self.pca_mean

        # Apply SVD
        _, S, self.component_matrix = torch.linalg.svd(x_weighted_centered, full_matrices=False)

        explained_variance = (S ** 2) / (x_weighted.shape[0] - 1)
        self.explained_variance_ratio = explained_variance / torch.sum(explained_variance)

        n_components = self.n_components
        if n_components <= 0:
            # If using variance threshold - select components that explain enough variance
            cumulative_var_ratio = torch.cumsum(self.explained_variance_ratio, dim=0)
            n_components = torch.sum(cumulative_var_ratio <= self.var_threshold).item() + 1
            n_components = max(1, min(n_components, len(self.explained_variance_ratio)))

        self.reduced_space_dim = n_components

        if self.verbose:
            print(f"\nUsing {n_components} principal components with "
                  f"{torch.sum(self.explained_variance_ratio[:n_components]) * 100:.2f}% explained variance")

        # Transform all points to the reduced space
        # We need to transform the centered, but unweighted data
        self.__z_evals = (x_centered - self.pca_mean) @ self.component_matrix[: n_components].T

    def _transform_points_to_original_space(self, z: torch.tensor) -> torch.tensor:
        """Transform points from reduced space back to the original space.

        Args:
            z (torch.tensor): A `batch_shape x d`-dim Tensor of points in the reduced space.

        Returns:
            torch.tensor: A `batch_shape x r`-dim Tensor of corresponding point in the original space.
        """
        # Handle the case before PCA is fitted
        if self.component_matrix is None:
            return (torch.rand(self.bounds.shape[0], device=self.device, dtype=self.dtype)
                    * (self.bounds[:, 1] - self.bounds[:, 0]) + self.bounds[:, 0])

        return z @ self.component_matrix[: self.reduced_space_dim] + self.pca_mean + self.data_mean

    def _initialize_model(self, **kwargs):
        """Initialize and fit the Gaussian Process Regression model in the reduced space.

        Args:
            **kwargs: Additional keyword arguments for upcoming development.
        """
        if not len(self.__z_evals):
            return

        # Get bounds for the reduced space
        if len(self.__z_evals) > 1:
            z_min = self.__z_evals.min(dim=0)[0]
            z_max = self.__z_evals.max(dim=0)[0]
            z_range = z_max - z_min
            z_bounds = torch.stack([z_min - 0.1 * z_range, z_max + 0.1 * z_range])
        else:
            # Default bounds if we have only one point
            z_bounds = torch.stack([-torch.ones(self.reduced_space_dim, device=self.device, dtype=self.dtype),
                                    torch.ones(self.reduced_space_dim, device=self.device, dtype=self.dtype)])

        train_obj = torch.from_numpy(self.f_evals).to(device=self.device, dtype=self.dtype)

        # Initialize and fit the GPR
        start_time = perf_counter()
        self.__model_obj = SingleTaskGP(
            self.__z_evals,
            train_obj,
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=self.__z_evals.shape[-1],
                bounds=z_bounds
            )
        )
        mll = ExactMarginalLogLikelihood(self.__model_obj.likelihood, self.__model_obj)
        fit_gpytorch_mll(mll)
        self.timing_logs["SingleTaskGP"].append(perf_counter() - start_time)

    def optimize_acqf_and_get_observation(self) -> Tensor:
        """Optimize the acquisition function in the reduced space.

        Returns:
            Tensor: `batch_shape x r`-dim Tensor - new candidate points in the reduced space.
        """
        if not len(self.__z_evals):
            # If no points in reduced space yet, return a random point
            return torch.randn(1, self.reduced_space_dim, device=self.device, dtype=self.dtype)

        # Create original bounds tensor
        original_bounds = torch.from_numpy(self.bounds).to(device=self.device, dtype=self.dtype)

        # Calculate z bounds
        r = torch.min(torch.abs(original_bounds[:, 0] - original_bounds[:, 1]) / 2)
        z_bounds = (torch.tensor([[-r], [r]], device=self.device, dtype=self.dtype)
                    .expand(-1, self.reduced_space_dim))

        # Set up the acquisition function
        self.acquisition_function = self.acquisition_function_class(
            model=self.__model_obj,
            best_f=self.current_best,
            maximize=self.maximization
        )

        # Create the PEI acquisition function
        self.__pacqf = PenalizedAcqf(
            acquisition_function=self.acquisition_function,
            model=self.__model_obj,
            best_f=self.current_best,
            original_bounds=original_bounds,
            pca_transform_fn=self._transform_points_to_original_space,
            penalty_factor=10.0
        )

        # # Calculate all the 2*d inequality constraints List[Tuple[Tensor, Tensor, float]]
        # inequality_constraints = []
        # lbm = original_bounds[:, 0] - self.pca_mean - self.data_mean
        # ubm = original_bounds[:, 1] - self.pca_mean - self.data_mean
        # for i in range(self.dimension):
        #     inequality_constraints.append((
        #         torch.arange(self.reduced_space_dim),
        #         self.component_matrix[: self.reduced_space_dim, i],
        #         float(lbm[i])
        #     ))
        #     inequality_constraints.append((
        #         torch.arange(self.reduced_space_dim),
        #         -self.component_matrix[: self.reduced_space_dim, i],
        #         float(-ubm[i])
        #     ))
        #
        # foot = perf_counter()
        #
        # batch_initial_conditions = sample_q_batches_from_polytope(
        #     n=self.__torch_config['RAW_SAMPLES']//10,
        #     q=self.__torch_config['BATCH_SIZE'],
        #     bounds=z_bounds,
        #     n_burnin=200,
        #     n_thinning=10,
        #     seed=self.random_seed,
        #     inequality_constraints=inequality_constraints,
        # )
        #
        # print(f"foot: {perf_counter() - foot}")

        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=self.__pacqf,
            bounds=z_bounds,
            q=self.__torch_config['BATCH_SIZE'],
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],
            # batch_initial_conditions=batch_initial_conditions,
            # inequality_constraints=inequality_constraints,
            options={"batch_limit": 1, "maxiter": 500},
            return_best_only=True
        )
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        return candidates

    def get_orthogonal_samples(self, x: Tensor):
        """Get samples in the PCA's orthogonal dimension to the provided candidate.

        Args:
            x: `q x d`-dim Tensor

        Returns:
            A `(ortho_samples + 1) * q x d` Tensor with new candidates.
        """
        # Create original bounds tensor
        original_bounds = torch.from_numpy(self.bounds).to(device=self.device, dtype=self.dtype)

        # Calculate ortho bounds
        r = torch.min(torch.abs(original_bounds[:, 0] - original_bounds[:, 1]) / 2)
        ortho_bounds = (torch.tensor([[-r], [r]], device=self.device, dtype=self.dtype)
                        .expand(-1, self.dimension - self.reduced_space_dim))

        new_x = torch.empty(0)

        for candidate in x:
            # Calculate all the 2*d inequality constraints List[Tuple[Tensor, Tensor, float]]
            inequality_constraints = []
            lbc = original_bounds[:, 0] - candidate
            ubc = original_bounds[:, 1] - candidate
            for i in range(self.dimension):
                inequality_constraints.append((
                    torch.arange(self.dimension - self.reduced_space_dim),
                    self.component_matrix[self.reduced_space_dim:, i],
                    float(lbc[i])
                ))
                inequality_constraints.append((
                    torch.arange(self.dimension - self.reduced_space_dim),
                    -self.component_matrix[self.reduced_space_dim:, i],
                    float(-ubc[i])
                ))

            ortho_lin_comb = sample_q_batches_from_polytope(
                n=self.ortho_samples*4,
                q=1,
                bounds=ortho_bounds,
                n_burnin=max(100, 10 * (self.dimension - self.reduced_space_dim)),
                n_thinning=max(4, (self.dimension - self.reduced_space_dim) // 3),
                inequality_constraints=inequality_constraints,
            ).squeeze(1)

            ortho_part = ortho_lin_comb @ self.component_matrix[self.reduced_space_dim:]

            # Select self.ortho_samples weighted by the proximity to center
            weights = torch.exp(-torch.norm(ortho_part, dim=1) ** 2 / (2 * 0.2**2))
            ortho_part = ortho_part[torch.multinomial(weights, self.ortho_samples, replacement=False)]

            new_x = torch.cat([new_x, candidate.unsqueeze(0),
                               candidate + ortho_part])

        return new_x

    def __repr__(self):
        return super().__repr__()

    def reset(self):
        """Reset the optimizer state."""
        super().reset()

        self.data_mean = None
        self.component_matrix = None
        self.pca_mean = None
        self.explained_variance_ratio = None
        self.reduced_space_dim = None
        self.__z_evals = torch.tensor(0, device=self.device, dtype=self.dtype)

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
            self.__acq_func_class = ExpectedImprovement
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
