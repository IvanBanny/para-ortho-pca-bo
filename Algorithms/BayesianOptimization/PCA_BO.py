"""A standard PCA-BO implementation."""

import os
from typing import Union, Callable, Optional, Dict, Any
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from gpytorch.kernels import MaternKernel

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

from sklearn.decomposition import PCA
from ioh.iohcpp.problem import RealSingleObjective

from Algorithms.utils.tqdm_write_stream import redirect_stdout_to_tqdm, restore_stdout
from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import AbstractBayesianOptimizer
from Algorithms.BayesianOptimization.PenalizedAcqf import PenalizedAcqf
from Algorithms.utils.vis_utils import PCABOVisualizer

import warnings
from sklearn.exceptions import ConvergenceWarning
from botorch.exceptions.warnings import NumericsWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Filter warnings from sklearn PCA
warnings.filterwarnings("ignore", category=NumericsWarning)  # Filter warnings from EI
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

    Attributes:
        n_components (int): Number of principal components to use.
        pca (PCA): The PCA transformer object.
        explained_variance_ratio (numpy.ndarray): Explained variance ratio of each component.
        __torch_config (dict): Configuration for the PyTorch operations.
        __parallel_optimizer (ParallelOptimizer): The parallel optimizer for multi-GPU processing.
    """

    TIME_PROFILES = ["SingleTaskGP", "optimize_acqf", "pca"]

    def __init__(
            self,
            budget: int,
            n_DoE: int = 0,
            n_components: int = 0,
            var_threshold: float = 0.95,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 43,
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

        if torch_config is None:
            torch_config = {
                "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                "dtype": torch.float,
                "BATCH_SIZE": 1,
                "NUM_RESTARTS": 20,
                "RAW_SAMPLES": 1024
            }

        # Set up the acquisition function
        self.__acq_func_class = None
        self.acquisition_function_name = acquisition_function
        self.__acq_func = None
        self.__pacqf = None

        self.__torch_config = torch_config

        # Set device and dtype from torch_config
        self.device = self.__torch_config["device"]
        self.dtype = self.__torch_config["dtype"]

        if self.verbose:
            print(f"Using device: {self.device}")

        # Ensure CUDA is available if device is CUDA
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA specified but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
            self.__torch_config["device"] = self.device

        # Initialize parallel optimizer for multi-GPU processing
        self.__parallel_optimizer = None

        # Set PCA parameters
        self.n_components = n_components
        self.var_threshold = var_threshold

        self.data_mean = None
        self.pca = None
        self.component_matrix = None
        self.explained_variance_ratio = None
        self.reduced_space_dim_num = None

        # Variables for storing the transformed data
        self.__z_evals = []  # Transformed points in the reduced space

        # Initialize visualizer if requested
        self.visualize = visualize
        self.visualizer = None
        if self.visualize:
            self.visualizer = PCABOVisualizer(output_dir=vis_output_dir)

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
        if self._pbar is not None:
            original_stdout = redirect_stdout_to_tqdm(self._pbar)

        # Save current randomness states and impose seed
        self.impose_random_seed()

        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)

        if self._pbar is not None:
            self._pbar.update(self.n_DoE)

        # Start the optimization loop
        while self.number_of_function_evaluations < self.budget:
            # Initialize z_evals list with transformed points
            self._transform_points_to_reduced_space()

            # Initialize and fit the GPR model
            self._initialize_model(**kwargs)

            new_z = self.optimize_acqf_and_get_observation()

            # Transform the points back to the original space and evaluate
            for _, new_z_arr in enumerate(new_z):
                if self.number_of_function_evaluations >= self.budget:
                    break

                # Convert tensor to numpy for processing
                new_z_numpy = new_z_arr.cpu().detach().numpy().ravel()

                # Transform the point from reduced space to original space
                new_x_numpy = self._transform_point_to_original_space(new_z_numpy)

                is_outside_bounds = (not np.all(new_x_numpy >= self.bounds[:, 0]) or
                                     not np.all(new_x_numpy <= self.bounds[:, 1]))
                # Ensure the point is within bounds
                if is_outside_bounds:
                    if self.verbose:
                        print(f"Warning: PCA transformed point {new_x_numpy} was out of bounds")
                    # new_x_numpy = np.clip(new_x_numpy, self.bounds[:, 0], self.bounds[:, 1])

                # Append the new points to both spaces
                self.x_evals.append(new_x_numpy)
                self.__z_evals.append(new_z_numpy)

                new_f_eval = problem(new_x_numpy)

                print(f"Sampled:\n"
                      f"    x: {new_x_numpy.tolist()}\n"
                      f"    y: {new_f_eval}")

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
                    f"Evals: {self.number_of_function_evaluations}/{self.budget}",
                    f"Best:\n"
                    f"    x: {self.x_evals[self.current_best_index]}\n"
                    f"    y: {self.current_best}",
                    flush=True
                )

            # Save logs
            if self.save_logs:
                acqf_values_log, penalty_log, pei_values_log = self.pacqf.log_forward(
                    torch.tensor(self.__z_evals).to(device=self.device, dtype=self.dtype).unsqueeze(1)
                )
                df = pd.DataFrame({
                    "x": self.x_evals,
                    "z": self.__z_evals,
                    "p": self.f_evals,
                    "acqf": acqf_values_log.detach().numpy(),
                    "penalty": penalty_log.detach().numpy(),
                    "pacqf": pei_values_log.detach().numpy(),
                })

                run_log_dir = os.path.join(
                    self.log_dir,
                    f"log_{problem.meta_data.problem_id}_{problem.meta_data.instance}"
                )
                os.makedirs(run_log_dir, exist_ok=True)
                df.to_csv(os.path.join(run_log_dir, f"{self.number_of_function_evaluations}.csv"), index=False)

            # Create visualizations
            if self.visualize and self.visualizer is not None:
                self.visualizer.visualize_pcabo(torch.tensor(self.x_evals),
                                                torch.tensor(self.x_evals[self.current_best_index]),
                                                torch.tensor(self.bounds), torch.tensor(new_x_numpy).unsqueeze(0),
                                                self.component_matrix, self.data_mean + self.pca.mean_, problem,
                                                margin=0.1)

        # Save the visualization gifs if it was enabled
        if self.visualize and self.visualizer is not None:
            self.visualizer.save_gifs(
                postfix=f"{problem.meta_data.problem_id}_{problem.meta_data.instance}",
                duration=1000
            )

        if self.verbose:
            print("Optimization Process finalized!")

        # Restore initial randomness states
        self.restore_random_states()

        if self._pbar is not None:
            restore_stdout(original_stdout)

    def assign_new_best(self):
        """Assign the new best solution found so far."""
        super().assign_new_best()

    def _calculate_weights(self) -> np.ndarray:
        """Calculate rank-based weights for PCA transformation.

        This implements the rank-based weighting scheme described in the original PCA-BO paper,
        where better points (with lower function values for minimization) are assigned higher weights.

        Returns:
            np.ndarray: Weights for each data point.
        """
        n = len(self.f_evals)

        # Get the ranking of points (1 = best, n = worst)
        if self.maximization:
            # For maximization, higher values are better
            ranks = np.argsort(np.argsort(-np.array(self.f_evals))) + 1
        else:
            # For minimization, lower values are better
            ranks = np.argsort(np.argsort(np.array(self.f_evals))) + 1

        # Calculate pre-weights
        pre_weights = np.log(n) - np.log(ranks)

        # Normalize weights
        weights = pre_weights / pre_weights.sum()

        return weights

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
                self.__z_evals = [np.zeros(1)]
            return

        # Stack points into a matrix
        X = np.vstack(self.x_evals)

        # Calculate weights
        weights = self._calculate_weights()

        # Center the data first
        self.data_mean = np.mean(X, axis=0)
        X_centered = X - self.data_mean

        # Apply the weights
        # Note: applying a square root here, which makes more sense, but isn't mentioned in the original paper
        weighted_X = X_centered * np.sqrt(weights[:, np.newaxis])

        # Add a small amount of noise to avoid numerical issues
        noise = np.random.normal(0, 1e-8, size=weighted_X.shape)
        weighted_X += noise

        # Initialize and fit pca
        self.pca = PCA()

        start_time = perf_counter()
        self.pca.fit(weighted_X)
        self.timing_logs["pca"].append(perf_counter() - start_time)

        self.component_matrix = np.copy(self.pca.components_)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        n_components = self.n_components
        if n_components <= 0:
            # If using variance threshold - select components that explain enough variance
            cumulative_var_ratio = np.cumsum(self.explained_variance_ratio)
            n_components = np.sum(cumulative_var_ratio <= self.var_threshold) + 1
            n_components = max(1, min(n_components, len(self.explained_variance_ratio)))

        self.reduced_space_dim_num = n_components

        # Clip the component matrix
        self.pca.components_ = self.pca.components_[:self.reduced_space_dim_num]

        if self.verbose:
            print(f"\nUsing {n_components} principal components with "
                  f"{np.sum(self.pca.explained_variance_ratio_[:n_components]) * 100:.2f}% explained variance")

        # Transform all points to the reduced space
        # We need to transform the centered, but unweighted data
        Z = self.pca.transform(X_centered)
        self.__z_evals = [Z[i, :] for i in range(Z.shape[0])]

    def _transform_point_to_original_space(self, z: np.ndarray) -> np.ndarray:
        """Transform a point from reduced space back to the original space.

        Args:
            z (np.ndarray): Point in the reduced space.

        Returns:
            np.ndarray: Corresponding point in the original space.
        """
        # Handle the case before PCA is fitted
        if self.pca is None:
            return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])

        # Reshape to 2D for sklearn
        z_2d = z.reshape(1, -1)

        # Transform back to original space
        x = self.pca.inverse_transform(z_2d) + self.data_mean

        # # Add a small random perturbation to avoid getting stuck
        # # This helps explore the space more effectively
        # noise = np.random.normal(0, 0.01, size=x.shape)
        # x = x + noise

        return x.ravel()

    def _initialize_model(self, **kwargs):
        """Initialize/fit the Gaussian Process Regression model in the reduced space.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if not self.__z_evals:
            return

        # Get bounds for the reduced space
        if len(self.__z_evals) > 1:
            z_array = np.vstack(self.__z_evals)
            z_min = np.min(z_array, axis=0)
            z_max = np.max(z_array, axis=0)
            z_range = z_max - z_min
            # Add some padding
            z_bounds = np.vstack([z_min - 0.1 * z_range, z_max + 0.1 * z_range]).T
        else:
            # Default bounds if we have only one point
            z_bounds = np.vstack([-np.ones(self.pca.n_components_),
                                  np.ones(self.pca.n_components_)]).T

        # Convert bounds array to Torch and move to device
        bounds_torch = torch.from_numpy(z_bounds.transpose()).to(device=self.device, dtype=self.dtype)

        # Convert the initial values to Torch Tensors and move to device
        train_z = np.array(self.__z_evals).reshape((-1, len(self.__z_evals[0])))
        train_z = torch.from_numpy(train_z).to(device=self.device, dtype=self.dtype)

        train_obj = np.array(self.f_evals).reshape((-1, 1))
        train_obj = torch.from_numpy(train_obj).to(device=self.device, dtype=self.dtype)

        start_time = perf_counter()
        self.__model_obj = SingleTaskGP(
            train_z,
            train_obj,
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=train_z.shape[-1],
                bounds=bounds_torch
            )
        )
        self.timing_logs["SingleTaskGP"].append(perf_counter() - start_time)

    def optimize_acqf_and_get_observation(self) -> Tensor:
        """Optimize the acquisition function in the reduced space.

        Returns:
            Tensor: New candidate point in the reduced space.
        """
        if not self.__z_evals:
            # If no points in reduced space yet, return a random point
            return torch.randn(1, 1, device=self.device, dtype=self.dtype)

        # Define the function to transform points from reduced space to original space
        def pca_transform_fn(z):
            # Handle batched inputs
            z_np = z.cpu().detach().numpy()

            # Quick path for single points
            if len(z_np.shape) == 1:
                return torch.tensor([self._transform_point_to_original_space(z_np)],
                                    device=self.device, dtype=self.dtype)

            # Process each point individually
            x_list = []
            for i in range(z_np.shape[0]):
                x_i = self._transform_point_to_original_space(z_np[i])
                x_list.append(x_i)

            # Stack results and convert back to tensor
            x_batch = np.vstack(x_list)
            return torch.tensor(x_batch, device=self.device, dtype=self.dtype)

        # Create original bounds tensor
        original_bounds = torch.tensor(self.bounds, device=self.device, dtype=self.dtype)

        # Calculate z bounds
        r = torch.min(torch.abs(original_bounds[:, 0] - original_bounds[:, 1]) / 2)
        z_bounds = (torch.tensor([[-r], [r]], device=self.device, dtype=self.dtype)
                    .expand(-1, self.reduced_space_dim_num))

        # Set up the acquisition function
        self.acquisition_function = self.acquisition_function_class(
            model=self.__model_obj,
            best_f=self.current_best,
            maximize=self.maximization
        )

        # Create the PEI acquisition function
        self.pacqf = PenalizedAcqf(
            acquisition_function=self.acquisition_function,
            model=self.__model_obj,
            best_f=self.current_best,
            original_bounds=original_bounds,
            pca_transform_fn=pca_transform_fn,
            penalty_factor=1000.0
        )

        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=self.pacqf,
            bounds=z_bounds,
            q=self.__torch_config['BATCH_SIZE'],
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],
            options={"batch_limit": 5, "maxiter": 500},
            return_best_only=True
        )
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        return candidates

    def __repr__(self):
        return super().__repr__()

    def reset(self):
        """Reset the optimizer state."""
        super().reset()
        self.__z_evals = []
        self.pca = None
        self.explained_variance_ratio = None

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
