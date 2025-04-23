"""A standard PCA-BO implementation."""

from typing import Union, Callable, Optional, Dict, Any
from time import perf_counter
import numpy as np
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel
from sklearn.decomposition import PCA
from ioh.iohcpp.problem import RealSingleObjective

from Algorithms.utils.tqdm_write_stream import redirect_stdout_to_tqdm, restore_stdout
from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import AbstractBayesianOptimizer
from Algorithms.BayesianOptimization.PEI import PenalizedExpectedImprovement

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Filter warnings from sklearn PCA


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
            random_seed: int = 43,
            torch_config: Optional[Dict[str, Any]] = None,
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
        for cur_iteration in range(self.budget - self.n_DoE):
            if self.number_of_function_evaluations >= self.budget:
                break

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
                        print(f"Warning: PCA transformed point {new_x_numpy} was out of bounds, clipping to bounds")
                    new_x_numpy = np.clip(new_x_numpy, self.bounds[:, 0], self.bounds[:, 1])

                # Append the new points to both spaces
                self.x_evals.append(new_x_numpy)
                self.__z_evals.append(new_z_numpy)

                new_f_eval = problem(new_x_numpy)

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
            print(f"Using {n_components} principal components with "
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

        # Get bounds for the reduced space
        z_array = np.vstack(self.__z_evals)
        z_min = np.min(z_array, axis=0)
        z_max = np.max(z_array, axis=0)
        z_range = z_max - z_min

        # Add substantial padding to allow more exploration
        # This is especially important in the early iterations
        z_bounds = np.vstack([z_min - 0.5 * z_range, z_max + 0.5 * z_range]).T

        # Ensure bounds aren't too tight early in the optimization
        min_range = 0.1
        for i in range(z_bounds.shape[0]):
            if z_bounds[i, 1] - z_bounds[i, 0] < min_range:
                mid = (z_bounds[i, 1] + z_bounds[i, 0]) / 2
                z_bounds[i, 0] = mid - min_range / 2
                z_bounds[i, 1] = mid + min_range / 2

        # Convert to torch tensor and move to device
        bounds_torch = torch.from_numpy(z_bounds.transpose()).to(device=self.device, dtype=self.dtype)

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

        # Create the PEI acquisition function
        acq_function = PenalizedExpectedImprovement(
            model=self.__model_obj,
            best_f=self.current_best,
            original_bounds=original_bounds,
            pca_transform_fn=pca_transform_fn,
            maximize=self.maximization,
            penalty_factor=1000.0
        )

        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds_torch,
            q=self.__torch_config['BATCH_SIZE'],
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],
            options={"batch_limit": 5, "maxiter": 500}
        )
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        # Observe new values
        new_z = candidates.detach()
        new_z = new_z.reshape(shape=(1, -1)).detach()

        return new_z

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
