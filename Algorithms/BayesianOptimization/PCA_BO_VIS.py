"""A standard PCA-BO implementation."""

from typing import Union, Callable, Optional
import os
from time import perf_counter
import numpy as np
import torch
from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    AnalyticAcquisitionFunction
)
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel
from sklearn.decomposition import PCA
from ioh.iohcpp.problem import RealSingleObjective

from Algorithms.utils.visualization_utils import Visualizer
from Algorithms.utils.tqdm_write_stream import redirect_stdout_to_tqdm, restore_stdout
from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import AbstractBayesianOptimizer

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # Filter warnings from sklearn PCA

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
    """PCA-assisted Bayesian Optimization implementation.

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
        __acq_func (AnalyticAcquisitionFunction): The acquisition function.
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
            visualize: bool = False,
            **kwargs
    ):
        """Initialize the PCA-BO optimizer with the given parameters.

        Args:
            budget (int): Maximum number of function evaluations.
            n_DoE (int, optional): Number of initial design of experiments samples. Defaults to 0.
            n_components (int, optional): Number of principal components to use. If 0, determined
                                        by var_threshold. Defaults to 0.
            var_threshold (float, optional): Variance threshold for selecting components. Defaults to 0.95.
            acquisition_function (str, optional): Acquisition function name. Defaults to "expected_improvement".
            random_seed (int, optional): Random seed for reproducibility. Defaults to 43.
            visualize (bool, optional): Whether to generate visualizations. Defaults to False.
            **kwargs: Additional keyword arguments for the parent class.
        """
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

        # Set PCA parameters
        self.n_components = n_components
        self.var_threshold = var_threshold

        self.data_mean = None
        self.pca = None
        self.component_matrix = None
        self.explained_variance_ratio = None
        self.reduced_space_dim_num = None

        # Set up visualization
        self.visualize = visualize
        if self.visualize:
            self.visualizer = Visualizer()
            self.iteration = 0
            self.obj_function = None

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

        # Store the objective function for visualization if needed
        if self.visualize:
            self.obj_function = problem
            self.iteration = 0

            # Visualize initial design
            X = np.vstack(self.x_evals)
            y = np.array(self.f_evals).reshape(-1, 1)
            self.visualizer.visualize_initial_design(X, y, self.dimension, self.bounds)

        # Start the optimization loop
        for cur_iteration in range(self.budget - self.n_DoE):
            if self.number_of_function_evaluations >= self.budget:
                break

            # Initialize z_evals list with transformed points
            self._transform_points_to_reduced_space()

            # Initialize and fit the GPR model
            self._initialize_model(**kwargs)

            # Set up the acquisition function
            self.acquisition_function = self.acquisition_function_class(
                model=self.__model_obj,
                best_f=self.current_best,
                maximize=self.maximization
            )

            new_z = self.optimize_acqf_and_get_observation()

            # Visualize acquisition function if enabled
            if self.visualize and hasattr(self, 'test_points') and hasattr(self, 'last_acquisition_values'):
                z_array = np.vstack(self.__z_evals)
                # Get the latest point index in the reduced space
                latest_idx = len(self.__z_evals) - 1 if len(self.__z_evals) > 0 else None

                # Project bounds to PCA space
                pca_bounds = self._project_bounds_to_pca_space()

                self.visualizer.visualize_acquisition(
                    z_array,
                    self.last_acquisition_values,
                    cur_iteration,
                    self.test_points,
                    pca_bounds,
                    latest_idx=latest_idx
                )

                # Visualize Gaussian Process model
                y_array = np.array(self.f_evals).reshape(-1, 1)
                # Use the same latest_idx as for the acquisition function
                self.visualizer.visualize_gaussian_process(
                    self.__model_obj,
                    z_array,
                    y_array,
                    self.test_points,
                    cur_iteration,
                    pca_bounds,
                    latest_idx=latest_idx
                )

            # Transform the points back to the original space and evaluate
            for _, new_z_arr in enumerate(new_z):
                if self.number_of_function_evaluations >= self.budget:
                    break

                new_z_numpy = new_z_arr.detach().numpy().ravel()

                # Transform the point from reduced space to original space
                new_x_numpy = self._transform_point_to_original_space(new_z_numpy)

                is_outside_bounds = not np.all(new_x_numpy >= self.bounds[:,0]) or not np.all(new_x_numpy <= self.bounds[:,1])
                # Ensure the point is within bounds
                if is_outside_bounds:
                    if self.verbose:
                        print(f"Warning: PCA transformed point {new_x_numpy} was out of bounds, giving penalty")

                # Append the new points to both spaces
                self.x_evals.append(new_x_numpy)
                self.__z_evals.append(new_z_numpy)

                # Evaluate the function # TODO give correct penalty if outside bounds
                if is_outside_bounds:
                    new_f_eval = problem(new_x_numpy) + (-1e5 if self.maximization else 1e5)
                else:
                    new_f_eval = problem(new_x_numpy)

                if self._pbar is not None:
                    self._pbar.update(1)

                # Append the function evaluation
                self.f_evals.append(new_f_eval)

                # Increment the number of function evaluations
                self.number_of_function_evaluations += 1

                # Print if found a better solution
                if ((self.maximization and new_f_eval > self.current_best) or
                    (not self.maximization and new_f_eval < self.current_best)) and self.verbose:
                    print(f"Found better solution: {new_f_eval}")
                    print(f"At point: {new_x_numpy}")

            # Assign the new best
            self.assign_new_best()

            # Visualize optimization progress if enabled
            # Visualize PCA step if enabled
            if self.visualize and self.dimension == 2:
                X = np.vstack(self.x_evals)
                #weights = self._calculate_weights()
                # Get the latest point index
                latest_idx = len(self.x_evals) - 1 if len(self.x_evals) > 0 else None
                self.visualizer.visualize_pca_step(X, self.pca, self.f_evals, self.data_mean, self.component_matrix,
                                                   self.obj_function,
                                                   cur_iteration, self.bounds, latest_idx=latest_idx)
            if self.visualize:
                self.visualizer.visualize_optimization_progress(
                    self.f_evals,
                    cur_iteration,
                    self.maximization
                )

            # Print best to screen if verbose
            if self.verbose:
                print(
                    # f"Iteration: {cur_iteration + 1}",
                    f"Evaluations: {self.number_of_function_evaluations}/{self.budget}",
                    f"Best: x:{self.x_evals[self.current_best_index]} y:{self.current_best}",
                    flush=True
                )

        if self.visualize:
            self.visualizer.save_all_animations()

        if self.verbose:
            print("Optimization Process finalized!")

        # Restore initial randomness states
        self.restore_random_states()

        if self._pbar is not None:
            restore_stdout(original_stdout)

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

        if self.visualize and X.shape[1] == 2:  # Only visualize for 2D problems
            self.visualizer.visualize_weighted_transform(X, weights, self.pca)


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

        # Reshape to 2D for sklearn
        z_2d = z.reshape(1, -1)

        # Transform back to original space
        x = self.pca.inverse_transform(z_2d) + self.data_mean

        # # Add a small random perturbation to avoid getting stuck
        # # This helps explore the space more effectively
        # noise = np.random.normal(0, 0.01, size=x.shape)
        # x = x + noise

        return x.ravel()

    def _project_bounds_to_pca_space(self) -> np.ndarray:
        """Project the original bounds into the PCA space.

        This method creates a grid of points along the boundaries of the original space,
        transforms them into the PCA space, and then finds the min/max values in each dimension.

        Returns:
            np.ndarray: Bounds in the PCA space with shape (n_components, 2)
        """
        assert self.pca is not None

        # Create a list to store all boundary points
        boundary_points = []

        n_dims = self.bounds.shape[0]

        # For each dimension, create points along the min and max bounds
        for dim in range(n_dims):
            # Number of points to sample along each boundary
            n_samples = 1000

            # For the min bound of this dimension
            for i in range(n_samples):
                point = np.zeros(n_dims)
                # Set this dimension to its min bound
                point[dim] = self.bounds[dim, 0]
                # Set other dimensions to random values within their bounds
                for other_dim in range(n_dims):
                    if other_dim != dim:
                        point[other_dim] = np.random.uniform(
                            self.bounds[other_dim, 0],
                            self.bounds[other_dim, 1]
                        )
                boundary_points.append(point)

            # For the max bound of this dimension
            for i in range(n_samples):
                point = np.zeros(n_dims)
                # Set this dimension to its max bound
                point[dim] = self.bounds[dim, 1]
                # Set other dimensions to random values within their bounds
                for other_dim in range(n_dims):
                    if other_dim != dim:
                        point[other_dim] = np.random.uniform(
                            self.bounds[other_dim, 0],
                            self.bounds[other_dim, 1]
                        )
                boundary_points.append(point)

        # Convert to numpy array
        boundary_points = np.vstack(boundary_points)

        # Transform boundary points to PCA space
        if len(boundary_points) > 0:
            # Transform to PCA space
            pca_boundary_points = self.pca.transform(boundary_points)

            # Find min and max values in each PCA dimension
            pca_bounds = np.zeros((self.pca.n_components_, 2))
            pca_bounds[:, 0] = np.min(pca_boundary_points, axis=0)
            pca_bounds[:, 1] = np.max(pca_boundary_points, axis=0)

            return pca_bounds

        return None

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

        # Convert bounds array to Torch
        bounds_torch = torch.from_numpy(z_bounds.transpose()).float()

        # Convert the initial values to Torch Tensors
        train_z = np.array(self.__z_evals).reshape((-1, len(self.__z_evals[0])))
        train_z = torch.from_numpy(train_z).float()

        train_obj = np.array(self.f_evals).reshape((-1, 1))
        train_obj = torch.from_numpy(train_obj).float()

        start_time = perf_counter()
        train_Yvar = torch.full_like(train_obj, 1e-6)

        # Initialize the GP model
        self.__model_obj = SingleTaskGP(
            train_z,
            train_obj,
            train_Yvar,
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=train_z.shape[-1],
                bounds=bounds_torch
            )
        )
        self.timing_logs["SingleTaskGP"].append(perf_counter() - start_time)

        # Fit the model using exact marginal log likelihood
        mll = ExactMarginalLogLikelihood(self.__model_obj.likelihood, self.__model_obj)
        fit_gpytorch_mll(mll)

    def optimize_acqf_and_get_observation(self) -> Tensor:
        """Optimize the acquisition function in the reduced space.

        Returns:
            Tensor: New candidate point in the reduced space.
        """
        if not self.__z_evals:
            # If no points in reduced space yet, return a random point
            return torch.from_numpy(np.random.randn(1, 1)).float()

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

        # Convert to torch tensor
        bounds_torch = torch.from_numpy(z_bounds.transpose()).float()

        # Create a grid of test points for visualization if enabled
        if self.visualize:
            if z_array.shape[1] == 1:
                # For 1D, create a line
                test_x = np.linspace(z_bounds[0, 0], z_bounds[0, 1], 100).reshape(-1, 1)
            else:
                # For 2D or higher, create a grid or sample points
                if z_array.shape[1] == 2:
                    # For 2D, create a grid
                    x = np.linspace(z_bounds[0, 0], z_bounds[0, 1], 20)
                    y = np.linspace(z_bounds[1, 0], z_bounds[1, 1], 20)
                    xx, yy = np.meshgrid(x, y)
                    test_x = np.column_stack((xx.ravel(), yy.ravel()))
                else:
                    # For higher dimensions, use Latin Hypercube Sampling
                    from pyDOE import lhs
                    test_x = lhs(z_array.shape[1], samples=400)
                    # Scale to bounds
                    for i in range(z_array.shape[1]):
                        test_x[:, i] = z_bounds[i, 0] + test_x[:, i] * (z_bounds[i, 1] - z_bounds[i, 0])

            self.test_points = torch.tensor(test_x, dtype=torch.float32)

            # Evaluate acquisition function at test points
            with torch.no_grad():
                self.last_acquisition_values = self.acquisition_function(self.test_points.unsqueeze(-2)).numpy()

        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=bounds_torch,
            q=1,  # self.__torch_config['BATCH_SIZE'],
            num_restarts=self.__torch_config['NUM_RESTARTS'],
            raw_samples=self.__torch_config['RAW_SAMPLES'],  # Used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )
        print(candidates, self._transform_point_to_original_space(candidates[0]))
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        self.plot_acqf_table(bounds_torch)

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

        # Reset visualization-related attributes if visualization is enabled
        if hasattr(self, 'visualize') and self.visualize:
            self.visualizer = Visualizer()
            self.iteration = 0
            if hasattr(self, 'test_points'):
                delattr(self, 'test_points')
            if hasattr(self, 'last_acquisition_values'):
                delattr(self, 'last_acquisition_values')

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
            # Assign the name
            self.__acquisition_function_name = ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS[new_name]
        else:
            if new_name.lower() in ALLOWED_ACQUISITION_FUNCTION_STRINGS:
                self.__acquisition_function_name = new_name
            else:
                raise ValueError("Oddly defined name")

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
    def plot_acqf_table(self, bounds_torch):
        pass
