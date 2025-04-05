from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import AbstractBayesianOptimizer
import warnings
from sklearn.exceptions import ConvergenceWarning

# Filter warnings from sklearn PCA
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from typing import Union, Callable, Optional, List
from ioh.iohcpp.problem import RealSingleObjective, BBOB
import numpy as np
import torch
import os
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
from sklearn.decomposition import PCA

# Constants for acquisition function names - reusing from Vanilla_BO
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

    def __init__(
            self,
            budget: int,
            n_DoE: int = 0,
            n_components: int = 0,
            var_threshold: float = 0.95,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 43,
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
            **kwargs: Additional keyword arguments for the parent class.
        """
        # Call the superclass
        super().__init__(budget, n_DoE, random_seed, **kwargs)

        # Check the defaults
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
        dtype = torch.double
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

        # Set-up the acquisition function
        self.__acq_func = None
        self.acquistion_function_name = acquisition_function

        # Set PCA parameters
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.pca = None
        self.explained_variance_ratio = None

        # Variables for storing the transformed data
        self.__z_evals = []  # Transformed points in the reduced space

    def __str__(self):
        return "This is an instance of PCA-assisted BO Optimizer"

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
        # Call the superclass to run the initial sampling of the problem
        super().__call__(problem, dim, bounds, **kwargs)

        # Get a default beta (for UCB)
        beta = kwargs.pop("beta", 0.2)

        # Initialize z_evals list with transformed DoE points
        self._transform_points_to_reduced_space()

        # Run the model initialization
        self._initialize_model(**kwargs)

        # Start the optimization loop
        for cur_iteration in range(self.budget - self.n_DoE):
            # Set up the acquisition function
            self.acquisition_function = self.acquisition_function_class(
                model=self.__model_obj,
                best_f=self.current_best,
                maximize=self.maximisation
            )

            new_z = self.optimize_acqf_and_get_observation()

            # Transform the point back to the original space and evaluate
            for _, new_z_arr in enumerate(new_z):
                new_z_numpy = new_z_arr.detach().numpy().ravel()

                # Transform the point from reduced space to original space
                new_x_numpy = self._transform_point_to_original_space(new_z_numpy)

                # Ensure the point is within bounds
                new_x_numpy = np.clip(new_x_numpy, self.bounds[:, 0], self.bounds[:, 1])

                # Append the new points to both spaces
                self.x_evals.append(new_x_numpy)
                self.__z_evals.append(new_z_numpy)

                # Evaluate the function
                new_f_eval = problem(new_x_numpy)

                # Append the function evaluation
                self.f_evals.append(new_f_eval)

                # Increment the number of function evaluations
                self.number_of_function_evaluations += 1

                # Print if we found a better solution
                if ((self.maximisation and new_f_eval > self.current_best) or
                    (not self.maximisation and new_f_eval < self.current_best)) and self.verbose:
                    print(f"Found better solution: {new_f_eval}")
                    print(f"At point: {new_x_numpy}")

            # Assign the new best
            self.assign_new_best()

            # Print best to screen if verbose
            if self.verbose:
                print(
                    f"Current Iteration: {cur_iteration + 1}",
                    f"Current Best: x:{self.x_evals[self.current_best_index]} y:{self.current_best}",
                    flush=True
                )

            # Update PCA with the new point and re-transform all points
            self._transform_points_to_reduced_space()

            # Re-fit the GPR
            self._initialize_model()

        print("Optimization Process finalized!")

    def assign_new_best(self):
        """Assign the new best solution found so far."""
        # Call the super class
        super().assign_new_best()

    def _calculate_weights(self) -> np.ndarray:
        """Calculate rank-based weights for PCA transformation.

        This implements the rank-based weighting scheme described in the paper,
        where better points (with lower function values for minimization) are
        assigned higher weights.

        Returns:
            np.ndarray: Weights for each data point.
        """
        n = len(self.f_evals)

        # Get the ranking of points (1 = best, n = worst)
        if self.maximisation:
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

        # Create a new PCA instance
        if self.n_components > 0:
            # Use fixed number of components
            n_components = min(self.n_components, X.shape[1], X.shape[0] - 1)
            self.pca = PCA(n_components=n_components)
        else:
            # Use variance threshold to determine number of components
            self.pca = PCA(n_components=min(X.shape[1], X.shape[0] - 1))

        # Since PCA.fit() doesn't accept sample_weight directly, we need to implement
        # weighted PCA manually by scaling the data with the square root of weights
        # This is mathematically equivalent to weighted PCA
        weighted_X = X * np.sqrt(weights[:, np.newaxis])
        self.pca.fit(weighted_X)

        # If using variance threshold, select components that explain enough variance
        if self.n_components <= 0:
            explained_var_ratio = self.pca.explained_variance_ratio_
            cumulative_var_ratio = np.cumsum(explained_var_ratio)
            n_components = np.sum(cumulative_var_ratio <= self.var_threshold) + 1
            n_components = min(n_components, len(explained_var_ratio))

            # Ensure we always keep at least one component
            n_components = max(1, n_components)

            # Create a new PCA with the determined number of components
            self.pca = PCA(n_components=n_components, random_state=self.random_seed)

            # We need to center the data first
            X_centered = X - np.mean(X, axis=0)

            # Apply the weighted PCA approach
            weighted_X = X_centered * np.sqrt(weights[:, np.newaxis])

            # Add a small amount of noise to avoid numerical issues
            noise = np.random.normal(0, 1e-8, size=weighted_X.shape)
            weighted_X += noise

            self.pca.fit(weighted_X)

        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        if self.verbose:
            print(f"Using {self.pca.n_components_} principal components with "
                  f"{np.sum(self.pca.explained_variance_ratio_) * 100:.2f}% explained variance")

        # Transform all points to the reduced space
        # We need to transform the original (unweighted) data
        Z = self.pca.transform(X)
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
        x = self.pca.inverse_transform(z_2d)

        # Add a small random perturbation to avoid getting stuck
        # This helps explore the space more effectively
        noise = np.random.normal(0, 0.01, size=x.shape)
        x = x + noise

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

        # Convert bounds array to Torch
        bounds_torch = torch.from_numpy(z_bounds.transpose()).double()

        # Convert the initial values to Torch Tensors
        train_z = np.array(self.__z_evals).reshape((-1, len(self.__z_evals[0])))
        train_z = torch.from_numpy(train_z).double()

        train_obj = np.array(self.f_evals).reshape((-1, 1))
        train_obj = torch.from_numpy(train_obj).double()

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

    def optimize_acqf_and_get_observation(self) -> Tensor:
        """Optimize the acquisition function in the reduced space.

        Returns:
            Tensor: New candidate point in the reduced space.
        """
        if not self.__z_evals:
            # If no points in reduced space yet, return a random point
            return torch.from_numpy(np.random.randn(1, 1)).double()

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
        bounds_torch = torch.from_numpy(z_bounds.transpose()).double()

        # Try several times with different settings if needed
        for attempt in range(3):
            try:
                # Optimize
                candidates, acq_value = optimize_acqf(
                    acq_function=self.acquisition_function,
                    bounds=bounds_torch,
                    q=1,
                    num_restarts=self.__torch_config['NUM_RESTARTS'] * (attempt + 1),
                    raw_samples=self.__torch_config['RAW_SAMPLES'] * (attempt + 1),
                    options={"batch_limit": 5, "maxiter": 200 * (attempt + 1)},
                )

                # Check if acquisition value is significantly different from zero
                if acq_value.item() > 1e-6:
                    break

                # If we're on the last attempt, add some random noise
                if attempt == 2:
                    noise = torch.randn(candidates.shape, device=candidates.device) * 0.1
                    candidates = candidates + noise

            except Exception as e:
                if self.verbose:
                    print(f"Optimization attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:
                    # If all attempts fail, generate a random point
                    d = bounds_torch.shape[1]
                    random_point = torch.rand(1, d, device=bounds_torch.device)
                    # Scale to bounds
                    candidates = bounds_torch[0] + random_point * (bounds_torch[1] - bounds_torch[0])

        # Observe new values
        new_z = candidates.detach()
        new_z = new_z.reshape(shape=((1, -1))).detach()

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

    @property
    def acquistion_function_name(self) -> str:
        """Get the acquisition function name."""
        return self.__acquisition_function_name

    @acquistion_function_name.setter
    def acquistion_function_name(self, new_name: str) -> None:
        """Set the acquisition function name.

        Args:
            new_name (str): Name of the acquisition function.

        Raises:
            ValueError: If the acquisition function name is not recognized.
        """
        # Remove some spaces
        new_name = new_name.strip()

        # Start with a dummy variable
        dummy_var = ""

        # Check in the reduced
        if new_name in ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS:
            # Assign the name
            dummy_var = ALLOWED_SHORTHAND_ACQUISITION_FUNCTION_STRINGS[new_name]
        else:
            if new_name.lower() in ALLOWED_ACQUISITION_FUNCTION_STRINGS:
                dummy_var = new_name
            else:
                raise ValueError("Oddly defined name")

        self.__acquisition_function_name = dummy_var
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
                "Cannot assign the acquisition function as this does not inherit from the class `AnalyticAcquisitionFunction`",
                name="acquisition_function",
                obj=self.__acq_func
            )
