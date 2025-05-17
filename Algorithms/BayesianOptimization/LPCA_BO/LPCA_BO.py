"""
Author: Adela Greganova
This file is a modified version of the PCA_BO.py implementation that I made together with Ivan Banny.

The trust region logic follows the implementation of https://github.com/uber-research/TuRBO/blob/master/turbo/turbo_1.py .

"""
import math
# noinspection DuplicatedCode

import os
from typing import Union, Callable, Optional, Dict, Any
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from botorch import fit_gpytorch_mll
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.models import gp
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

from .util import *

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


# Algorithm 2: PCA-BO and Trust Region Integration Optimization Loop
#
# while the budget is not exhausted:
#     if the trust region is too small:
#         Restart the process
#     else:
#         - Center data points around the trust region origin and apply weights
#         - Fit PCA to the existing data within the trust region
#         - Project the unweighted points into the PCA-reduced space
#         - Fit a Gaussian Process (GP) model in the reduced space
#         - Optimize the acquisition function and select a point within the
#           reduced trust region that:
#           - maps back inside the original space
#           - has the highest Expected Improvement (EI) value (use PEI)
#         - Evaluate the objective function at the selected point
#         - Adjust the trust region

class LPCA_BO(AbstractBayesianOptimizer):
    """PCA-assisted Bayesian Optimization with Trust Regions (TuRBO-1 algorithm) implementation.

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
    """
    """The TuRBO-1 algorithm.

        Parameters
        ----------
        f : function handle
        lb : Lower variable bounds, numpy.array, shape (d,).
        ub : Upper variable bounds, numpy.array, shape (d,).
        n_init : Number of initial points (2*dim is recommended), int.
        max_evals : Total evaluation budget, int.
        batch_size : Number of points in each batch, int.
        verbose : If you want to print information about the optimization progress, bool.
        use_ard : If you want to use ARD for the GP kernel.
        max_cholesky_size : Largest number of training points where we use Cholesky, int
        n_training_steps : Number of training steps for learning the GP hypers, int
        min_cuda : We use float64 on the CPU if we have this or fewer datapoints
        device : Device to use for GP fitting ("cpu" or "cuda")
        dtype : Dtype to use for GP fitting ("float32" or "float64")

        Example usage:
            turbo1 = Turbo1(f=f, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals)
            turbo1.optimize()  # Run optimization
            X, fX = turbo1.X, turbo1.fX  # Evaluated points
        """
    TIME_PROFILES = ["SingleTaskGP", "optimize_acqf", "pca"]

    def __init__(
            self,
            budget: int,
            n_DoE: int = 0, #n_init in turbo
            n_components: int = 0,
            var_threshold: float = 0.95,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 43,
            torch_config: Optional[Dict[str, Any]] = None,
            visualize: bool = False,
            vis_output_dir: str = "./visualizations",
            save_logs: bool = False,
            log_dir: str = "./logs",
            **kwargs,
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
                "device": torch.device("cpu"),
                "dtype": torch.float,
                "BATCH_SIZE": 1,
                "NUM_RESTARTS": 20,
                "RAW_SAMPLES": 1024
            }

        # Set up the acquisition function
        self.__acq_func_class = None
        self.acquisition_function_name = acquisition_function
        self.__acq_func = None

        self.__torch_config = torch_config

        # Set device and dtype from torch_config
        self.dtype = self.__torch_config["dtype"]

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

        def __call__(
                self,
                problem: Union[RealSingleObjective, Callable],
                dim: Optional[int] = -1,
                bounds: Optional[np.ndarray] = None,
                **kwargs
        ) -> None:
            """Execute the optimization algorithm with trust region approach.

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

            # Initialize trust region parameters
            self._initialize_trust_region()

            # Start the optimization loop
            while self.number_of_function_evaluations < self.budget and self.length >= self.length_min:
                # Initialize z_evals list with transformed points
                self._transform_points_to_reduced_space()

                # Initialize and fit the GPR model
                self._initialize_model(**kwargs)

                # Store previous best function value for trust region adjustment
                prev_best_value = self.current_best if hasattr(self, 'current_best') else float('inf')

                # Get the next point using acquisition function optimization
                sampled_z = self.optimize_acqf_and_get_observation()

                # Track function values for this batch to adjust trust region
                batch_f_vals = []

                # Transform the point back to the original space and evaluate
                for _, new_z_arr in enumerate(sampled_z):

                    # Convert tensor to numpy for processing
                    new_z_numpy = new_z_arr.cpu().detach().numpy().ravel()

                    # Transform the point from reduced space to original space
                    new_x_numpy = transform_point_to_original_space(self, new_z_numpy)

                    is_outside_bounds = (not np.all(new_x_numpy >= self.bounds[:, 0]) or
                                         not np.all(new_x_numpy <= self.bounds[:, 1]))

                    # Ensure the point is within bounds
                    if is_outside_bounds:
                        raise f"Error: PCA transformed point {new_x_numpy} was out of bounds"

                    # Append the new points to both spaces
                    self.x_evals.append(new_x_numpy)
                    self.__z_evals.append(new_z_numpy)

                    new_f_eval = problem(new_x_numpy)
                    batch_f_vals.append(new_f_eval)

                    if self.verbose:
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

                # Adjust trust region based on improvement
                if batch_f_vals:
                    batch_best = min(batch_f_vals)
                    self._adjust_trust_region(np.array([batch_best]), np.array([prev_best_value]))

                # Print best to screen if verbose
                if self.verbose:
                    print(
                        f"Evals: {self.number_of_function_evaluations}/{self.budget} (TR size: {self.length:.4f})",
                        f"Best:\n"
                        f"    x: {self.x_evals[self.current_best_index]}\n"
                        f"    y: {self.current_best}",
                        flush=True
                    )

                # Save logs
                if self.save_logs:
                    acqf_values_log, penalty_log, pei_values_log = self.pacqf.log_forward(
                        torch.tensor(self.__z_evals).to(dtype=self.dtype).unsqueeze(1)
                    )
                    df = pd.DataFrame({
                        "x": self.x_evals,
                        "z": self.__z_evals,
                        "p": self.f_evals,
                        "acqf": acqf_values_log.detach().numpy(),
                        "penalty": penalty_log.detach().numpy(),
                        "pacqf": pei_values_log.detach().numpy(),
                        "tr_size": [self.length] * len(self.x_evals),
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
                                                    torch.tensor(self.bounds),
                                                    torch.tensor(self.x_evals[-1]).unsqueeze(0),
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

    def _transform_points_to_reduced_space(self) -> None:
        """Transform evaluated points to the reduced space using weighted PCA.

        Steps:
        1. Select points within the trust region
        2. Calculate weights based on function values (rank-based)
        3. Fit PCA on those points
        4. Transform all points to reduced space
        """
        if len(self.x_evals) < 2:
            if len(self.x_evals) == 1:
                self.__z_evals = [np.zeros(1)]
            return

        X_all = np.vstack(self.x_evals)
        f_all = np.array(self.f_evals)

        # Step 1: Determine trust region center
        if hasattr(self, 'current_best_index') and self.current_best_index is not None:
            x_center = self.x_evals[self.current_best_index]
        else:
            raise "No current best"


        # Step 3: Identify points within the trust region (box in original space)
        lower_bound = x_center - 0.5 * self.length
        upper_bound = x_center + 0.5 * self.length

        inside_tr_mask = np.all((X_all >= lower_bound) & (X_all <= upper_bound), axis=1)
        # points within the trust region
        X_tr = X_all[inside_tr_mask]
        f_tr = f_all[inside_tr_mask]

        # Step 4: Center the selected data
        self.data_mean = np.mean(X_tr, axis=0)
        X_centered = X_tr - self.data_mean

        # Step 5: Compute rank-based weights
        weights = calculate_weights(f_tr, maximization=self.maximization)
        weighted_X = X_centered * weights[:, np.newaxis]

        # Step 6: Fit PCA
        self.pca = PCA()
        start_time = perf_counter()
        self.pca.fit(weighted_X)
        self.timing_logs["pca"].append(perf_counter() - start_time)

        self.component_matrix = np.copy(self.pca.components_)
        self.explained_variance_ratio = self.pca.explained_variance_ratio_

        n_components = self.n_components
        if n_components <= 0:
            cumulative_var_ratio = np.cumsum(self.explained_variance_ratio)
            n_components = np.sum(cumulative_var_ratio <= self.var_threshold) + 1
            n_components = max(1, min(n_components, len(self.explained_variance_ratio)))

        self.reduced_space_dim_num = n_components
        self.pca.components_ = self.pca.components_[:n_components]

        if self.verbose:
            explained = np.sum(self.pca.explained_variance_ratio_[:n_components]) * 100
            print(f"\nUsing {n_components} principal components with {explained:.2f}% explained variance")

        # Step 7: Transform trust region points to reduced space using full PCA
        X_tr_centered = X_tr - self.data_mean
        Z_tr = self.pca.transform(X_tr_centered)
        self.__z_evals = [Z_tr[i, :] for i in range(Z_tr.shape[0])]
        self.Z_matrix = Z_tr

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
        bounds_torch = torch.from_numpy(z_bounds.transpose()).to(dtype=self.dtype)

        # Convert the initial values to Torch Tensors and move to device
        train_z = np.array(self.__z_evals).reshape((-1, len(self.__z_evals[0])))
        train_z = torch.from_numpy(train_z).to(dtype=self.dtype)

        train_obj = np.array(self.f_evals).reshape((-1, 1))
        train_obj = torch.from_numpy(train_obj).to(dtype=self.dtype)

        start_time = perf_counter()
        train_Yvar = torch.full_like(train_obj, fill_value=1e-6)
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
            raise "No DoE before using the BO loop with AcquisitionFunction?!?!?"

        # Define the function to transform points from reduced space to original space
        def pca_transform_fn(z):
            # Handle batched inputs
            z_np = z.cpu().detach().numpy()

            # Quick path for single points
            # TODO: Maybe reshape z_np instead of duplicating the code?
            if len(z_np.shape) == 1:
                return torch.tensor([transform_point_to_original_space(self, z_np)],
                                    dtype=self.dtype)

            # Process each point individually
            x_list = []
            for i in range(z_np.shape[0]):
                x_i = transform_point_to_original_space(self, z_np[i])
                x_list.append(x_i)

            # Stack results and convert back to tensor
            x_batch = np.vstack(x_list)
            return torch.tensor(x_batch, dtype=self.dtype)

        # Create original bounds tensor
        original_bounds = torch.tensor(self.bounds, dtype=self.dtype)

        # Calculate z bounds
        r = torch.min(torch.abs(original_bounds[:, 0] - original_bounds[:, 1]) / 2)
        z_bounds = (torch.tensor([[-r], [r]], dtype=self.dtype)
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

        lb, ub = self._compute_trust_region_bounds()
        bounds = torch.tensor(np.vstack((lb, ub)).T, dtype=self.dtype).to(self.device)

        # Optimize
        start_time = perf_counter()
        candidates, _ = optimize_acqf(
            acq_function=self.pacqf,
            bounds=bounds, #changed from z_bounds
            q=self.batch_size,
            num_restarts=20,
            raw_samples=100,
            #q=self.__torch_config['BATCH_SIZE'],
            #num_restarts=self.__torch_config['NUM_RESTARTS'],
            #raw_samples=self.__torch_config['RAW_SAMPLES'],
            options={"batch_limit": 5, "maxiter": 500},
            return_best_only=True
        )
        self.timing_logs["optimize_acqf"].append(perf_counter() - start_time)

        return candidates

    #
    #                      TRUST REGIONS CODE
    #

    def _initialize_trust_region(self):
        # Trust region length settings
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8
        self.length = self.length_init
    def _update_trust_region(self):
        """Update trust region parameters based on reduced space dimensionality."""

        # Success/failure tolerances (TuRBO-1 rules)
        dim = self.reduced_space_dim_num
        if self.batch_size == 1:
            self.failtol = int(np.ceil(np.max([4.0, dim])))
            self.succtol = 3
        else:
            self.failtol = int(np.ceil(np.max([4.0 / self.batch_size, dim / self.batch_size])))
            self.succtol = 3

        self.failcount = 0
        self.succcount = 0

        # Set center of trust region (best point so far in reduced space)
        Z = np.vstack(self.__z_evals)
        fX = np.array(self.f_evals)
        self.tr_center = Z[fX.argmin().item(), :][None, :]  # shape (1, d)

        # Fit model if needed to extract lengthscales
        if not hasattr(self, '__model_obj'):
            raise RuntimeError("GP model must be initialized before calling _initialize_trust_region.")

        # Extract and normalize lengthscales from GP
        raw_weights = self.__model_obj.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = raw_weights / raw_weights.mean()
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))  # Normalize so prod = 1
        self.tr_weights = weights  # Save for candidate sampling, etc.

        # Compute trust region bounds in reduced space
        lb = np.clip(self.tr_center - weights * self.length / 2.0, -1.0, 1.0)
        ub = np.clip(self.tr_center + weights * self.length / 2.0, -1.0, 1.0)
        self.tr_lb = lb
        self.tr_ub = ub

    def _compute_trust_region_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute trust region bounds in the reduced space (PCA space).

        Returns:
            tuple[np.ndarray, np.ndarray]: Lower and upper bounds of the trust region.
        """
        # Use reduced-space evaluated points
        Z = np.vstack(self.__z_evals)
        fX = np.array(self.f_evals)

        # Trust region center: best point so far in reduced space
        z_center = Z[np.argmin(fX)][None, :]

        # Extract GP model lengthscales
        ls = self.__model_obj.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()

        # Normalize weights (product of weights = 1)
        weights = ls / ls.mean()
        weights /= np.prod(weights ** (1.0 / len(weights)))

        # Compute trust region bounds
        lb = np.clip(z_center - weights * self.length / 2.0, -1.0, 1.0)
        ub = np.clip(z_center + weights * self.length / 2.0, -1.0, 1.0)

        return lb, ub

    def _update_trust_region(self, fX_next, fX_prev):
            """Adjust trust region size based on optimization progress.

            Parameters
            ----------
            fX_next : numpy.ndarray
                Function values of newly evaluated points.
            fX_prev : numpy.ndarray
                Function values of previously evaluated points.
            """
            if np.min(fX_next) < np.min(fX_prev) - 1e-3 * math.fabs(np.min(fX_prev)):
                self.succcount += 1
                self.failcount = 0
            else:
                self.succcount = 0
                self.failcount += 1

            if self.succcount == self.succtol:  # Expand trust region
                self.length = min([2.0 * self.length, self.length_max])
                self.succcount = 0
            elif self.failcount == self.failtol:  # Shrink trust region
                self.length /= 2.0
                self.failcount = 0

            if hasattr(self, 'length_prev'):
                if self.verbose and self.length < self.length_prev:
                    print(f"Trust region size decreased to {self.length:.4f}")
                elif self.verbose and self.length > self.length_prev:
                    print(f"Trust region size increased to {self.length:.4f}")

                self.length_prev = self.length
            else:
                self.length_prev = self.length

        #
        # END TRUST REGION CODE
        #
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
