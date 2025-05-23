import typing
from enum import Enum
from typing import Callable

import numpy as np
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound, AnalyticAcquisitionFunction
)
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel
from numpy.linalg import norm
from sklearn.decomposition import PCA

from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import LHS_sampler
from Algorithms.BayesianOptimization.PenalizedAcqf import PenalizedAcqf

USE_CONSTRAINTS = True

class DOE:
    def __init__(
            self,
            n: int,
            **kwargs
    ):
        self.n = n

        self.lhs_parameters = {
            "criterion": "center",
            "iterations": 1000,
            "sample_zero": False,
        }

        for key, value in kwargs:
            if key.lower().strip() == "doe_parameters":
                self.lhs_parameters = {
                    **self.lhs_parameters,
                    **value
                }

    def get_points(self, bounds: np.ndarray):
        unscaled_points = LHS_sampler(**self.lhs_parameters)(
            bounds.shape[0],
            self.n
        )

        points = np.empty_like(unscaled_points)

        for dim in range(bounds.shape[0]):
            # Compute the multiplier
            multiplier = bounds[dim, 1] - bounds[dim, 0]
            points[:, dim] = multiplier * unscaled_points[:, dim] + bounds[dim, 0]

        return points


class PCBANumComponents:
    def __init__(
            self,
            num_components: typing.Optional[int] = None,
            var_threshold: typing.Optional[float] = None
    ):
        self.num_components = num_components
        self.var_threshold = var_threshold
        
        assert num_components is None or var_threshold is None, "Cannot specify both num_components and var_threshold"
        
    def __call__(self, pca: PCA) -> int:
        if self.num_components is not None:
            return self.num_components
        
        if self.var_threshold is not None:
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= self.var_threshold) + 1
            return n_components

        raise "Illegal state"


class MyPCA:
    def __init__(
            self,
            points_x: np.ndarray,
            points_y: np.ndarray,
            maximization: bool,
            pca_num_components: PCBANumComponents
    ):
        weights = calculate_weights(maximization, points_y)

        self.data_mean = np.mean(points_x, axis=0)
        points_x_centered = points_x - self.data_mean

        weighted_points_x = points_x_centered * weights[:, np.newaxis]

        self.pca = PCA()
        self.pca.fit(weighted_points_x)

        self.pca.components_ = self.pca.components_[:pca_num_components(self.pca)]

    def transform_to_reduced(self, points_x: np.ndarray) -> np.ndarray:
        assert points_x.shape[-1] == self.data_mean.shape[0]

        return self.pca.transform(points_x - self.data_mean)

    def transform_to_original(self, points_z: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert points_z.shape[-1] == self.pca.components_.shape[0], f"{points_z} {self.pca.components_.shape[0]}"

        if isinstance(points_z, torch.Tensor):
            components = torch.from_numpy(self.pca.components_)
            mean = torch.from_numpy(self.data_mean)
            mean_ = torch.from_numpy(self.pca.mean_)
            return mean + torch.matmul(points_z, components) + mean_

        return self.data_mean + self.pca.inverse_transform(points_z)



    

class CleanPCABO:

    function_evaluation_count = 0

    def __init__(
            self,
            problem: Callable[[np.ndarray], float],
            budget: int,
            bounds: np.ndarray,
            doe: DOE,
            maximization: bool,
            acquisition_function_class,
            pca_num_components: PCBANumComponents
    ):
        self.problem = problem
        self.budget = budget
        self.bounds = bounds
        self.maximization = maximization
        self.acquisition_function_class = acquisition_function_class
        self.pca_num_components = pca_num_components
        self.doe = doe


        print(f"acquisition_function_class: {acquisition_function_class}")

        self.X = np.zeros((0, self.d))
        self.fX = np.zeros(0)

        self.optimize()

        assert self.function_evaluation_count <= self.budget


    def optimize(self):
        # Get and evaluate initial DoE points
        [self.eval_at(point) for point in self.doe.get_points(self.bounds)]

        while self.budget > self.function_evaluation_count:
            self.iteration()
        print(self.current_best)

    def iteration(self):
        pca = self.fit_pca()

        z_bounds = calculate_reduced_space_bounds(self.return_tr_bounds(), pca)

        points_z = pca.transform_to_reduced(self.X)

        gpr_model = self.create_gpr_model(points_z, z_bounds)

        acquisition_function = self.create_acquisition_function(gpr_model)

        penalized_acquisition_function = self.create_penalized_acquisition(acquisition_function, gpr_model, pca)

        chosen_point_z = self.optimize_acquisition(pca, penalized_acquisition_function, z_bounds)

        chosen_point_x = pca.transform_to_original(chosen_point_z)

        self.eval_at(chosen_point_x)

        print(chosen_point_x, self.fX[-1])

    def fit_pca(self):
        return MyPCA(
            points_x=self.X,
            points_y=self.fX,
            pca_num_components=self.pca_num_components,
            maximization=self.maximization
        )

    def create_gpr_model(self, points_z, z_bounds):
        model = SingleTaskGP(
            torch.from_numpy(points_z),
            torch.from_numpy(self.fX.reshape((-1, 1))),
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=points_z.shape[-1],
                bounds=torch.from_numpy(z_bounds)
            ),
        )

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

    def create_acquisition_function(self, gpr_model) -> AnalyticAcquisitionFunction:
        return self.acquisition_function_class(
            model=gpr_model,
            best_f=self.current_best,
            maximize=self.maximization
        )


    def return_tr_bounds(self):
        return self.bounds

    def create_penalized_acquisition(self, acquisition_function, gpr_model, pca):
        if USE_CONSTRAINTS:
            return acquisition_function
        return PenalizedAcqf(
            acquisition_function=acquisition_function,
            model=gpr_model,
            best_f=self.current_best,
            original_bounds=torch.from_numpy(self.return_tr_bounds()),
            transform_to_original=lambda points_x_tensor: pca.transform_to_original(points_x_tensor),
            transform_to_reduced=lambda points_x_tensor: torch.from_numpy(
                pca.transform_to_reduced(points_x_tensor.detach().numpy())
            ),
            penalty_factor=1000
        )

    def optimize_acquisition(self, pca: MyPCA, penalized_acquisition_function, z_bounds):
        inequality_constraints = None
        if USE_CONSTRAINTS:
            # Get PCA components and means for transformation
            components = torch.from_numpy(pca.pca.components_)  # shape: [n_components, n_features]
            data_mean = torch.from_numpy(pca.data_mean)  # shape: [n_features]
            pca_mean = torch.from_numpy(
                pca.pca.mean_ if hasattr(pca.pca, 'mean_') else np.zeros_like(pca.data_mean)) # shape: [n_features]
    
            # Get original bounds as tensors
            lower_bounds = torch.from_numpy(self.return_tr_bounds()[:, 0])  # shape: [n_features]
            upper_bounds = torch.from_numpy(self.return_tr_bounds()[:, 1])  # shape: [n_features]
    
            # The transformation from reduced to original space is:
            # x_orig = data_mean + components.T @ z + pca_mean
            # So we need to ensure:
            # lower_bounds <= data_mean + components.T @ z + pca_mean <= upper_bounds
    
            # Calculate the offset (data_mean + pca_mean)
            total_offset = data_mean + pca_mean
    
            # Format inequality constraints according to botorch requirements
            # We need to create constraints in the form:
            # sum_i (X[indices[i]] * coefficients[i]) >= rhs
    
            inequality_constraints = []
            n_components = components.shape[0]  # Number of PCA components
    
            # For each dimension in the original space
            for dim in range(self.d):
                # Upper bound constraint: components[dim] @ z <= upper_bounds[dim] - total_offset[dim]
                # Convert to: -components[dim] @ z >= -(upper_bounds[dim] - total_offset[dim])
                upper_indices = torch.arange(n_components, dtype=torch.long)
                upper_coefficients = -components[:, dim].cpu()  # Transfer to CPU for constraint definition
                upper_rhs = -(upper_bounds[dim] - total_offset[dim]).cpu()
                inequality_constraints.append((upper_indices, upper_coefficients, upper_rhs))
    
                # Lower bound constraint: components[dim] @ z >= lower_bounds[dim] - total_offset[dim]
                lower_indices = torch.arange(n_components, dtype=torch.long)
                lower_coefficients = components[:, dim].cpu()  # Transfer to CPU for constraint definition
                lower_rhs = (lower_bounds[dim] - total_offset[dim]).cpu()
                inequality_constraints.append((lower_indices, lower_coefficients, lower_rhs))
    
            # Move acquisition function to the selected device
            if hasattr(penalized_acquisition_function, 'to'):
                penalized_acquisition_function = penalized_acquisition_function
    
            # If the acquisition function contains a model, move it to the device
            if hasattr(penalized_acquisition_function, 'model') and hasattr(penalized_acquisition_function.model, 'to'):
                penalized_acquisition_function.model = penalized_acquisition_function.model


        raw_samples = 5  # TODO make configurable
        # Optimize the acquisition function
        candidates, _ = optimize_acqf(
            acq_function=penalized_acquisition_function,
            bounds=torch.from_numpy(z_bounds),
            q=1,
            num_restarts=2,
            raw_samples=raw_samples,
            #options={"batch_limit": 50, "maxiter": 500, "device": device},
            return_best_only=True,
            inequality_constraints=inequality_constraints,  # Properly formatted inequality constraints
        )

        # Transfer results back to CPU and convert to numpy
        return candidates.cpu().detach().numpy().reshape(-1)



    def eval_at(self, point_x: np.ndarray):
        value = self.problem(point_x)

        self.X = np.vstack((self.X, point_x))
        self.fX = np.append(self.fX, [value])

        self.function_evaluation_count += 1


    @property
    def d(self) -> int:
        return self.bounds.shape[0]

    @property
    def current_best(self) -> float:
        return self.fX.max()  if self.maximization else self.fX.min()


def calculate_weights(maximization: bool, points_y: np.ndarray) -> np.ndarray:
    """Calculate rank-based weights for PCA transformation.

    This implements the rank-based weighting scheme described in the original PCA-BO paper,
    where better points (with lower function values for minimization) are assigned higher weights.

    Returns:
        np.ndarray: Weights for each data point.
    """
    n = len(points_y)

    # Get the ranking of points (1 = best, n = worst)
    if maximization:
        ranks = np.argsort(np.argsort(-np.array(points_y))) + 1
    else:
        ranks = np.argsort(np.argsort(np.array(points_y))) + 1

    # Calculate pre-weights
    pre_weights = np.log(n) - np.log(ranks)

    # Normalize weights
    weights = pre_weights / pre_weights.sum()

    return weights


def calculate_reduced_space_bounds(tr_bounds: np.ndarray, pca: MyPCA):
    C = np.abs(tr_bounds[:, 0] - tr_bounds[:, 1]) / 2
    radius = norm(tr_bounds[:, 0] - C)

    z_bounds = np.array([[-radius], [radius]]).repeat(pca.pca.components_.shape[0], axis=1)

    return z_bounds


class AcquisitionFunctionEnum(Enum):
    EXPECTED_IMPROVEMENT = (LogExpectedImprovement, "EI", "expected_improvement")
    PROBABILITY_OF_IMPROVEMENT = (ProbabilityOfImprovement, "PI", "probability_of_improvement")
    UPPER_CONFIDENCE_BOUND = (UpperConfidenceBound, "UCB", "upper_confidence_bound")

    @property
    def class_type(self):
        return self.value[0]

    @property
    def shorthand(self):
        return self.value[1]

    @property
    def name(self):
        return self.value[2]

    @classmethod
    def from_name(cls, name):
        for acq_func in cls:
            if acq_func.value[2] == name:
                return acq_func
        raise ValueError(f"Invalid acquisition function name: {name}")

