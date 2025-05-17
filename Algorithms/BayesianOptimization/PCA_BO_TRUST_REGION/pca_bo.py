import typing
from typing import Callable

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel
from sklearn.decomposition import PCA

from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import LHS_sampler
from Algorithms.BayesianOptimization.PenalizedAcqf import PenalizedAcqf


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

        self.X = np.zeros((0, self.d))
        self.fX = np.zeros(0)

        # Get and evaluate initial DoE points
        [self.eval_at(point) for point in doe.get_points(bounds)]

        self.optimize()

        assert self.function_evaluation_count <= self.budget


    def optimize(self):
        while self.budget > self.function_evaluation_count:
            self.iteration()


    def iteration(self):
        pca = self.fit_pca()

        points_z = pca.transform_to_reduced(self.X)

        gpr_model = self.create_gpr_model(points_z)

        acquisition_function = self.create_acquisition_function(gpr_model)

        penalized_acquisition_function = self.create_penalized_acquisition(acquisition_function, gpr_model, pca)

        chosen_point_z = self.optimize_acquisition(penalized_acquisition_function)

        chosen_point_x = pca.transform_to_original(chosen_point_z)

        self.eval_at(chosen_point_x)

    def fit_pca(self):
        return MyPCA(
            points_x=self.X,
            points_y=self.fX,
            pca_num_components=self.pca_num_components,
            maximization=self.maximization
        )

    def create_gpr_model(self, points_z):
        z_bounds = self.calculate_reduced_space_bounds()
        return SingleTaskGP(
            torch.from_numpy(points_z),
            torch.from_numpy(self.fX.reshape((-1, 1))),
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=points_z.shape[-1],
                bounds=torch.from_numpy(z_bounds)
            )
        )

    def create_acquisition_function(self, gpr_model):
        return self.acquisition_function_class(
            model=gpr_model,
            best_f=self.current_best,
            maximize=self.maximization
        )

    def create_penalized_acquisition(self, acquisition_function, gpr_model, pca):
        return PenalizedAcqf(
            acquisition_function=acquisition_function,
            model=gpr_model,
            best_f=self.current_best,
            original_bounds=torch.from_numpy(self.bounds),
            pca_transform_fn=lambda points_x_tensor: torch.from_numpy(
                pca.transform_to_original(points_x_tensor.detach().numpy())
            ),
            penalty_factor=1000.0
        )

    def optimize_acquisition(self, penalized_acquizition_function):
        z_bounds = self.calculate_reduced_space_bounds()
        candidates, _ = optimize_acqf(
            acq_function=penalized_acquizition_function,
            bounds=torch.from_numpy(z_bounds),
            q=1,
            num_restarts=5,
            raw_samples=150, # TODO make configurable
            options={"batch_limit": 5, "maxiter": 500},
            return_best_only=True
        )
        return candidates.detach().numpy().reshape(-1)

    def calculate_reduced_space_bounds(self):
        # TODO fix this up
        r = np.min(np.abs(self.bounds[:, 0] - self.bounds[:, 1])) / 2
        z_bounds = np.array([[-r], [r]]).repeat(1, axis=1)

        return z_bounds


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

    def transform_to_original(self, points_z: np.ndarray) -> np.ndarray:
        assert points_z.shape[-1] == self.pca.n_components_

        return self.data_mean + self.pca.inverse_transform(points_z)


from enum import Enum
from botorch.acquisition import (
    LogExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound
)



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

