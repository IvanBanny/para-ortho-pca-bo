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

from Algorithms.BayesianOptimization.AbstractBayesianOptimizer import LHS_sampler
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import DOE


class CleanVanillaBO:

    function_evaluation_count = 0

    def __init__(
            self,
            problem: Callable[[np.ndarray], float],
            budget: int,
            bounds: np.ndarray,
            doe: DOE,
            maximization: bool,
            acquisition_function_class,
    ):
        self.problem = problem
        self.budget = budget
        self.bounds = bounds
        self.maximization = maximization
        self.acquisition_function_class = acquisition_function_class
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

        # Create GP model directly on original space
        gpr_model = self.create_gpr_model(self.X, self.bounds)

        acquisition_function = self.create_acquisition_function(gpr_model)

        #optimize acquisition function
        chosen_point_x = self.optimize_acquisition(acquisition_function, self.bounds)

        #evaluate chosen point
        self.eval_at(chosen_point_x)
        print(chosen_point_x, self.fX[-1])


    def create_gpr_model(self, points_z, z_bounds):
        model = SingleTaskGP(
            torch.from_numpy(points_z),
            torch.from_numpy(self.fX.reshape((-1, 1))),
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=points_z.shape[-1],
                bounds=torch.from_numpy(z_bounds.T)
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

    def optimize_acquisition(self, acquisition_function, bounds):
        raw_samples = 5  # make configurable

        # Optimize the acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acquisition_function,
            bounds=torch.from_numpy(bounds.T),  # botorch expects bounds as [2, d] -> CHECK!!
            q=1,
            num_restarts=5,
            raw_samples=raw_samples,
            # options={"batch_limit": 50, "maxiter": 500, "device": device},
            return_best_only=True,
        )

        # Transfer results back to CPU and convert to numpy
        return candidates.cpu().detach().numpy().reshape(-1)

    def eval_at(self, point_x: np.ndarray):
        # First clip to problem bounds
        clipped_point = np.clip(point_x, self.bounds[:, 0], self.bounds[:, 1])

        # Evaluate the objective function at the clipped point
        value = self.problem(clipped_point)

        # Store the clipped point and its value
        self.X = np.vstack((self.X, clipped_point))
        self.fX = np.append(self.fX, [value])

        self.function_evaluation_count += 1

        # Optionally, you could return information about whether clipping occurred
        clipping_occurred = not np.allclose(point_x, clipped_point)
        if clipping_occurred:
            print(f"Warning: Point {point_x} was clipped to {clipped_point} to satisfy bounds")

        return value

    @property
    def d(self) -> int:
        return self.bounds.shape[0]

    @property
    def current_best(self) -> float:
        return self.fX.max() if self.maximization else self.fX.min()
