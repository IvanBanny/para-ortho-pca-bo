from ..AbstractBayesianOptimizer import AbstractBayesianOptimizer
from typing import Union, Callable, Optional
from ioh.iohcpp.problem import RealSingleObjective, BBOB
import numpy as np
import torch
import os
from torch import Tensor
from sklearn.decomposition import PCA
from scipy.stats import rankdata

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound
)
from botorch.optim import optimize_acqf
from gpytorch.kernels import MaternKernel

class PCA_BO:
    def __init__(self,
                 budget: int,
                 n_DoE: int = 10,
                 n_components: Union[int, float] = 2,
                 acquisition_function: str = "expected_improvement",
                 random_seed: int = 42,
                 verbose: bool = True,
                 bounds: Optional[np.ndarray] = None,
                 **kwargs):

        self.budget = budget
        self.n_DoE = n_DoE
        self.n_components = n_components
        self.acquisition_function = acquisition_function
        self.verbose = verbose
        self.random_seed = random_seed # for reproducibility
        self.bounds = bounds if bounds is not None else np.array([[-5, 5], [-5, 5]])

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

        self.x_evals = [] # evaluated points
        self.f_evals = [] # corresponding objective values after function evaluation
        self.current_best = float('inf')
        self.maximization = False

    def _initial_sample(self):
        # generates n_DoE random input samples within the provided bounds, which are the first samples to evaluate the objective function
        dim = self.bounds.shape[0]
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.n_DoE, dim))

    def _scale_X(self, X, func_vals):
        # weighting the function values (better performing samples get higher weights)
        r = rankdata(func_vals)
        N = len(func_vals)
        w = np.log(N) - np.log(r)
        w /= np.sum(w)
        return X * w.reshape(-1, 1)

    def _apply_pca(self, X, n_components):
        # dimensionality reduction with PCA
        # keeps only n_components most important PCs
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        # returns the fitted PCA model + transformed data
        return pca, X_pca

    def _train_gp(self, X, Y):
        # train GP on the PCA-reduced data
        train_x = torch.from_numpy(X).float()
        train_y = torch.from_numpy(Y).float().unsqueeze(-1)
        model = SingleTaskGP(train_x, train_y)
        return model.eval() # GP learns a smooth surrogate of the objective function

    def _create_acquisition(self, model, best_f):
        if self.acquisition_function == "expected_improvement":
            return ExpectedImprovement(model=model, best_f=best_f)
        elif self.acquisition_function == "probability_of_improvement":
            return ProbabilityOfImprovement(model=model, best_f=best_f)
        elif self.acquisition_function == "upper_confidence_bound":
            return UpperConfidenceBound(model=model, beta=0.2)
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")

    def _optimize_acqf(self, acq_func, bounds):
        # choose a point where to sample next (using acquisition function)
        bounds_t = torch.tensor(bounds.T).float()  # shape [2, d]
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_t,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        return candidates.detach().numpy()[0]

    def ask(self):
        # get initial samples (before the optimization loop starts)
        return self._initial_sample()

    def tell(self, X, func_vals):
        X = np.array(X)
        func_vals = np.array(func_vals)

        # weight and apply PCA to the data
        X_weighted = self._scale_X(X, func_vals)
        pca, X_pca = self._apply_pca(X_weighted, self.n_components)

        # train GP
        model = self._train_gp(X_pca, func_vals)

        # acquisition function
        acq_func = self._create_acquisition(model, best_f=np.min(func_vals))

        # optimize acquisition in PCA space
        new_pca_x = self._optimize_acqf(acq_func, bounds=np.array([[-5, 5]] * self.n_components))

        # map the new PCA point back to original space
        new_x = pca.inverse_transform(new_pca_x.reshape(1, -1))[0]
        return new_x

    def optimize(self, problem: Callable):
        # the main optimization loop:
        # 1. generate and evaluate n_DoE initial points
        # 2. repeat until the budget is exhausted: call tell() to propose a new sample, evaluate the new sample
        X_init = self.ask()
        for x in X_init:
            fx = problem(x)
            self.x_evals.append(x)
            self.f_evals.append(fx)

        for _ in range(self.budget - self.n_DoE):
            new_x = self.tell(self.x_evals, self.f_evals)
            fx = problem(new_x)
            self.x_evals.append(new_x)
            self.f_evals.append(fx)

            if self.verbose:
                print(f"Evaluated: {new_x}, f(x): {fx}")

        return np.array(self.x_evals), np.array(self.f_evals)