import math

import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from gpytorch.kernels import MaternKernel

from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import CleanPCABO, MyPCA


class CleanLPCABO(CleanPCABO):

    # INITIALIZE TRUST REGION
    def initialize_restart(self):
        # Trust region length settings
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8
        self.length = self.length_ini
        # new points + function values with the current best
        self.X = np.zeros((0, self.X.shape[1]))
        self.fX = np.zeros(0)
        self.failcount = 0
        self.succcount = 0
        self.succtol = 3
        self.failtol = 3
        self.length = self.length_init

    # OPTIMIZATION LOOP
    def optimize(self):
        while self.budget > self.function_evaluation_count:
            self.initialize_restart()
            while self.budget > self.function_evaluation_count and self.length >= self.length_min:
                self.iteration()
                self.update_trust_region()

    def filter_points(self):
        # 1. Determine center of trust region (current best)
        idx_best = np.argmin(self.X)  # For minimization
        x_center = self.X[idx_best]

        # 2. Compute trust region bounds
        lb = np.clip(x_center - self.length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + self.length / 2.0, 0.0, 1.0)

        # 3. Filter points inside trust region
        return np.all((self.X >= lb) & (self.X <= ub), axis=1)

    # FIT PCA TO DATA WITHIN TRUST REGION
    def fit_pca(self):
        in_tr = self.filter_points()
        X_tr = self.X[in_tr]
        fX_tr = self.fX[in_tr]

        # Fit PCA on filtered points
        return MyPCA(
            points_x=X_tr,
            points_y=fX_tr,
            pca_num_components=self.pca_num_components,
            maximization=self.maximization
        )

    def create_gpr_model(self, points_z):
        z_bounds = self.calculate_reduced_space_bounds()
        in_tr = self.filter_points()
        return SingleTaskGP(
            torch.from_numpy(points_z[in_tr]),
            torch.from_numpy(self.fX.reshape((-1, 1))),
            covar_module=MaternKernel(2.5),  # Use the Matern 5/2 Kernel
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=points_z.shape[-1],
                bounds=torch.from_numpy(z_bounds)
            )
        )

    # ADJUST TRUST REGION
    def update_trust_region(self, ):
        fX_next = self.X[-1]
        if np.min(fX_next) < np.min(self.fX) - 1e-3 * math.fabs(np.min(self.fX)):
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
