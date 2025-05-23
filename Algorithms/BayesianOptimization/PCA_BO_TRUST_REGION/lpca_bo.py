import math

import numpy
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize, Normalize
from gpytorch.kernels import MaternKernel
from numpy.linalg import norm

from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import CleanPCABO, MyPCA


class CleanLPCABO(CleanPCABO):

    # INITIALIZE TRUST REGION
    def initialize_restart(self):
        # Trust region length settings
        self.length_min = 0.5 ** 7
        self.length_max = 1.6
        self.length_init = 0.8
        self.length = self.length_init
        # new points + function values with the current best
        self.X = np.zeros((0, self.X.shape[1]))
        self.fX = np.zeros(0)
        self.failcount = 0
        self.succcount = 0
        self.succtol = 3
        self.failtol = 3
        self.length = self.length_init
        [self.eval_at(point) for point in self.doe.get_points(self.bounds)]

    # OPTIMIZATION LOOP
    def optimize(self):
        while self.budget > self.function_evaluation_count:
            self.initialize_restart()
            while self.budget > self.function_evaluation_count and self.length >= self.length_min:
                self.iteration()
                self.update_trust_region()

    def filter_points(self):
        # get trust region bounds
        bounds = self.return_tr_bounds()

        lb = bounds[:, 0]
        ub = bounds[:, 1]

        # Filter points inside trust region
        mask = np.all((self.X >= lb) & (self.X <= ub), axis=1)

        min_num_points = max(bounds.shape[0], 2)

        if sum(mask) < min_num_points:
            # Calculate Manhattan distance from each point to the trust region bounds
            distances = np.zeros(len(self.X))
            for i, point in enumerate(self.X):
                # Distance to trust region is 0 if inside, otherwise sum of distances to closest bounds
                dist_to_bounds = 0
                for j in range(len(point)):
                    if point[j] < lb[j]:
                        dist_to_bounds += lb[j] - point[j]
                    elif point[j] > ub[j]:
                        dist_to_bounds += point[j] - ub[j]
                distances[i] = dist_to_bounds

            # Get indices of the n closest points to the trust region
            closest_indices = np.argsort(distances)[:2]

            # Update mask to include these n closest points
            mask[closest_indices] = True

        assert np.sum(mask) >= min_num_points

        return mask

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

    # CALCULATE TRUST REGION BOUNDS
    def return_tr_bounds(self):
        # 1. Determine center of trust region (current best)
        idx_best = np.argmin(self.fX)  # For minimization
        x_center = self.X[idx_best]

        ub = self.bounds[:, 1]
        lb = self.bounds[:, 0]

        bounds_width = ub - lb

        assert np.all(bounds_width > 0)

        # 2. Compute trust region bounds
        tr_lb = x_center - self.length / 2.0 * bounds_width
        tr_ub = x_center + self.length / 2.0 * bounds_width

        tr_ub = np.minimum(tr_ub, ub)
        tr_lb = np.maximum(tr_lb, lb)

        return np.stack([tr_lb, tr_ub], axis=1)

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
