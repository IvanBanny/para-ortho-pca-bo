from typing import Tuple

import numpy as np
import torch
from botorch.optim import optimize_acqf
import pandas as pd
from numpy import ndarray

import Algorithms.BayesianOptimization.PCA_BO_VIS as PCA_BO_VIS


class TableAcqfVis(PCA_BO_VIS.PCA_BO):
    def plot_acqf_table(self, bounds_torch):
        # get the candidates using acqf optimizer
        candidates, acq_values = optimize_acqf(
            acq_function=self.acquisition_function,
            bounds=bounds_torch,
            q=1,  # self.__torch_config['BATCH_SIZE'],
            return_best_only=False,
            num_restarts=self.torch_config['NUM_RESTARTS'],
            raw_samples=self.torch_config['RAW_SAMPLES'],  # Used for initialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )

        candidates = candidates.view(-1, candidates.shape[-1])
        acq_values = acq_values.view(-1, acq_values.shape[-1])

        print(f"candidates: {candidates.shape} {candidates}")

        penalty, below, above, penalty_dist = self.compute_penalty(candidates)

        penalty = penalty_dist > 0

        candidates = candidates.numpy()

        # inverse the points back to original using the function
        list_of_points_2d = []
        for point in candidates:
            point_original_space = self._transform_point_to_original_space(point)
            list_of_points_2d.append(point_original_space)

        list_of_points_1d = []
        for point in candidates:
            list_of_points_1d.append(point)

        d1 = []
        d2 = []
        # d1+d2
        total_penalty = penalty_dist

        for i in range(len(list_of_points_1d)):
            d1.append(max(above[i][0], below[i][0]))
            d2.append(max(above[i][1], below[i][1]))

        penalty_string = []
        for value in penalty:
            if value:
                penalty_string.append("penalty")
            else:
                penalty_string.append("no penalty")

        table = pd.DataFrame({
            'Point candidates in reduced (1D)': list_of_points_1d,
            'Point candidates in original (2D)': list_of_points_2d,
            'd1': d1,
            'd2': d2,
            'Total penalty (d1 + d2)': total_penalty,
            'Penalized yes/no': penalty_string,
            'EI value': acq_values.numpy().flatten()
        })

        print(table)

    def compute_penalty(self, X: torch.Tensor) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """
        Compute penalty for points that are outside the original bounds.
        Penalty is negative and equals the summed distance outside the bounds.
        """
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        X_flat = X.view(-1, X.shape[-1])

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
            return torch.tensor(x_batch)

        original_bounds = torch.Tensor(self.bounds)
        penalty_factor = 1000

        # Inverse PCA transform and add X_mean
        X_orig = pca_transform_fn(X_flat) # + self.X_mean

        lower_bounds = original_bounds[:, 0]
        upper_bounds = original_bounds[:, 1]

        # Compute distances outside bounds
        below = torch.clamp(lower_bounds - X_orig, min=0)
        above = torch.clamp(X_orig - upper_bounds, min=0)
        penalty_dist = torch.sum(below + above, dim=-1)  # (batch_size,)

        # Reshape and get min across q
        penalty_dist = penalty_dist.view(*batch_shape, q)
        min_penalty_dist = penalty_dist.min(dim=-1)[0]  # shape: batch_shape

        # If feasible (inside bounds), penalty is zero
        penalty = -penalty_factor * min_penalty_dist
        penalty = torch.where(min_penalty_dist == 0, torch.zeros_like(penalty), penalty)

        return penalty.numpy(), below.numpy(), above.numpy(), penalty_dist.numpy()

