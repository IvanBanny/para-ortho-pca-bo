from typing import Callable

import torch
from torch import Tensor
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform


class PenalizedAcqf(AnalyticAcquisitionFunction):
    """Penalized acquisition function wrapper described in the PCA-BO paper by Raponi et al.

    This acquisition function extends acquisition_function_class passed as an argument by
    incorporating a penalty for points that would fall outside the original search space
    when mapped back from the reduced PCA space.

    Attributes:
        acquisition_function: The acquisition function
        model: The surrogate model (typically a SingleTaskGP)
        original_bounds: The bounds of the original search space
        pca_d2r_fn: Function to map points from original to reduced space
        pca_r2d_fn: Function to map points from reduced to original space
        penalty_factor: Factor to control the penalty strength
    """

    def __init__(
            self,
            acquisition_function: Callable,
            model: Model,
            original_bounds: Tensor,
            pca_d2r_fn: Callable,
            pca_r2d_fn: Callable,
            penalty_factor: float = 100.0,
    ) -> None:
        """Initialize Penalized Expected Improvement.

        Args:
            acquisition_function: Acquisition function to base on
            model: A fitted model
            original_bounds: Tensor of shape (dim, 2) containing the bounds of the original space
            pca_d2r_fn: Function to map points from original to reduced space
            pca_r2d_fn: Function to map points from reduced to original space
            penalty_factor: Factor to control the strength of the penalty (default: 1.0)
        """
        super().__init__(model=model)
        # Expected Improvement component
        self.acquisition_function = acquisition_function
        # Original bounds of the search space [lower, upper]
        self.register_buffer("original_bounds", torch.as_tensor(original_bounds))
        # PCA transform function reference
        self.pca_d2r_fn = pca_d2r_fn
        self.pca_r2d_fn = pca_r2d_fn
        # Penalty scaling factor
        self.penalty_factor = penalty_factor

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate pacqf on the candidate set X.

        Args:
            X: A `batch_shape x q x r`-dim Tensor of inputs

        Returns:
            A `batch_shape`-dim Tensor of pacqf values at the given design points X
        """
        lb = self.original_bounds[:, 0]  # shape [d]
        ub = self.original_bounds[:, 1]  # shape [d]

        X_flat = X.view(-1, X.shape[-1])  # shape [(batch_shape * q) x r]
        X_orig = self.pca_r2d_fn(X_flat).view(*X.shape[: -1], -1)  # shape [batch_shape x q x d]

        X_clamped = torch.max(torch.min(X_orig, ub), lb)  # shape [batch_shape x q x d]

        within_bounds_per_point = ((X_orig >= lb) & (X_orig <= ub)).all(dim=-1)  # shape [batch_shape x q]
        all_q_within_bounds = within_bounds_per_point.all(dim=-1)  # shape [batch_shape]

        acqf_vals = self.acquisition_function(self.pca_d2r_fn(X_orig))  # shape [batch_shape]

        if all_q_within_bounds.all():
            return acqf_vals

        distances_per_point = torch.norm(X_orig - X_clamped, dim=-1)  # shape [batch_shape x q]
        sum_q_distances = distances_per_point.sum(dim=-1)  # shape [batch_shape]

        result = acqf_vals.clone()  # shape [batch_shape]
        needs_penalty = ~all_q_within_bounds  # shape [batch_shape]
        result[needs_penalty] = (self.acquisition_function(self.pca_d2r_fn(X_clamped[needs_penalty]))
                                 - self.penalty_factor * sum_q_distances[needs_penalty])

        return result  # shape [batch_shape]

    def log_forward(self, X: Tensor):
        """Evaluate pacqf on the candidate set X with verbose returns.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of inputs

        Returns:
            A `batch_shape`-dim Tensor of acqf values at the given design points X
            A `batch_shape`-dim Tensor of penalty values at the given design points X
            A `batch_shape`-dim Tensor of pacqf values at the given design points X
        """
        lb = self.original_bounds[:, 0]  # shape [d]
        ub = self.original_bounds[:, 1]  # shape [d]

        X_flat = X.view(-1, X.shape[-1])  # shape [(batch_shape * q) x r]
        X_orig = self.pca_r2d_fn(X_flat).view(*X.shape[: -1], -1)  # shape [batch_shape x q x d]

        X_clamped = torch.max(torch.min(X_orig, ub), lb)  # shape [batch_shape x q x d]

        within_bounds_per_point = ((X_orig >= lb) & (X_orig <= ub)).all(dim=-1)  # shape [batch_shape x q]
        all_q_within_bounds = within_bounds_per_point.all(dim=-1)  # shape [batch_shape]

        acqf_vals = self.acquisition_function(self.pca_d2r_fn(X_orig))  # shape [batch_shape]

        if all_q_within_bounds.all():
            return acqf_vals

        distances_per_point = torch.norm(X_orig - X_clamped, dim=-1)  # shape [batch_shape x q]
        sum_q_distances = distances_per_point.sum(dim=-1)  # shape [batch_shape]

        result = acqf_vals.clone()  # shape [batch_shape]
        needs_penalty = ~all_q_within_bounds  # shape [batch_shape]
        result[needs_penalty] = (self.acquisition_function(self.pca_d2r_fn(X_clamped[needs_penalty]))
                                 - self.penalty_factor * sum_q_distances[needs_penalty])

        return (self.acquisition_function(self.pca_d2r_fn(X_clamped)),
                self.penalty_factor * sum_q_distances,
                result)
