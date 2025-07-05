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
    """

    def __init__(
            self,
            acquisition_function: Callable,
            model: Model,
            original_bounds: Tensor,
            pca_r2d_fn: Callable,
            cont_acqf: bool = True,
            p_factor: float = 1e-2
    ) -> None:
        """Initialize Penalized Expected Improvement.

        Args:
            acquisition_function: Acquisition function to base on
            model: A fitted model
            original_bounds: Tensor of shape (dim, 2) containing the bounds of the original space
            pca_r2d_fn: Function to map points from reduced to original space
            cont_acqf: Whether to use the new penalization method with continuous acqf penalization
            p_factor: Factor to control the strength of the penalty (default: 1e-2)
        """
        super().__init__(model=model)
        # Expected Improvement component
        self.acquisition_function = acquisition_function
        # Original bounds of the search space [lower, upper]
        self.register_buffer("original_bounds", torch.as_tensor(original_bounds))
        # PCA transform function reference
        self.pca_r2d_fn = pca_r2d_fn
        # Use the new penalization method with continuous log acqf penalization
        self.cont_acqf = cont_acqf
        # Penalty scaling factor
        self.p_factor = p_factor

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate pacqf on the candidate set X.

        Args:
            X: A `batch_shape x q x r`-dim Tensor of inputs

        Returns:
            A `batch_shape`-dim Tensor of pacqf values for X
        """
        lb = self.original_bounds[:, 0]  # shape [d]
        ub = self.original_bounds[:, 1]  # shape [d]

        X_flat = X.view(-1, X.shape[-1])  # shape [(batch_shape * q) x r]
        X_orig = self.pca_r2d_fn(X_flat).view(*X.shape[: -1], -1)  # shape [batch_shape x q x d]

        dists_to_bounds = torch.clamp(lb - X_orig, min=0) + torch.clamp(X_orig - ub, min=0)  # shape [batch_shape]
        total_dists = dists_to_bounds.norm(dim=-1).sum(dim=-1)  # shape [batch_shape]

        acqf_vals = self.acquisition_function(X)

        if self.cont_acqf:
            return acqf_vals - total_dists / self.p_factor

        return torch.where(total_dists == 0, acqf_vals, -total_dists / self.p_factor)


    def log_forward(self, X: Tensor) -> tuple[Tensor]:
        """Evaluate pacqf on the candidate set X.

        Args:
            X: A `batch_shape x q x r`-dim Tensor of inputs

        Returns:
            A `batch_shape`-dim Tensor of acqf values for X
            A `batch_shape`-dim Tensor of raw penalty values for X
            A `batch_shape`-dim Tensor of pacqf values for X
        """
        lb = self.original_bounds[:, 0]  # shape [d]
        ub = self.original_bounds[:, 1]  # shape [d]

        X_flat = X.view(-1, X.shape[-1])  # shape [(batch_shape * q) x r]
        X_orig = self.pca_r2d_fn(X_flat).view(*X.shape[: -1], -1)  # shape [batch_shape x q x d]

        dists_to_bounds = torch.clamp(lb - X_orig, min=0) + torch.clamp(X_orig - ub, min=0)  # shape [batch_shape]
        total_dists = dists_to_bounds.norm(dim=-1).sum(dim=-1)  # shape [batch_shape]

        acqf_vals = self.acquisition_function(X)

        if self.cont_acqf:
            return acqf_vals, total_dists / self.p_factor, acqf_vals - total_dists / self.p_factor

        return (acqf_vals, total_dists / self.p_factor,
                torch.where(total_dists == 0, acqf_vals, -total_dists / self.p_factor))
