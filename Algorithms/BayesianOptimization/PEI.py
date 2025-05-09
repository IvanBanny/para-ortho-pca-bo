from typing import Union

import torch
from torch import Tensor
from botorch.acquisition import AcquisitionFunction, ExpectedImprovement
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from typing import Callable, Union


class PenalizedExpectedImprovement(AcquisitionFunction):
    """Penalized Expected Improvement acquisition function as described in the PCA-BO paper by Raponi et al.

    This acquisition function extends the standard Expected Improvement (EI) by incorporating
    a penalty for points that would fall outside the original search space when mapped back
    from the reduced PCA space.

    Attributes:
        model: The surrogate model (typically a SingleTaskGP)
        best_f: The best objective value observed so far
        original_bounds: The bounds of the original search space
        pca_transform_fn: Function to map points from reduced to original space
        maximize: Whether to maximize or minimize the objective
        penalty_factor: Factor to control the penalty strength
    """

    def __init__(
        self,
        model: Model,
        best_f: Union[float, torch.Tensor],
        original_bounds: torch.Tensor,
        pca_transform_fn: Callable,
        X_mean: torch.Tensor,
        maximize: bool = True,
        penalty_factor: float = 1.0,
    ) -> None:
        super().__init__(model=model)
        self.maximize = maximize
        self.ei = ExpectedImprovement(model=model, best_f=best_f, maximize=maximize)
        self.pca_transform_fn = pca_transform_fn
        self.penalty_factor = penalty_factor

        # Register original bounds and X_mean as buffers for GPU compatibility
        self.register_buffer("original_bounds", torch.as_tensor(original_bounds))
        self.register_buffer("X_mean", torch.as_tensor(X_mean))

    def _compute_penalty(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute penalty for points that are outside the original bounds.
        Penalty is negative and equals the summed distance outside the bounds.
        """
        batch_shape = X.shape[:-2]
        q = X.shape[-2]
        X_flat = X.view(-1, X.shape[-1])

        # Inverse PCA transform and add X_mean
        X_orig = self.pca_transform_fn(X_flat) + self.X_mean

        lower_bounds = self.original_bounds[:, 0]
        upper_bounds = self.original_bounds[:, 1]

        # Compute distances outside bounds
        below = torch.clamp(lower_bounds - X_orig, min=0)
        above = torch.clamp(X_orig - upper_bounds, min=0)
        penalty_dist = torch.sum(below + above, dim=-1)  # (batch_size,)

        # Reshape and get min across q
        penalty_dist = penalty_dist.view(*batch_shape, q)
        min_penalty_dist = penalty_dist.min(dim=-1)[0]  # shape: batch_shape

        # If feasible (inside bounds), penalty is zero
        penalty = -self.penalty_factor * min_penalty_dist
        penalty = torch.where(min_penalty_dist == 0, torch.zeros_like(penalty), penalty)
        return penalty

    @t_batch_mode_transform()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate penalized expected improvement at X.
        If point is feasible, return EI. Otherwise, return penalty.
        """
        ei = self.ei(X)  # shape: batch_shape
        penalty = self._compute_penalty(X)
        is_feasible = (penalty == 0)

        # Use EI if feasible, otherwise penalty
        pei = torch.where(is_feasible, ei, penalty)
        return pei


