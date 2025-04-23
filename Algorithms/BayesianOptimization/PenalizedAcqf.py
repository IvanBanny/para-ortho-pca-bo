from typing import Union, Callable

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
        pca_transform_fn: Function to map points from reduced to original space
        maximize: Whether to maximize or minimize the objective
        penalty_factor: Factor to control the penalty strength
    """

    def __init__(
            self,
            acquisition_function_class: Callable,
            model: Model,
            best_f: Union[float, Tensor],
            original_bounds: Tensor,
            pca_transform_fn: callable,
            maximize: bool = True,
            penalty_factor: float = 1.0,
    ) -> None:
        """Initialize Penalized Expected Improvement.

        Args:
            acquisition_function_class: Acquisition function to base on
            model: A fitted model
            best_f: The best function value observed so far
            original_bounds: Tensor of shape (dim, 2) containing the bounds of the original space
            pca_transform_fn: Function that maps points from reduced space to original space
            maximize: If True, consider the problem a maximization problem
            penalty_factor: Factor to control the strength of the penalty (default: 1.0)
        """
        super().__init__(model=model)
        self.maximize = maximize
        # Expected Improvement component
        self.acquisition_function = acquisition_function_class(model=model, best_f=best_f, maximize=maximize)
        # Original bounds of the search space [lower, upper]
        self.register_buffer("original_bounds", torch.as_tensor(original_bounds))
        # PCA transform function reference
        self.pca_transform_fn = pca_transform_fn
        # Penalty scaling factor
        self.penalty_factor = penalty_factor

    def _compute_penalty(self, X: Tensor) -> Tensor:
        """Compute the penalty for points that would fall outside the original space.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of inputs

        Returns:
            A `batch_shape`-dim Tensor of penalties (non-positive values)
        """
        # Save original batch shape for reshaping at the end
        batch_shape = X.shape[:-2]
        q = X.shape[-2]

        # Reshape input for processing
        X_flat = X.view(-1, X.shape[-1])

        # Map points to original space using the provided transformation function
        X_orig = self.pca_transform_fn(X_flat)

        # Get original bounds
        lower_bounds = self.original_bounds[:, 0]
        upper_bounds = self.original_bounds[:, 1]

        # Check if points are outside bounds
        outside_lower = torch.clamp(lower_bounds - X_orig, min=0)
        outside_upper = torch.clamp(X_orig - upper_bounds, min=0)

        # Compute distance to boundary for points outside bounds
        distance_to_boundary = torch.sum(outside_lower + outside_upper, dim=-1)

        # Reshape to match batch dimensions and q
        distance_to_boundary = distance_to_boundary.view(*batch_shape, q)

        # We need to aggregate across q-dimension to match expected output shape
        # Using minimum distance (most conservative approach)
        distance_to_boundary = distance_to_boundary.min(dim=-1)[0]

        # Calculate penalty (negative distance, so points outside have negative values)
        penalty = -self.penalty_factor * distance_to_boundary

        # Points inside bounds get zero penalty, points outside get negative penalty
        # Ensure all feasible points have exactly zero penalty
        is_feasible = (distance_to_boundary == 0)
        penalty = torch.where(is_feasible, torch.zeros_like(penalty), penalty)

        return penalty

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Penalized Expected Improvement on the candidate set X.

        Args:
            X: A `batch_shape x q x d`-dim Tensor of inputs

        Returns:
            A `batch_shape`-dim Tensor of PEI values at the given design points X
        """
        # Compute regular expected improvement
        acqf_values = self.acquisition_function(X)

        # Check if points would be feasible in original space
        penalty = self._compute_penalty(X)

        # Combine EI with penalty:
        # - For feasible points, use EI value (penalty is 0)
        # - For infeasible points, use penalty (negative value)
        is_feasible = (penalty == 0)

        # Where feasible, use EI; where infeasible, use penalty
        pei_values = torch.where(is_feasible, acqf_values, penalty)

        return pei_values
