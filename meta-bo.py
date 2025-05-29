#!/usr/bin/env python3
import torch
import polars as pl
from pathlib import Path
from typing import Tuple

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood

from Algorithms.utils.experiment_loss import get_loss


class BayesianOptimizer:
    def __init__(self):
        # Parameter bounds: [gpr_p, gpr_val_factor, onorm_factor]
        self.bounds = torch.tensor([
            [0.3, 0.0, 0.0],  # lower bounds
            [1.0, 1.0, 4.0]  # upper bounds
        ], dtype=torch.float64)

    def sample_initial_candidates(self, n_candidates: int = 20) -> torch.Tensor:
        """Generate initial candidates using Sobol sampling with proper onorm_factor handling."""
        # Generate Sobol samples in [0,1]^3
        sobol_samples = draw_sobol_samples(
            bounds=torch.stack([torch.zeros(3), torch.ones(3)]),
            n=n_candidates,
            q=1
        ).squeeze(1)

        # Transform to actual parameter ranges
        candidates = unnormalize(sobol_samples, self.bounds)

        # Handle onorm_factor constraint: either 0.0 or between 1.0 and 4.0
        # Use the third Sobol dimension to decide between 0.0 and [1.0, 4.0]
        onorm_raw = sobol_samples[:, 2]

        # Split candidates: if raw value < 0.5, set to 0.0, else map to [1.0, 4.0]
        mask_zero = onorm_raw < 0.5
        mask_range = ~mask_zero

        candidates[mask_zero, 2] = 0.0
        candidates[mask_range, 2] = 1.0 + 3.0 * (onorm_raw[mask_range] - 0.5) / 0.5

        return candidates

    def load_current_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load current evaluation data."""
        df = get_loss("meta-bo")

        # Convert to torch tensors
        X = torch.tensor(
            df.select(['gpr_p', 'gpr_val_factor', 'onorm_factor']).to_numpy(),
            dtype=torch.float64
        )
        # Want to minimize loss, so negate for maximization
        y = torch.tensor(-df['loss'].to_numpy(), dtype=torch.float64).unsqueeze(-1)

        return X, y

    def constraint_onorm_factor(self, X: torch.Tensor) -> torch.Tensor:
        """Apply constraint: onorm_factor must be 0.0 or >= 1.0"""
        X_constrained = X.clone()

        # For values between 0 and 1 (exclusive), set to 0 or 1 based on proximity
        mask = (X_constrained[:, 2] > 0.0) & (X_constrained[:, 2] < 1.0)
        X_constrained[mask, 2] = torch.where(
            X_constrained[mask, 2] < 0.5,
            torch.zeros_like(X_constrained[mask, 2]),
            torch.ones_like(X_constrained[mask, 2])
        )

        return X_constrained

    def get_next_candidates(self, X: torch.Tensor, y: torch.Tensor,
                            n_candidates: int = 10) -> torch.Tensor:
        """Generate next batch of candidates using Bayesian Optimization."""

        # Normalize inputs for better GP performance
        X_norm = normalize(X, self.bounds)

        # Fit Gaussian Process model
        gp = SingleTaskGP(X_norm, y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        # Define batch acquisition function
        qLogEI = qLogExpectedImprovement(gp, best_f=y.max())

        # Optimize acquisition function to get next batch of candidates
        candidates_norm, _ = optimize_acqf(
            acq_function=qLogEI,
            bounds=torch.stack([torch.zeros(3), torch.ones(3)]),
            q=n_candidates,
            num_restarts=20,
            raw_samples=100,
        )

        # Unnormalize candidates
        candidates = unnormalize(candidates_norm, self.bounds)

        # Apply constraints
        candidates = self.constraint_onorm_factor(candidates)

        return candidates

    def save_candidates(self, candidates: torch.Tensor):
        """Save candidates for external evaluation."""
        candidates_df = pl.DataFrame({
            'gpr_p': candidates[:, 0].numpy(),
            'gpr_val_factor': candidates[:, 1].numpy(),
            'onorm_factor': candidates[:, 2].numpy()
        })

        print("Next candidates to evaluate:")
        print(candidates_df)

        # Save to file for external use
        candidates_df.write_csv("next_candidates.csv")
        print(f"Candidates saved to next_candidates.csv")

    def run_iteration(self):
        """Run one BO iteration."""
        try:
            # Load current evaluation data
            X, y = self.load_current_data()
            print(f"Loaded {len(X)} previous evaluations")
            print(f"Best loss so far: {-y.max().item():.6f}")

        except Exception as e:
            print(f"Could not load previous data: {e}")
            print("Starting with initial random sampling...")

            # Generate initial candidates
            candidates = self.sample_initial_candidates(20)
            self.save_candidates(candidates)
            return

        # Check if we have enough data for BO
        if len(X) < 5:
            print("Not enough data for BO. Adding more random samples...")
            additional_candidates = self.sample_initial_candidates(10)
            self.save_candidates(additional_candidates)
            return

        # Generate next candidates using BO
        print("Generating next candidates using Bayesian Optimization...")
        candidates = self.get_next_candidates(X, y, n_candidates=10)
        self.save_candidates(candidates)


def main():
    """Main execution function."""
    print("\n=== OPCABO Hyperparameter Tuning ===")

    optimizer = BayesianOptimizer()
    optimizer.run_iteration()

    print("\nIteration complete!\n")


if __name__ == "__main__":
    main()
