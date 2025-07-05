"""Analysis module for Bayesian Optimization experiments.

This module provides tools for loading and analyzing the results of experiments
comparing Vanilla BO, PCA-BO, and O-PCA-BO algorithms, calculating AUC and
final performance metrics.
"""

from typing import List, Optional
import os
import numpy as np
import polars as pl

import src.utils.iohreader as iohreader


class ExperimentAnalyzer:
    """Class to analyze experimental results from Bayesian Optimization algorithms."""

    def __init__(
            self,
            experiment_dir: str,
            algorithms: Optional[List[str]] = None,
            batch_sizes: Optional[List[int]] = None,
            dimensions: Optional[List[int]] = None,
            functions: Optional[List[int]] = None,
            cache: bool = True
    ):
        """Initialize the experiment analyzer.

        Args:
            experiment_dir: Directory containing experiment data.
            algorithms: src to analyze (None for all).
            batch_sizes: Batch sizes to analyze (None for all).
            dimensions: List of problem dimensions to analyze (None for all).
            functions: List of function IDs to analyze (None for all).
            cache: Use cached results.
        """
        self.experiment_dir = experiment_dir
        self.algorithms = algorithms
        self.batch_sizes = batch_sizes
        self.dimensions = dimensions
        self.functions = functions
        self.cache = cache

        self.data = None

    def analyze_all(self) -> None:
        """Perform all analyses and print results."""
        self.load_data()
        self.analyze_convergence()
        self.analyze_times()
        self.data = None

    def load_data(self):
        """Load experiment data using IOHreader."""
        cache_path = os.path.join(self.experiment_dir, "cache.parquet")
        if self.cache and os.path.exists(cache_path):
            print("Reading data from cache...\n")
            self.data = pl.read_parquet(cache_path)
        else:
            manager = iohreader.DataManager()
            manager.add_folder(self.experiment_dir)

            cols = ['data_id', 'algorithm_name', 'batch_size', 'function_id',
                    'dimension', 'instance', 'time', 'evals', 'best_y', 'budget',
                    'random_seed', 'doe', 'evaluations', 'raw_y', 'raw_y_best']

            self.data = pl.concat([
                manager.select(algorithms=[algo], dimensions=[dim]).load(False, True)
                .filter(
                    (pl.col("algorithm_name").is_in(self.algorithms) if self.algorithms is not None else pl.lit(True)) &
                    (pl.col("batch_size").is_in(self.batch_sizes) if self.batch_sizes is not None else pl.lit(True)) &
                    (pl.col("dimension").is_in(self.dimensions) if self.dimensions is not None else pl.lit(True)) &
                    (pl.col("function_id").is_in(self.functions) if self.functions is not None else pl.lit(True))
                )
                .select(cols)
                .drop_nulls()
                for algo in manager.overview["algorithm_name"].unique().to_list()
                for dim in self.dimensions or manager.overview["dimension"].unique().to_list()
            ])

            if self.cache:
                print("Caching results...\n")
                self.data.write_parquet(cache_path)

    def analyze_convergence(self):
        """Analyze convergence metrics (AUC and final performance improvement)."""
        batch_sizes = self.batch_sizes or self.data["batch_size"].unique().sort()
        dimensions = self.dimensions or self.data["dimension"].unique().sort()
        functions = self.functions or self.data["function_id"].unique().sort()
        algorithms = self.algorithms or self.data["algorithm_name"].unique().sort()

        print("=== CONVERGENCE ANALYSIS ===")

        for batch_size in batch_sizes:
            for dimension in dimensions:
                for function in functions:
                    for algorithm in algorithms:
                        metrics = self._calculate_convergence_metrics(
                            batch_size, dimension, function, algorithm
                        )

                        if metrics is not None:
                            auc, final_improvement = metrics
                            print(f"batch_size: {batch_size}, dim: {dimension}, f: {function}, "
                                  f"algo: {algorithm} | auc: {auc:.6f}, "
                                  f"final_improvement_ratio: {final_improvement:.6f}")

    def _calculate_convergence_metrics(self, batch_size: int, dimension: int,
                                     function: int, algorithm: str) -> Optional[tuple]:
        """Calculate AUC and final performance improvement for specific configuration.

        Args:
            batch_size: Batch size.
            dimension: Dimension.
            function: Function ID.
            algorithm: Algorithm name.

        Returns:
            Tuple of (auc, final_improvement_ratio) or None if no data.
        """
        df = self.data.filter(
            (pl.col("batch_size") == batch_size) &
            (pl.col("dimension") == dimension) &
            (pl.col("function_id") == function) &
            (pl.col("algorithm_name") == algorithm)
        )

        if len(df) == 0:
            return None

        # Calculate metrics per run, then aggregate
        run_metrics = []

        for data_id in df["data_id"].unique():
            run_data = df.filter(pl.col("data_id") == data_id).sort("evaluations")

            if len(run_data) < 2:
                continue

            # Get initial and final values
            initial_value = run_data["raw_y_best"].first()
            final_value = run_data["raw_y_best"].last()

            # Skip if initial value is 0 or negative (would cause issues with ratio)
            if initial_value <= 0:
                continue

            # Calculate AUC using trapezoidal rule (normalized by evaluation budget)
            evaluations = run_data["evaluations"].to_numpy()
            values = run_data["raw_y_best"].to_numpy()

            # Normalize values by initial value for relative improvement
            normalized_values = values / initial_value

            # Calculate AUC using trapezoidal integration
            auc = np.trapz(normalized_values, evaluations)
            # Normalize by evaluation range to get average normalized performance
            eval_range = evaluations[-1] - evaluations[0]
            if eval_range > 0:
                auc = auc / eval_range
            else:
                auc = normalized_values[0]  # Single point case

            # Calculate final improvement ratio (how much better final vs initial)
            final_improvement = final_value / initial_value

            run_metrics.append((auc, final_improvement))

        if not run_metrics:
            return None

        # Aggregate across runs (mean)
        auc_values, final_improvements = zip(*run_metrics)
        mean_auc = np.mean(auc_values)
        mean_final_improvement = np.mean(final_improvements)

        return mean_auc, mean_final_improvement

    def analyze_times(self):
        """Analyze execution time statistics."""
        batch_sizes = self.batch_sizes or self.data["batch_size"].unique().sort()
        dimensions = self.dimensions or self.data["dimension"].unique().sort()
        functions = self.functions or self.data["function_id"].unique().sort()
        algorithms = self.algorithms or self.data["algorithm_name"].unique().sort()

        print("\n=== TIME ANALYSIS ===")

        # Get unique runs (one time per data_id) and ensure time is numeric
        time_df = (
            self.data.unique(subset=["data_id"], keep="first")
            .select(["algorithm_name", "batch_size", "dimension", "function_id", "time"])
            .with_columns([
                pl.col("time").cast(pl.Float64, strict=False).alias("time")
            ])
            .drop_nulls(subset=["time"])  # Remove rows where time couldn't be converted
        )

        for batch_size in batch_sizes:
            for dimension in dimensions:
                for function in functions:
                    for algorithm in algorithms:
                        stats = self._calculate_time_stats(
                            time_df, batch_size, dimension, function, algorithm
                        )

                        if stats is not None:
                            median_time, mean_time = stats
                            print(f"batch_size: {batch_size}, dim: {dimension}, f: {function}, "
                                  f"algo: {algorithm} | time_median: {median_time:.6f}, "
                                  f"time_mean: {mean_time:.6f}")

    def _calculate_time_stats(self, time_df: pl.DataFrame, batch_size: int,
                            dimension: int, function: int, algorithm: str) -> Optional[tuple]:
        """Calculate time statistics for specific configuration.

        Args:
            time_df: DataFrame with time data.
            batch_size: Batch size.
            dimension: Dimension.
            function: Function ID.
            algorithm: Algorithm name.

        Returns:
            Tuple of (median_time, mean_time) or None if no data.
        """
        filtered_df = time_df.filter(
            (pl.col("batch_size") == batch_size) &
            (pl.col("dimension") == dimension) &
            (pl.col("function_id") == function) &
            (pl.col("algorithm_name") == algorithm)
        )

        if len(filtered_df) == 0:
            return None

        try:
            # Get times and ensure they're numeric
            times = filtered_df["time"].to_numpy()

            # Filter out any remaining non-finite values
            times = times[np.isfinite(times)]

            if len(times) == 0:
                return None

            median_time = float(np.median(times))
            mean_time = float(np.mean(times))

            return median_time, mean_time

        except (TypeError, ValueError) as e:
            print(f"Warning: Could not calculate time stats for {algorithm} "
                  f"(batch_size: {batch_size}, dim: {dimension}, f: {function}): {e}")
            return None


def main():
    """Example usage of the ExperimentAnalyzer."""
    # Example configuration - adjust paths and parameters as needed
    experiment_dir = "./experiment"

    analyzer = ExperimentAnalyzer(
        experiment_dir=experiment_dir,
        algorithms=None,  # Use all algorithms
        batch_sizes=None,  # Use all batch sizes
        dimensions=None,   # Use all dimensions
        functions=None,    # Use all functions
        cache=True
    )

    analyzer.analyze_all()

if __name__ == "__main__":
    main()
