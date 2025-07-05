"""Statistical analysis module for Bayesian Optimization experiments.

This module provides tools for conducting paired Wilcoxon signed-rank tests comparing
Vanilla BO, PCA-BO, and O-PCA-BO algorithms.
"""

from typing import List, Optional
import os
from scipy import stats
import polars as pl
from itertools import combinations

import src.utils.iohreader as iohreader


class ExperimentAnalyzer:
    """Statistical analysis for Bayesian Optimization experiments using paired tests."""

    def __init__(
            self,
            experiment_dir: str,
            algorithms: Optional[List[str]] = None,
            batch_sizes: Optional[List[int]] = None,
            dimensions: Optional[List[int]] = None,
            cache: bool = True,
            output_dir: str = "analysis"
    ):
        """Initialize the experiment analyzer.

        Args:
            experiment_dir: Directory containing experiment data.
            algorithms: src to analyze (None for all).
            batch_sizes: Batch sizes to analyze (None for all).
            dimensions: Dimensions to analyze (None for all).
            cache: Use cached results.
            output_dir: Directory to save analysis outputs.
        """
        self.experiment_dir = experiment_dir
        self.algorithms = algorithms
        self.batch_sizes = batch_sizes
        self.dimensions = dimensions
        self.cache = cache
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)
        self.data = None

    def load_data(self):
        """Load experiment data using the same cache as visualizer."""
        cache_path = os.path.join(self.experiment_dir, "cache.parquet")

        if self.cache and os.path.exists(cache_path):
            print("Reading data from cache...")
            # Load full cache and select only needed columns
            full_data = pl.read_parquet(cache_path)
            self.data = full_data.select([
                'data_id', 'algorithm_name', 'batch_size', 'dimension', 'instance', 'raw_y_best'
            ]).filter(
                (pl.col("algorithm_name").is_in(self.algorithms) if self.algorithms is not None else pl.lit(True)) &
                (pl.col("batch_size").is_in(self.batch_sizes) if self.batch_sizes is not None else pl.lit(True)) &
                (pl.col("dimension").is_in(self.dimensions) if self.dimensions is not None else pl.lit(True))
            ).drop_nulls()
        else:
            print("Loading experimental data...")
            manager = iohreader.DataManager()
            manager.add_folder(self.experiment_dir)

            cols = ['data_id', 'algorithm_name', 'batch_size', 'dimension', 'instance', 'raw_y_best']

            self.data = pl.concat([
                manager.select(algorithms=[algo], dimensions=[dim]).load(False, True)
                .filter(
                    (pl.col("algorithm_name").is_in(self.algorithms) if self.algorithms is not None else pl.lit(True)) &
                    (pl.col("batch_size").is_in(self.batch_sizes) if self.batch_sizes is not None else pl.lit(True)) &
                    (pl.col("dimension").is_in(self.dimensions) if self.dimensions is not None else pl.lit(True))
                )
                .select(cols)
                .drop_nulls()
                for algo in manager.overview["algorithm_name"].unique().to_list()
                for dim in self.dimensions or manager.overview["dimension"].unique().to_list()
            ])

    def get_final_performance(self) -> pl.DataFrame:
        """Get final best performance for each run."""
        return (
            self.data
            .group_by("data_id")
            .agg([
                pl.col("algorithm_name").first(),
                pl.col("batch_size").first(),
                pl.col("dimension").first(),
                pl.col("instance").first(),
                pl.col("raw_y_best").last().alias("final_performance")
            ])
        )

    def paired_wilcoxon_tests(self) -> pl.DataFrame:
        """Perform pairwise paired Wilcoxon signed-rank tests between algorithms."""
        if self.data is None:
            self.load_data()

        final_perf = self.get_final_performance()

        # Get unique combinations of batch_size and dimension
        conditions = (
            final_perf
            .select(["batch_size", "dimension"])
            .unique()
            .sort(["batch_size", "dimension"])
        )

        results = []

        for condition in conditions.iter_rows(named=True):
            batch_size = condition["batch_size"]
            dimension = condition["dimension"]

            # Filter data for this condition
            condition_data = final_perf.filter(
                (pl.col("batch_size") == batch_size) &
                (pl.col("dimension") == dimension)
            )

            # Get algorithms present in this condition
            algorithms = condition_data["algorithm_name"].unique().to_list()

            if len(algorithms) < 2:
                continue

            # Perform pairwise comparisons
            for alg1, alg2 in combinations(algorithms, 2):
                # Get data for both algorithms
                data_alg1 = condition_data.filter(pl.col("algorithm_name") == alg1).select(["instance", "final_performance"])
                data_alg2 = condition_data.filter(pl.col("algorithm_name") == alg2).select(["instance", "final_performance"])

                # Join on instance to create paired data
                paired_data = data_alg1.join(
                    data_alg2,
                    on="instance",
                    how="inner",
                    suffix="_alg2"
                )

                if len(paired_data) < 3:  # Wilcoxon test needs at least 3 pairs
                    print(f"Warning: Not enough paired samples for {alg1} vs {alg2} (b{batch_size}, d{dimension}): {len(paired_data)} pairs")
                    continue

                # Extract paired values
                values_alg1 = paired_data["final_performance"].to_list()
                values_alg2 = paired_data["final_performance_alg2"].to_list()

                # Calculate differences for paired test
                differences = [v1 - v2 for v1, v2 in zip(values_alg1, values_alg2)]

                # Check if all differences are zero (would cause test to fail)
                if all(d == 0 for d in differences):
                    print(f"Warning: All differences are zero for {alg1} vs {alg2} (b{batch_size}, d{dimension})")
                    results.append({
                        "batch_size": batch_size,
                        "dimension": dimension,
                        "algorithm_1": alg1,
                        "algorithm_2": alg2,
                        "n_pairs": len(paired_data),
                        "mean_1": sum(values_alg1) / len(values_alg1),
                        "mean_2": sum(values_alg2) / len(values_alg2),
                        "mean_difference": 0.0,
                        "winner": "tie",
                        "winner_mean": sum(values_alg1) / len(values_alg1),
                        "loser": "tie",
                        "loser_mean": sum(values_alg2) / len(values_alg2),
                        "statistic": None,
                        "p_value": 1.0,
                        "significant": False
                    })
                    continue

                # Perform paired Wilcoxon signed-rank test
                try:
                    # Use two-sided test to detect any difference
                    statistic, p_value = stats.wilcoxon(values_alg1, values_alg2, alternative='two-sided')

                    mean1 = sum(values_alg1) / len(values_alg1)
                    mean2 = sum(values_alg2) / len(values_alg2)
                    mean_diff = sum(differences) / len(differences)

                    # Determine winner (assuming minimization: lower is better)
                    if mean1 < mean2:
                        winner = alg1
                        winner_mean = mean1
                        loser = alg2
                        loser_mean = mean2
                        # Adjust sign: winner - loser should be negative for minimization
                        display_diff = mean_diff
                    else:
                        winner = alg2
                        winner_mean = mean2
                        loser = alg1
                        loser_mean = mean1
                        # Flip sign since alg2 won but diff was calculated as alg1-alg2
                        display_diff = -mean_diff

                    results.append({
                        "batch_size": batch_size,
                        "dimension": dimension,
                        "algorithm_1": alg1,
                        "algorithm_2": alg2,
                        "n_pairs": len(paired_data),
                        "mean_1": mean1,
                        "mean_2": mean2,
                        "mean_difference": display_diff,
                        "winner": winner,
                        "winner_mean": winner_mean,
                        "loser": loser,
                        "loser_mean": loser_mean,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    })

                except Exception as e:
                    print(f"Error in paired test {alg1} vs {alg2} (b{batch_size}, d{dimension}): {e}")
                    continue

        return pl.DataFrame(results) if results else pl.DataFrame()

    def run_analysis(self) -> pl.DataFrame:
        """Run complete analysis and save results."""
        print("Performing paired Wilcoxon signed-rank tests...")

        results = self.paired_wilcoxon_tests()

        if len(results) == 0:
            print("No valid comparisons found!")
            return results

        # Print summary
        significant_tests = results.filter(pl.col("significant") == True)
        n_significant = len(significant_tests)
        n_total = len(results)

        print(f"\nResults: {n_significant}/{n_total} significant differences (p < 0.05)")

        if n_significant > 0:
            print("\nMost significant differences:")
            top_results = (
                significant_tests
                .sort("p_value")
                .head(50)
                .select(["winner", "loser", "batch_size", "dimension", "n_pairs", "winner_mean", "loser_mean", "mean_difference", "p_value"])
            )

            print(f"{'Winner':<8} {'Loser':<8} {'Batch':<5} {'Dim':<3} {'Pairs':<5} {'Winner Mean':<11} {'Loser Mean':<11} {'Mean Diff':<9} {'p-value'}")
            print("-" * 95)
            for row in top_results.iter_rows(named=True):
                print(f"{row['winner']:<8} {row['loser']:<8} {row['batch_size']:<5} {row['dimension']:<3} "
                      f"{row['n_pairs']:<5} {row['winner_mean']:<11.2f} {row['loser_mean']:<11.2f} "
                      f"{row['mean_difference']:<9.2f} {row['p_value']:.2e}")

        # Print overall winner summary
        if n_significant > 0:
            print(f"\nOverall winner summary:")
            winner_counts = {}
            for row in significant_tests.iter_rows(named=True):
                winner = row['winner']
                winner_counts[winner] = winner_counts.get(winner, 0) + 1

            print("Algorithm wins in significant comparisons:")
            for alg, count in sorted(winner_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {alg}: {count}/{n_significant} ({count/n_significant*100:.1f}%)")

        # Print pairing statistics
        print("\nPairing statistics:")
        pairing_stats = (
            results
            .group_by(["batch_size", "dimension"])
            .agg([
                pl.col("n_pairs").mean().alias("avg_pairs"),
                pl.col("n_pairs").min().alias("min_pairs"),
                pl.col("n_pairs").max().alias("max_pairs")
            ])
            .sort(["batch_size", "dimension"])
        )

        print(f"{'Batch':<5} {'Dim':<3} {'Avg Pairs':<9} {'Min Pairs':<9} {'Max Pairs'}")
        print("-" * 40)
        for row in pairing_stats.iter_rows(named=True):
            print(f"{row['batch_size']:<5} {row['dimension']:<3} {row['avg_pairs']:<9.1f} "
                  f"{row['min_pairs']:<9} {row['max_pairs']}")

        # Save results
        output_file = os.path.join(self.output_dir, "paired_wilcoxon_results.csv")
        results.write_csv(output_file)
        print(f"\nResults saved to {output_file}")

        return results
