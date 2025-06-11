"""Statistical analysis module for Bayesian Optimization experiments.

This module provides tools for conducting Wilcoxon rank-sum tests comparing
Vanilla BO, PCA-BO, and O-PCA-BO algorithms.
"""

from typing import List, Optional
import os
from scipy import stats
import polars as pl
from itertools import combinations

import Algorithms.utils.iohreader as iohreader


class ExperimentAnalyzer:
    """Simple statistical analysis for Bayesian Optimization experiments."""

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
            algorithms: Algorithms to analyze (None for all).
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

    def wilcoxon_tests(self) -> pl.DataFrame:
        """Perform pairwise Wilcoxon rank-sum tests between algorithms."""
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
                data1 = condition_data.filter(pl.col("algorithm_name") == alg1)["final_performance"].to_list()
                data2 = condition_data.filter(pl.col("algorithm_name") == alg2)["final_performance"].to_list()

                if len(data1) == 0 or len(data2) == 0:
                    continue

                # Perform Wilcoxon rank-sum test
                try:
                    statistic, p_value = stats.ranksums(data1, data2)

                    mean1 = sum(data1) / len(data1)
                    mean2 = sum(data2) / len(data2)

                    # Determine winner (assuming minimization: lower is better)
                    if mean1 < mean2:
                        winner = alg1
                        winner_mean = mean1
                        loser = alg2
                        loser_mean = mean2
                    else:
                        winner = alg2
                        winner_mean = mean2
                        loser = alg1
                        loser_mean = mean1

                    results.append({
                        "batch_size": batch_size,
                        "dimension": dimension,
                        "algorithm_1": alg1,
                        "algorithm_2": alg2,
                        "n_samples_1": len(data1),
                        "n_samples_2": len(data2),
                        "mean_1": mean1,
                        "mean_2": mean2,
                        "winner": winner,
                        "winner_mean": winner_mean,
                        "loser": loser,
                        "loser_mean": loser_mean,
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    })

                except Exception as e:
                    print(f"Error in test {alg1} vs {alg2} (b{batch_size}, d{dimension}): {e}")
                    continue

        return pl.DataFrame(results) if results else pl.DataFrame()

    def run_analysis(self) -> pl.DataFrame:
        """Run complete analysis and save results."""
        print("Performing Wilcoxon rank-sum tests...")

        results = self.wilcoxon_tests()

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
                .select(["winner", "loser", "batch_size", "dimension", "winner_mean", "loser_mean", "p_value"])
            )

            print(f"{'Winner':<8} {'Loser':<8} {'Batch':<5} {'Dim':<3} {'Winner Mean':<11} {'Loser Mean':<11} {'p-value'}")
            print("-" * 75)
            for row in top_results.iter_rows(named=True):
                print(f"{row['winner']:<8} {row['loser']:<8} {row['batch_size']:<5} {row['dimension']:<3} "
                      f"{row['winner_mean']:<11.2f} {row['loser_mean']:<11.2f} {row['p_value']:.2e}")

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

        # Save results
        output_file = os.path.join(self.output_dir, "wilcoxon_results.csv")
        results.write_csv(output_file)
        print(f"\nResults saved to {output_file}")

        return results
