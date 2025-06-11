#!/usr/bin/env python3
"""
Entry point script for statistical analysis of Bayesian Optimization experiment results.
Performs Wilcoxon rank-sum tests comparing algorithms.
"""

import os
import argparse
from Algorithms.Experiment.ExperimentAnalyzer import ExperimentAnalyzer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Statistical analysis of Bayesian Optimization experiments"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment",
        help="Directory containing experiment data (default: experiment)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Directory to save analysis outputs (default: analysis_results)"
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        help="Algorithms to analyze (default: all found in data)"
    )

    parser.add_argument(
        "--batches",
        type=int,
        nargs="+",
        default=None,
        help="Batch sizes to analyze (default: all found in data)"
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=None,
        help="Dimensions to analyze (default: all found in data)"
    )

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Don't use cached results (default: False = use cache)"
    )

    return parser.parse_args()


def main():
    """Run the statistical analysis."""
    args = parse_arguments()

    # Verify experiment directory exists
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory '{args.experiment_dir}' not found.")
        print("Please run your experiment first to generate data.")
        return

    print("Bayesian Optimization Statistical Analysis")
    print("=" * 50)
    print(f"Experiment directory: {args.experiment_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Algorithms: {args.algorithms or 'all'}")
    print(f"Batch sizes: {args.batches or 'all'}")
    print(f"Dimensions: {args.dimensions or 'all'}")
    print(f"Cache: {not args.no_cache}")
    print()

    # Initialize and run analyzer
    analyzer = ExperimentAnalyzer(
        experiment_dir=args.experiment_dir,
        algorithms=args.algorithms,
        batch_sizes=args.batches,
        dimensions=args.dimensions,
        cache=not args.no_cache,
        output_dir=args.output_dir
    )

    # Run analysis
    results = analyzer.run_analysis()

    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()
