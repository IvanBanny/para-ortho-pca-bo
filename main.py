#!/usr/bin/env python3
"""
Entry point script for running Bayesian Optimization comparison experiments.
This script configures and executes experiments comparing Vanilla BO and PCA-BO
on benchmark functions from the BBOB suite.
"""

import os
import argparse
import time
import torch

from Algorithms import ExperimentRunner


def parse_arguments():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization comparison experiments."
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help="Problem dimensions to test (default: 10 20 40)"
    )

    parser.add_argument(
        "--problems",
        type=int,
        nargs="+",
        default=[15, 16, 17],
        help="BBOB problem IDs to test (default: 15 16 17)"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of independent runs per problem and dimension (default: 30)"
    )

    parser.add_argument(
        "--budget_factor",
        type=int,
        default=10,
        help="Budget factor for problem evaluations: budget = budget_factor * dim + 50 (default: 10)"
    )

    parser.add_argument(
        "--doe_factor",
        type=int,
        default=3,
        help="Factor for initial design size: n_doe = doe_factor * dim (default: 3.0)"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment",
        help="Directory to store experiment results (default: pca-bo-experiment)"
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="expected_improvement",
        choices=["expected_improvement", "probability_of_improvement", "upper_confidence_bound"],
        help="Acquisition function to use (default: expected_improvement)"
    )

    parser.add_argument(
        "--var_threshold",
        type=float,
        default=0.95,
        help="Variance threshold for PCA component selection (default: 0.95)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a minimal experiment for quick testing"
    )

    return parser.parse_args()


def main():
    """Configure and run the experiment based on command line arguments."""
    args = parse_arguments()

    # For quick testing, override with minimal settings if --quick flag is used
    if args.quick:
        args.dimensions = [5]  # Use only 5D
        args.problems = [15, 20]  # Use only problems 15 and 20
        args.runs = 30  # Just 30 runs
        args.budget_factor = 5  # Small budget
        args.doe_factor = 2.0  # Small DoE
        print("\nRunning in quick test mode with minimal settings")

    # Initialize experiment runner
    experiment = ExperimentRunner(
        algorithms=["clean-pca"],
        dimensions=args.dimensions,
        problem_ids=args.problems,
        num_runs=args.runs,
        budget_factor=args.budget_factor,
        doe_factor=args.doe_factor,
        root_dir=os.getcwd(),
        experiment_name=args.experiment_dir,
        acquisition_function=args.acquisition,
        pca_components=0,  # Automatic selection based on var_threshold
        var_threshold=args.var_threshold,
        #torch_config={
        #    "device": torch.device("cpu"),
        #    "dtype": torch.float,
        #    "BATCH_SIZE": 1,
        #    "NUM_RESTARTS": 20,
        #    "RAW_SAMPLES": 1024
        #},
        verbose=args.verbose
    )

    # Print experiment configuration
    print("\nBayesian Optimization Experiment Configuration:")
    print(f"  Dimensions: {args.dimensions}")
    print(f"  Problems: {args.problems}")
    print(f"  Runs: {args.runs}")
    print(f"  Budget factor: {args.budget_factor}")
    print(f"  DoE factor: {args.doe_factor}")
    print(f"  Acquisition function: {args.acquisition}")
    print(f"  PCA variance threshold: {args.var_threshold}")
    print(f"  Output directory: {args.experiment_dir}")
    print(f"  Verbose mode: {args.verbose}")
    print("\nStarting experiment...\n")

    # Run the experiment with timing
    start_time = time.time()
    experiment.run_experiment()
    total_time = time.time() - start_time

    print(f"\nExperiment completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Results saved to {args.experiment_dir}")
    print("Run 'python plot_results.py' to visualize the results")


if __name__ == "__main__":
    main()
