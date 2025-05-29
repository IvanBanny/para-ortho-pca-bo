#!/usr/bin/env python3
"""
Entry point script for running Bayesian Optimization comparison experiments in multiple processes.
This script configures and executes experiments comparing Vanilla BO, PCA-BO, and O-PCA-BO
on benchmark functions from the BBOB suite.
"""

import os
import argparse
from time import perf_counter
import torch

from Algorithms import ExperimentRunner


def parse_arguments():
    """Parse command line arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization comparison experiments."
    )

    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=["vanilla", "pca", "opca"],
        help="Algorithms to test (default: vanilla pca opca)"
    )

    parser.add_argument(
        "--batch",
        type=int,
        nargs="+",
        default=[1, 5],
        help="Batch sizes to test (default: 1 5)"
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
        default=[15, 18, 20, 21],
        help="BBOB problem IDs to test (default: 15, 18, 20, 21)"
    )

    parser.add_argument(
        "--instances",
        type=int,
        nargs="+",
        default=None,
        help="BBOB problem instances to test (default: None)"
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
        help="Factor for initial design size: n_doe = doe_factor * dim (default: 3)"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="experiment",
        help="Directory to store experiment results (default: experiment)"
    )

    parser.add_argument(
        "--acquisition",
        type=str,
        default="EI",
        choices=["EI", "PI"],
        help="Acquisition function to use (default: EI)"
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
        args.algorithms = ["vanilla", "pca", "opca"]
        args.batch = [1, 3]
        args.dimensions = [10]
        args.problems = [17, 20]
        args.runs = 5
        args.budget_factor = 5
        args.doe_factor = 2.0
        print("\nRunning in quick test mode with minimal settings")

    # Initialize experiment runner
    experiment = ExperimentRunner(
        algorithms=args.algorithms,
        batch_sizes=args.batch,
        dimensions=args.dimensions,
        problem_ids=args.problems,
        instances=args.instances,
        num_runs=args.runs,
        budget_factor=args.budget_factor,
        doe_factor=args.doe_factor,
        random_seed=69,
        acquisition_function=args.acquisition,
        var_threshold=args.var_threshold,
        root_dir=os.getcwd(),
        experiment_name=args.experiment_dir,
        torch_config={
            "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            "dtype": torch.float,
            "NUM_RESTARTS": 20,
            "RAW_SAMPLES": 4096,
            "OPTIMIZE_ACQF_OPTIONS": {"maxiter": 100, "method": "L-BFGS-B"}
        },
        verbose=args.verbose
    )

    # Print experiment configuration
    print("\nBayesian Optimization Experiment Configuration:")
    print(f"  Algorithms: {args.algorithms}")
    print(f"  Batch sizes: {args.batch}")
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
    start_time = perf_counter()
    experiment()
    total_time = perf_counter() - start_time

    print(f"\nExperiment completed in {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Results saved to {args.experiment_dir}")
    print("Run 'python plots.py' to visualize the results")


if __name__ == "__main__":
    main()
