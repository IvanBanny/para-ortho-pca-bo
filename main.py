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

from src import ExperimentRunner


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
        help="src to test (default: vanilla pca opca)"
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
        default=15,
        help="Budget factor for problem evaluations:"
             "budget = int((self.budget_factor * dim + 50) * (1 + 0.3 * log(batch_size))) (default: 15)"
    )

    parser.add_argument(
        "--doe_factor",
        type=int,
        default=4,
        help="Factor for initial design size: n_doe = doe_factor * dim (default: 4)"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="data",
        help="Directory to store experiment results (default: data)"
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
        "--gpr_p",
        type=float,
        default=0.842655,
        help="Percentage of ranked points to use in GPR fitting (default: 0.842655)"
    )

    parser.add_argument(
        "--gpr_val_factor",
        type=float,
        default=0.417592,
        help="Relative influence of value rank to distance rank in GPR fitting point selection (default: 0.417592)"
    )

    parser.add_argument(
        "--onorm_factor",
        type=float,
        default=8.0,
        help="O-norm sampling multiplier (default: 8.0)"
    )

    parser.add_argument(
        "--no_cont_acqf",
        action="store_true",
        help="Don't use continuous acqf penalization"
    )

    parser.add_argument(
        "--p_factor",
        type=float,
        default=1e-2,
        help="Penalty factor in pacqf (default: e1-2)"
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
        args.runs = 1
        args.budget_factor = 5.0
        args.doe_factor = 2.0
        args.experiment_dir = "./quick"
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
        gpr_p=args.gpr_p,
        gpr_val_factor=args.gpr_val_factor,
        onorm_factor=args.onorm_factor,
        cont_acqf=(not args.no_cont_acqf),
        p_factor=args.p_factor,
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
    print(f"  algorithms: {args.algorithms}")
    print(f"  Batch sizes: {args.batch}")
    print(f"  Dimensions: {args.dimensions}")
    print(f"  Problems: {args.problems}")
    print(f"  Instances: {range(args.runs) if args.instances is None else args.instances}")
    print(f"  Budget factor: {args.budget_factor}")
    print(f"  DoE factor: {args.doe_factor}")
    print(f"  Random_seed: {args.doe_factor}")
    print(f"  Acquisition function: {args.acquisition}")
    print(f"  PCA variance threshold: {args.var_threshold}")
    print(f"  gpr_p: {args.gpr_p}")
    print(f"  gpr_val_factor: {args.gpr_val_factor}")
    print(f"  onorm_factor: {args.onorm_factor}")
    print(f"  cont_acqf: {not args.no_cont_acqf}")
    print(f"  p_factor: {args.p_factor}")
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
