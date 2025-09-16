#!/usr/bin/env python3
"""
Entry point script for visualizing Bayesian Optimization experiment results.
This script loads, analyzes, and visualizes the results of experiments comparing
Vanilla BO and PCA-BO algorithms.
"""

import os
import argparse
from Algorithms import ExperimentVisualizer


def parse_arguments():
    """Parse command line arguments for visualization configuration."""
    parser = argparse.ArgumentParser(
        description="Visualize Bayesian Optimization experiment results"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=#r"C:\Users\Adela\PycharmProjects\para-ortho-pca-bo\experiments_20250708_194219\experiments", # C:\Users\Adela\PycharmProjects\para-ortho-pca-bo\experiments_20250610_113450", #  r"C:\Users\Adela\PycharmProjects\para-ortho-pca-bo\Adela_visualizations\gubic",
            r"C:\Users\Adela\PycharmProjects\para-ortho-pca-bo\experiments_20250803_162142",
        help="Directory containing experiment data (default: experiment)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualizations",
        help="Directory to save visualization outputs (default: ./visualizations)"
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
        "--functions",
        type=int,
        nargs="+",
        default=None,
        help="Function IDs to analyze (default: all found in data)"
    )

    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save visualization files (display only)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output formats (default: 300)"
    )

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Don't use cached results (default: False = use cache)"
    )

    return parser.parse_args()


def main():
    """Configure and run the visualization based on command line arguments."""
    args = parse_arguments()

    # Verify experiment directory exists
    if not os.path.exists(args.experiment_dir):
        print(f"Error: Experiment directory '{args.experiment_dir}' not found.")
        print("Please run 'python main.py' first to generate experiment data.")
        return

    # Initialize visualizer
    visualizer = ExperimentVisualizer(
        experiment_dir=args.experiment_dir,
        algorithms=args.algorithms,
        batch_sizes=args.batches,
        dimensions=args.dimensions,
        functions=args.functions,
        save_figures=not args.no_save,
        output_dir=args.output_dir,
        dpi=args.dpi,
        cache=not args.no_cache
    )

    print("\nBayesian Optimization Visualization Configuration:")
    print(f"  Experiment directory: {args.experiment_dir}")
    print(f"  Output directory: {visualizer.output_dir}")
    print(f"  Batch sizes: {args.algorithms or 'all'}")
    print(f"  Batch sizes: {args.batches or 'all'}")
    print(f"  Dimensions: {args.dimensions or 'all'}")
    print(f"  Functions: {args.functions or 'all'}")
    print(f"  Save figures: {not args.no_save}")
    print(f"  DPI: {args.dpi}")
    print(f"  Cache: {not args.no_cache}")

    # Create visualizations
    visualizer.plot_all()

    print("Visualization completed successfully!")
    if not args.no_save:
        print(f"All visualizations saved to {visualizer.output_dir}\n")


if __name__ == "__main__":
    main()
