#!/usr/bin/env python3
"""
Entry point script for visualizing Bayesian Optimization experiment results.
This script loads, analyzes, and visualizes the results of experiments comparing
Vanilla BO and PCA-BO algorithms.
"""

import os
import argparse
from AlgorithmsOLD.Experiment.Visualization import ExperimentVisualizer


def parse_arguments():
    """Parse command line arguments for visualization configuration."""
    parser = argparse.ArgumentParser(
        description="Visualize Bayesian Optimization experiment results"
    )

    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="pca-bo-experiment",
        help="Directory containing experiment data (default: pca-bo-experiment)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save visualization outputs (default: experiment_dir/visualizations)"
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
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output file format for visualizations (default: png)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for raster output formats (default: 300)"
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

    # Set up output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(args.experiment_dir, "visualizations")

    # Initialize visualizer
    visualizer = ExperimentVisualizer(
        experiment_dir=args.experiment_dir,
        dimensions=args.dimensions,
        functions=args.functions,
        output_dir=output_dir,
        save_figures=not args.no_save,
        file_format=args.format,
        dpi=args.dpi
    )

    print("Bayesian Optimization Visualization Configuration:")
    print(f"  Experiment directory: {args.experiment_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Dimensions: {args.dimensions or 'all'}")
    print(f"  Functions: {args.functions or 'all'}")
    print(f"  Save figures: {not args.no_save}")
    print(f"  File format: {args.format}")
    print(f"  DPI: {args.dpi}\n")

    # Create visualizations
    visualizer.create_all_visualizations()

    print("\nVisualization completed successfully!")
    if not args.no_save:
        print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
