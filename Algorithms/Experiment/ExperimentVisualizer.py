"""Visualization module for Bayesian Optimization experiments.

This module provides tools for loading, analyzing, and visualizing
the results of experiments comparing Vanilla BO, PCA-BO, and O-PCA-BO algorithms.
"""

from typing import List, Optional
import os
import numpy as np
from scipy import stats
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle

import Algorithms.utils.iohreader as iohreader


class ExperimentVisualizer:
    """Class to visualize and analyze experimental results from Bayesian Optimization algorithms."""

    def __init__(
            self,
            experiment_dir: str,
            algorithms: Optional[List[str]] = None,
            batch_sizes: Optional[List[int]] = None,
            dimensions: Optional[List[int]] = None,
            functions: Optional[List[int]] = None,
            ci: float = 0.95,
            save_figures: bool = True,
            output_dir: str = "visualizations",
            dpi: int = 300,
            cache: bool = True
    ):
        """Initialize the experiment visualizer.

        Args:
            experiment_dir: Directory containing experiment data.
            algorithms: Algorithms to plot (None for all).
            batch_sizes: Batch sizes to plot (None for all).
            dimensions: List of problem dimensions to plot (None for all).
            functions: List of function IDs to plot (None for all).
            ci: Confidence interval. Defaults to 95%.
            save_figures: Whether to save generated figures.
            output_dir: Directory to save visualization outputs. Defaults to "./visualizations".
            dpi: DPI for raster output formats.
            cache: Use cached results.
        """
        self.experiment_dir = experiment_dir
        self.algorithms = algorithms
        self.batch_sizes = batch_sizes
        self.dimensions = dimensions
        self.functions = functions
        self.ci = ci
        self.save_figures = save_figures
        self.output_dir = output_dir
        self.dpi = dpi
        self.cache = cache

        self.data = None

        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
                       '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    def plot_all(self) -> None:
        """Create all plots for the experiment data."""
        self.load_data()
        self.plot_convergence()
        self.plot_times()
        self.data = None

    def load_data(self):
        """Load experiment data using IOHreader (because IOHinspector is shit)."""

        cache_path = os.path.join(self.experiment_dir, "cache.parquet")
        if self.cache and os.path.exists(cache_path):
            print("\nReading data from cache...\n")
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
                print("\nCaching results...\n")
                self.data.write_parquet(cache_path)

    def plot_convergence(self):
        """Plot convergence graphs per batch sizes, per dimension, per function."""

        max_row_len = 5
        fig_size_margin = (1, 1)
        fig_size_cell = (2, 2.25)

        batch_sizes = self.batch_sizes or self.data["batch_size"].unique()
        dimensions = self.dimensions or self.data["dimension"].unique()
        functions = self.functions or self.data["function_id"].unique()

        row_len = min(max_row_len, len(functions))
        rows_per_dim = (len(functions) + row_len - 1) // row_len
        col_len = rows_per_dim * len(dimensions)

        fig_size = (fig_size_margin[0] + fig_size_cell[0] * row_len,
                    fig_size_margin[1] + fig_size_cell[1] * col_len)

        # Create individual plots for different batch sizes
        for batch_size in batch_sizes:
            fig = plt.figure(figsize=fig_size, dpi=self.dpi)
            fig.subplots_adjust(left=0.08, right=0.97, top=0.97, bottom=0.07)

            # Create a grid of subplots for different dimensions
            dims_gs = gridspec.GridSpec(len(dimensions), 1, hspace=0.15)

            # For each dimension section - fill cells
            for d_idx, dimension in enumerate(dimensions):
                gs = gridspec.GridSpecFromSubplotSpec(rows_per_dim, row_len, dims_gs[d_idx, 0],
                                                      hspace=0.15, wspace=0.2)

                # For each cell - add subplot to the dimension section gs
                for f_idx, function in enumerate(functions):
                    row, col = f_idx // row_len, f_idx % row_len
                    ax = fig.add_subplot(gs[row, col])

                    # Find if this is the last plot in its column
                    is_last_in_column = True
                    for check_f_idx in range(f_idx + 1, len(functions)):
                        check_row, check_col = check_f_idx // row_len, check_f_idx % row_len
                        if check_col == col:
                            is_last_in_column = False
                            break

                    # Hide x-axis if not the last in column
                    if not is_last_in_column:
                        ax.tick_params(labelbottom=False)

                    self._plot_individual_convergence(batch_size, dimension, function, ax)

                # Add vertical text labels on the left (one for each dimension)
                y_pos = 1 - (d_idx * 0.95 + 0.5) / len(dimensions)  # Center vertically for each dimension section
                fig.text(0.02, y_pos, f"f - f* in {dimension}D",
                         rotation=90, verticalalignment="center", fontsize=13, fontweight="normal")

            # Add horizontal text label at the bottom
            fig.text(0.5, 0.03, "iteration", horizontalalignment="center", fontsize=13, fontweight="normal")

            algorithms = self.algorithms or self.data["algorithm_name"].unique().sort().to_list()

            legend_handles = []
            for i, algorithm in enumerate(algorithms):
                handle = plt.Line2D([0], [0], color=self.colors[i], linewidth=2, label=algorithm)
                legend_handles.append(handle)

            fig.legend(handles=legend_handles, loc="lower center",
                       bbox_to_anchor=(0.5, 0.00), ncol=len(algorithms),
                       frameon=False, fontsize=10)

            if self.save_figures:
                plt.savefig(os.path.join(self.output_dir, f"convergence-b{batch_size}.png"), format="png")

    def _plot_individual_convergence(self, batch_size: int, dimension: int, function: int, ax: plt.Axes):
        """Plot convergence graph for a specific axis.

        Args:
            batch_size: Batch size.
            dimension: Dimension.
            function: Function ID.
            ax: Matplotlib axis to plot on.
        """

        df = self.data.filter(
            (pl.col("batch_size") == batch_size) &
            (pl.col("dimension") == dimension) &
            (pl.col("function_id") == function)
        )

        stats_df = (
            df
            .group_by(["algorithm_name", "evaluations"])
            .agg([
                pl.col("raw_y_best").mean().alias("mean_y"),
                pl.col("raw_y_best").std().alias("std_y"),
                pl.col("raw_y_best").count().alias("n_runs"),
            ])
            .with_columns([
                # Calculate margin of error for 95% CI using t-distribution
                pl.when(pl.col("n_runs") > 1)
                .then(
                    pl.col("std_y") / pl.col("n_runs").sqrt() *
                    pl.col("n_runs").map_elements(
                        lambda n: stats.t.ppf(0.5 + self.ci / 2, df=n-1), return_dtype=pl.Float64
                    )
                )
                .otherwise(0.0)
                .alias("margin_error")
            ])
            .sort(["algorithm_name", "evaluations"])
        )

        algorithms = self.algorithms or stats_df["algorithm_name"].unique().sort().to_list()

        for i, algorithm in enumerate(algorithms):
            algo_data = stats_df.filter(pl.col("algorithm_name") == algorithm)

            x = algo_data["evaluations"].to_list()
            y = algo_data["mean_y"].to_list()
            margin = algo_data["margin_error"].to_list()

            lower = [y_val - m for y_val, m in zip(y, margin)]
            upper = [y_val + m for y_val, m in zip(y, margin)]

            ax.plot(x, y, c=self.colors[i], label=algorithm, linewidth=0.5)
            ax.fill_between(x, lower, upper, color=self.colors[i], alpha=0.2)

        ax.set_yscale("log")

        # Calculate y-axis limits with 5% margin
        if len(stats_df) > 0:
            bounds_df = stats_df.filter(pl.col("algorithm_name").is_in(algorithms)).with_columns([
                (pl.col("mean_y") - pl.col("margin_error")).alias("lower_bound"),
                (pl.col("mean_y") + pl.col("margin_error")).alias("upper_bound")
            ])

            y_min = bounds_df["lower_bound"].min()
            y_max = bounds_df["upper_bound"].max()

            if y_min > 0 and y_max > 0:  # Ensure positive values for log scale
                # Add 5% margin on log scale
                log_range = np.log10(y_max) - np.log10(y_min)
                margin = 0.05 * log_range

                y_min_with_margin = 10 ** (np.log10(y_min) - margin)
                y_max_with_margin = 10 ** (np.log10(y_max) + margin)

                ax.set_ylim(y_min_with_margin, y_max_with_margin)

        # # Let matplotlib set automatic limits
        # ax.relim()
        # ax.autoscale()

        # Check how many ticks matplotlib generated
        current_ticks = ax.get_yticks()
        visible_ticks = [tick for tick in current_ticks if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]]

        # If we don't have at least 3 visible ticks, generate our own
        if len(visible_ticks) < 3:
            y_min, y_max = ax.get_ylim()

            # Generate 4 evenly spaced ticks in log space
            log_min = np.log10(y_min)
            log_max = np.log10(y_max)

            # Create 4 evenly spaced points in log space
            log_ticks = np.linspace(log_min, log_max, 4)
            ticks = [10 ** log_tick for log_tick in log_ticks]

            ax.set_yticks(ticks)

            # Format ticks as integers if they're reasonable integers, otherwise use scientific

        def integer_or_scientific_formatter(x, pos):
            if x == 0:
                return "0"
            elif x >= 1e4 or x <= 1e-2:
                # Use compact scientific notation
                exp = int(np.floor(np.log10(abs(x))))
                mantissa = x / (10 ** exp)
                mantissa_rounded = int(round(mantissa))
                if mantissa_rounded == 1:
                    return f"$10^{{{exp}}}$"
                else:
                    return f"${mantissa_rounded}\\,10^{{{exp}}}$"
            elif x >= 1e2:
                return f"{int(round(x))}"
            else:
                # For all other numbers, show as integer if close to one, otherwise 1 decimal place
                if abs(x - round(x)) < 0.05:
                    return f"{int(round(x))}"
                else:
                    return f"{x:.1f}"

        ax.yaxis.set_major_formatter(FuncFormatter(integer_or_scientific_formatter))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, pos: ''))
        ax.tick_params(axis="both", labelsize=6, length=2, width=0.5)

        ax.set_facecolor("#f8f8f8")
        ax.grid(True, color="white", linewidth=1)
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

        ax.set_title(f"F{function}", fontsize=10, fontweight="normal", pad=3)

    def plot_times(self):
        """Plot execution times for each dimension, algorithm, and function."""

        fig_size_margin = (0.2, 0.5)
        fig_size_cell = (5, 5)

        batch_sizes = self.batch_sizes or self.data["batch_size"].unique()
        dimensions = self.dimensions or self.data["dimension"].unique()

        fig_size = (fig_size_margin[0] + fig_size_cell[0] * len(dimensions),
                    fig_size_margin[1] + fig_size_cell[1])

        # Create individual plots for different batch sizes
        for batch_size in batch_sizes:
            fig = plt.figure(figsize=fig_size, dpi=self.dpi)
            fig.subplots_adjust(left=0.04, right=0.97, top=0.97, bottom=0.12)

            # Create a grid of subplots for different dimensions
            dims_gs = gridspec.GridSpec(1, len(dimensions), wspace=0.05)

            ax0 = None
            # For each dimension section - fill cells
            for d_idx, dimension in enumerate(dimensions):
                ax = fig.add_subplot(dims_gs[0, d_idx])
                if ax0 is None:
                    ax0 = ax
                else:
                    ax.sharey(ax0)

                # Hide y-axis if not the first in row
                if d_idx > 0:
                    ax.tick_params(labelleft=False)

                self._plot_individual_times(batch_size, dimension, ax)

                algorithms = self.algorithms or self.data["algorithm_name"].unique().sort().to_list()

                from matplotlib.patches import Rectangle

                legend_handles = []
                for i, algorithm in enumerate(algorithms):
                    handle = Rectangle(
                        (0, 0), 1, 1,
                        facecolor=self.colors[i],
                        edgecolor="black",
                        linewidth=0.8,
                        alpha=0.5,
                        label=algorithm
                    )
                    legend_handles.append(handle)

                fig.legend(handles=legend_handles, loc="lower center",
                           bbox_to_anchor=(0.5, 0.01), ncol=len(algorithms),
                           frameon=False, fontsize=10)

                if self.save_figures:
                    plt.savefig(os.path.join(self.output_dir, f"times-b{batch_size}.png"), format="png")

    def _plot_individual_times(self, batch_size: int, dimension: int, ax: plt.Axes):
        """Plot convergence graph for a specific axis.

        Args:
            batch_size: Batch size.
            dimension: Dimension.
            ax: Matplotlib axis to plot on.
        """
        def integer_or_scientific_formatter(x, pos):
            if x == 0:
                return "0"
            elif x >= 1e4 or x <= 1e-2:
                # Use compact scientific notation
                exp = int(np.floor(np.log10(abs(x))))
                mantissa = x / (10 ** exp)
                mantissa_rounded = int(round(mantissa))
                if mantissa_rounded == 1:
                    return f"$10^{{{exp}}}$"
                else:
                    return f"${mantissa_rounded}\\,10^{{{exp}}}$"
            elif x >= 1e2:
                return f"{int(round(x))}"
            else:
                # For all other numbers, show as integer if close to one, otherwise 1 decimal place
                if abs(x - round(x)) < 0.05:
                    return f"{int(round(x))}"
                else:
                    return f"{x:.1f}"

        df = (
            self.data.unique(subset=["data_id"], keep="first")
            .select(["algorithm_name", "batch_size", "dimension", "function_id", "time"])
            .filter((pl.col("batch_size") == batch_size) & (pl.col("dimension") == dimension))
        )

        algorithms = self.algorithms or df["algorithm_name"].unique()
        functions = self.functions or df["function_id"].unique()

        # Single group_by operation instead of nested filtering
        grouped = (
            df.group_by(["algorithm_name", "function_id"])
            .agg(pl.col("time").cast(pl.Float64))
        )

        # Convert to the desired structure efficiently
        time_data = {}
        for algorithm in algorithms:
            algo_data = grouped.filter(pl.col("algorithm_name") == algorithm)

            # Create function_id to times mapping
            func_times = {
                row["function_id"]: row["time"]
                for row in algo_data.to_dicts()
            }

            # Build matrix directly in desired orientation
            max_len = max(len(func_times.get(f, [])) for f in functions)
            matrix = np.full((max_len, len(functions)), np.nan)

            for func_idx, func in enumerate(functions):
                times = func_times.get(func, [])
                matrix[:len(times), func_idx] = times

            # Fill NaNs with row means
            row_means = np.nanmean(matrix, axis=1, keepdims=True)
            time_data[algorithm] = np.where(np.isnan(matrix), row_means, matrix)

        f_margin = 0.2
        c_margin = 0.075
        width = (1 - f_margin) / len(algorithms)
        positions = np.arange(len(functions))

        for i, algorithm in enumerate(algorithms):
            ax.boxplot(
                time_data[algorithm], positions=(positions + i * width), widths=width - c_margin,
                patch_artist=True, showfliers=True,
                label=algorithm, boxprops=dict(facecolor=self.colors[i], alpha=0.85),
                whiskerprops=dict(color="black", linewidth=0.5),
                capprops=dict(color="black", linewidth=0.5),
                medianprops=dict(color="black", linewidth=1),
                flierprops=dict(marker="o", markersize=4, color="black", alpha=0.7)
            )

        # Set function ticks
        ax.set_xticks(positions + (len(algorithms) - 1) * width / 2)
        ax.set_xticklabels([f"F{fid}" for fid in functions])

        # Decrease x-axis margins
        ax.set_xlim(-(width + f_margin) / 2, len(functions) - (width + f_margin) / 2)

        # Red dotted vertical separators
        for pos in (positions[1:] - (width + f_margin) / 2):
            ax.axvline(x=pos, color='red', linewidth=1, alpha=0.3,
                       linestyle='--', zorder=0)

        ax.set_yscale("log")

        # Check how many ticks matplotlib generated
        current_ticks = ax.get_yticks()
        visible_ticks = [tick for tick in current_ticks if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]]

        # If we don't have at least 10 visible ticks, generate our own
        if len(visible_ticks) < 10:
            y_min, y_max = ax.get_ylim()

            # Generate 10 evenly spaced ticks in log space
            log_min = np.log10(y_min)
            log_max = np.log10(y_max)

            # Create 10 evenly spaced points in log space
            log_ticks = np.linspace(log_min, log_max, 10)
            ticks = [10 ** log_tick for log_tick in log_ticks]

            ax.set_yticks(ticks)

        ax.yaxis.set_major_formatter(FuncFormatter(integer_or_scientific_formatter))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, pos: ''))
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)

        ax.set_facecolor("#f8f8f8")
        ax.grid(True, color="white", linewidth=1)
        ax.grid(False, axis='x')
        ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
