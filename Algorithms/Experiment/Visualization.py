"""
Visualization module for Bayesian Optimization experiments.
This module provides tools for loading, analyzing, and visualizing
the results of experiments comparing Vanilla BO and PCA-BO algorithms.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import seaborn as sns

from iohinspector import DataManager
import polars as pl


class ExperimentVisualizer:
    """
    Class to visualize and analyze experimental results from Bayesian Optimization algorithms.

    Attributes:
        experiment_dir (str): Directory containing experiment data.
        dimensions (List[int]): List of problem dimensions used in the experiment.
        functions (List[int]): List of function IDs used in the experiment.
        output_dir (str): Directory to save visualization outputs.
        manager (DataManager): IOHinspector data manager.
        save_figures (bool): Whether to save generated figures.
        file_format (str): Output file format for saved figures.
        dpi (int): DPI for raster output formats.
    """

    def __init__(
            self,
            experiment_dir: str,
            dimensions: Optional[List[int]] = None,
            functions: Optional[List[int]] = None,
            output_dir: Optional[str] = None,
            save_figures: bool = True,
            file_format: str = "png",
            dpi: int = 300
    ):
        """
        Initialize the experiment visualizer.

        Args:
            experiment_dir: Directory containing experiment data.
            dimensions: List of problem dimensions to analyze (None for all).
            functions: List of function IDs to analyze (None for all).
            output_dir: Directory to save visualization outputs (None to use experiment_dir/visualizations).
            save_figures: Whether to save generated figures.
            file_format: Output file format for saved figures (png, pdf, svg).
            dpi: DPI for raster output formats.
        """
        self.experiment_dir = experiment_dir
        self.dimensions = dimensions
        self.functions = functions
        self.output_dir = output_dir or os.path.join(experiment_dir, 'visualizations')
        self.save_figures = save_figures
        self.file_format = file_format
        self.dpi = dpi

        # Create output directory if it doesn't exist
        if self.save_figures and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Initialize IOHinspector data manager
        self.manager = DataManager()
        self.manager.add_folder(experiment_dir)

        # Detect dimensions and functions if not specified
        if self.dimensions is None:
            self.dimensions = sorted(self.manager.overview['dimension'].unique().to_list())
        if self.functions is None:
            self.functions = sorted(self.manager.overview['function_id'].unique().to_list())

        print(f"Found data for dimensions: {self.dimensions}")
        print(f"Found data for functions: {self.functions}")

    def create_all_visualizations(self) -> None:
        """
        Create all visualizations for the experiment data.
        """
        # Load data
        print("Loading experiment data...")
        df = self.load_data()

        # Create convergence plots
        print("Creating convergence plots...")
        self.plot_convergence_by_dimension(df)

        # Create execution time plots
        print("Creating execution time plots...")
        self.plot_execution_times()

        # Analyze algorithm comparison
        print("Analyzing algorithm performance...")
        comparison_df = self.analyze_algorithm_comparison(df)
        print("\nAlgorithm Comparison Summary:")
        print(comparison_df)

    def load_data(self, make_monotonic: bool = True, include_metadata: bool = True) -> pl.DataFrame:
        """
        Load experiment data using IOHinspector.

        Args:
            make_monotonic: Whether to make the performance data monotonic.
            include_metadata: Whether to include metadata in the loaded data.

        Returns:
            DataFrame containing the loaded data.
        """
        df = self.manager.load(make_monotonic, include_metadata)
        print(f"Loaded data with shape: {df.shape}")
        return df

    def plot_convergence_by_dimension(
            self,
            df: pl.DataFrame
    ) -> Dict[int, plt.Figure]:
        """
        Plot convergence graphs for each dimension, showing all functions.

        Args:
            df: DataFrame with experiment data.

        Returns:
            Dictionary mapping dimensions to figure objects.
        """
        figures = {}

        for dim in self.dimensions:
            # Filter data for the current dimension
            dim_data = df.filter(pl.col("dimension") == dim)

            # Create figure with subplots for each function
            n_functions = len(self.functions)
            fig, axes = plt.subplots(1, n_functions, figsize=(5 * n_functions, 5), sharey=False)

            if n_functions == 1:
                axes = [axes]  # Ensure axes is always a list

            # Plot each function
            for i, func_id in enumerate(self.functions):
                ax = axes[i]
                func_data = dim_data.filter(pl.col("function_id") == func_id)

                # Get function name
                func_name = func_data.select(pl.col("function_name")).unique()[0, 0]

                # Plot fixed-target performance
                data_ft = self._plot_function_convergence(func_data, ax)

                # Set labels and title
                ax.set_xlabel("Iteration")
                ax.set_ylabel("f - f*") if i == 0 else None
                ax.set_title(f"F{func_id}: {func_name}")
                ax.grid(True, alpha=0.3)

                # Use log scale for y-axis if values have large range
                min_val = float(data_ft["f-f*"].min())
                max_val = float(data_ft["f-f*"].max())
                if max_val / max(min_val, 1e-10) > 100:
                    ax.set_yscale('log')

            # Add overall title and legend
            fig.suptitle(f"Convergence Curves for {dim}D Problems", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

            # Save figure if requested
            if self.save_figures:
                output_path = os.path.join(
                    self.output_dir, f"convergence_{dim}D.{self.file_format}"
                )
                fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
                print(f"Saved convergence plot to {output_path}")

            figures[dim] = fig

        return figures

    def _plot_function_convergence(
            self,
            func_data: pl.DataFrame,
            ax: plt.Axes
    ) -> pd.DataFrame:
        """
        Plot convergence graph for a specific function on a given axis.

        Args:
            func_data: DataFrame containing data for a specific function.
            ax: Matplotlib axis to plot on.

        Returns:
            DataFrame used for the plot.
        """
        # Convert to pandas for easier plotting
        df_pandas = func_data.to_pandas()

        # Calculate target precision (f - f*)
        if 'raw_y_best' in df_pandas.columns:
            df_pandas['f-f*'] = df_pandas['raw_y_best']
        else:
            df_pandas['f-f*'] = df_pandas['raw_y']

        # Group by algorithm and iteration
        grouped = (df_pandas.groupby(['algorithm_name', 'evaluations'])
                   .agg({'f-f*': ['mean', 'std']})
                   .reset_index())
        grouped.columns = ['algorithm_name', 'evaluations', 'f-f*', 'std']

        # Plot for each algorithm
        for algorithm, color in zip(['VANILLA BO', 'PCA BO'], ['red', 'blue']):
            alg_data = grouped[grouped['algorithm_name'] == algorithm]

            if len(alg_data) > 0:
                # Sort by evaluations
                alg_data = alg_data.sort_values('evaluations')

                # Plot mean with confidence interval
                ax.plot(alg_data['evaluations'], alg_data['f-f*'],
                        label=algorithm, color=color, linewidth=2)

                # Add confidence interval (95%)
                lower = alg_data['f-f*'] - 1.96 * alg_data['std']
                upper = alg_data['f-f*'] + 1.96 * alg_data['std']
                ax.fill_between(alg_data['evaluations'], lower, upper,
                                color=color, alpha=0.2)

        # Add legend
        ax.legend()

        return grouped

    def plot_execution_times(self) -> Optional[plt.Figure]:
        """
        Plot execution times for each algorithm, dimension, and function.

        Returns:
            Matplotlib figure object or None if execution time data is missing.
        """
        # Get execution time files
        vanilla_file = os.path.join(self.experiment_dir, "vanilla_execution_times.csv")
        pca_file = os.path.join(self.experiment_dir, "pca_execution_times.csv")

        if not (os.path.exists(vanilla_file) and os.path.exists(pca_file)):
            print("Execution time files not found.")
            return None

        # Load execution time data
        vanilla_times = pd.read_csv(vanilla_file)
        pca_times = pd.read_csv(pca_file)

        # Prepare data for plotting
        vanilla_times['algorithm'] = 'Vanilla BO'
        pca_times['algorithm'] = 'PCA BO'

        # Combine data
        all_times = pd.concat([vanilla_times, pca_times])

        # Create figure
        fig, axes = plt.subplots(1, len(self.dimensions), figsize=(15, 5), sharey=True)

        if len(self.dimensions) == 1:
            axes = [axes]  # Ensure axes is always a list

        # Plot for each dimension
        for i, dim in enumerate(self.dimensions):
            ax = axes[i]

            # Filter data for current dimension
            dim_data = all_times[all_times['dimension'] == dim]

            # Create boxplot
            sns.boxplot(
                x='function_id',
                y='time',
                hue='algorithm',
                data=dim_data,
                palette=['red', 'blue'],
                ax=ax
            )

            # Set labels and title
            ax.set_xlabel("Function ID")
            ax.set_ylabel("Execution Time (s)") if i == 0 else None
            ax.set_title(f"{dim}D Problems")
            ax.grid(True, alpha=0.3)

            # Format x-axis labels
            ax.set_xticklabels([f"F{x}" for x in sorted(dim_data['function_id'].unique())])

            # Use log scale for y-axis
            ax.set_yscale('log')

            # Only show legend in first subplot
            if i > 0:
                ax.get_legend().remove()

        # Add overall title
        fig.suptitle("Algorithm Execution Times", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle

        # Save figure if requested
        if self.save_figures:
            output_path = os.path.join(
                self.output_dir, f"execution_times.{self.file_format}"
            )
            fig.savefig(output_path, bbox_inches='tight', dpi=self.dpi)
            print(f"Saved execution time plot to {output_path}")

        return fig

    def analyze_algorithm_comparison(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        Analyze and compare the performance of Vanilla BO and PCA-BO.

        Args:
            df: DataFrame containing experiment data.

        Returns:
            DataFrame with comparison statistics.
        """
        # Convert to pandas for easier analysis
        df_pandas = df.to_pandas()

        # Initialize results dictionary
        results = []

        # Analyze each dimension and function combination
        for dim in self.dimensions:
            for func_id in self.functions:
                # Filter data
                dim_func_data = df_pandas[
                    (df_pandas['dimension'] == dim) &
                    (df_pandas['function_id'] == func_id)
                    ]

                # Get best final values for each algorithm and run
                best_values = (dim_func_data.groupby(['algorithm_name', 'run_id'])
                               .agg({'raw_y_best': 'min'})
                               .reset_index())

                # Split by algorithm
                vanilla_results = best_values[best_values['algorithm_name'] == 'VANILLA BO']
                pca_results = best_values[best_values['algorithm_name'] == 'PCA BO']

                # Handle empty results
                if len(vanilla_results) == 0 or len(pca_results) == 0:
                    print(f"Warning: Missing data for dimension {dim}, function {func_id}")
                    continue

                # Calculate statistics
                vanilla_mean = vanilla_results['raw_y_best'].mean()
                vanilla_std = vanilla_results['raw_y_best'].std()
                pca_mean = pca_results['raw_y_best'].mean()
                pca_std = pca_results['raw_y_best'].std()

                # Calculate improvement
                improvement = 0
                if vanilla_mean != 0:
                    improvement = (vanilla_mean - pca_mean) / vanilla_mean * 100

                # Add to results
                results.append({
                    'dimension': dim,
                    'function_id': func_id,
                    'vanilla_mean': vanilla_mean,
                    'vanilla_std': vanilla_std,
                    'pca_mean': pca_mean,
                    'pca_std': pca_std,
                    'improvement': improvement
                })

        # Convert to DataFrame
        results_df = pd.DataFrame(results)

        # Save to CSV
        if self.save_figures:
            output_path = os.path.join(self.output_dir, "algorithm_comparison.csv")
            results_df.to_csv(output_path, index=False)
            print(f"Saved algorithm comparison to {output_path}")

        return results_df
