import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.cm import get_cmap
import torch
from typing import List, Optional, Tuple, Callable, Union
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from PIL import Image
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import matplotlib.colors as colors
from scipy.interpolate import griddata


class PCABOVisualizer:
    """Visualizer class for PCA-BO to create animations of the acquisition function.

    This class generates visualizations of how the acquisition function evolves
    along the first principal component dimension throughout optimization.
    """

    def __init__(
            self,
            output_dir: str = "./visualizations",
            fig_size: Tuple[int, int] = (12, 10),
            dpi: int = 100,
            cmap: str = "viridis",
            n_points: int = 200,
            n_contour_points: int = 50,
            fps: int = 1,
            margin: float = 0.2,  # Margin for plots as a fraction of range
            save_frames: bool = False  # Only save the final gif, not individual frames
    ):
        """Initialize the PCABOVisualizer.

        Args:
            output_dir: Directory to save the visualizations
            fig_size: Figure size for the plots
            dpi: DPI for the plots
            cmap: Colormap to use for the acquisition function
            n_points: Number of points to evaluate the acquisition function
            n_contour_points: Number of points in each dimension for contour plot grid
            fps: Frames per second for the gif
            margin: Margin to add around the data (as a fraction of data range)
            save_frames: Whether to save individual frames
        """
        self.output_dir = output_dir
        self.fig_size = fig_size
        self.dpi = dpi
        self.cmap = get_cmap(cmap)
        self.n_points = n_points
        self.n_contour_points = n_contour_points
        self.fps = fps
        self.margin = margin
        self.save_frames = save_frames

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Initialize figure with Agg backend for consistent rendering
        self.fig = Figure(figsize=self.fig_size, dpi=self.dpi)
        self.canvas = FigureCanvasAgg(self.fig)

        # Create a grid layout
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

        # Create two subplots
        self.ax1 = self.fig.add_subplot(self.gs[0])  # Acquisition function
        self.ax2 = self.fig.add_subplot(self.gs[1])  # Contour plot

        # Storage for frames and data
        self.frames = []
        self.iteration = 0
        self.z_evals_history = []
        self.f_evals_history = []
        self.acqf_history = []
        self.bounds = None  # Original problem bounds

    def visualize_acqf(
            self,
            acquisition_function: Callable,
            z_evals: List[np.ndarray],
            f_evals: List[float],
            x_evals: List[np.ndarray],
            component_matrix: np.ndarray,
            current_best_index: int,
            reduced_space_dim_num: int,
            bounds: np.ndarray,
            iteration: int
    ) -> None:
        """Create a visualization of the acquisition function along the first PC.

        Args:
            acquisition_function: The acquisition function to visualize
            z_evals: List of points in the reduced space
            f_evals: Function evaluations at each point
            x_evals: List of points in the original problem space
            component_matrix: PCA component matrix
            current_best_index: Index of the current best point
            reduced_space_dim_num: Number of dimensions in the reduced space
            bounds: Bounds in the original problem space [[min1, max1], [min2, max2], ...]
            iteration: Current iteration number
        """
        self.iteration = iteration
        self.bounds = bounds

        # Store data for history
        self.z_evals_history = z_evals.copy()
        self.f_evals_history = f_evals.copy()
        self.x_evals = x_evals.copy()

        # Create a new figure for each iteration to avoid any potential memory issues
        plt.close(self.fig)
        self.fig = Figure(figsize=self.fig_size, dpi=self.dpi)
        self.canvas = FigureCanvasAgg(self.fig)

        # Create a grid layout
        self.gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=self.fig)

        # Create two subplots
        self.ax1 = self.fig.add_subplot(self.gs[0])  # Acquisition function
        self.ax2 = self.fig.add_subplot(self.gs[1])  # Contour plot

        # Plot acquisition function in reduced space (first subplot)
        self._plot_acquisition_function(
            acquisition_function,
            z_evals,
            current_best_index,
            reduced_space_dim_num
        )

        # Plot contour in original space (second subplot)
        self._plot_objective_contour(
            x_evals,
            f_evals,
            component_matrix,
            current_best_index,
            bounds
        )

        # Add overall title
        self.fig.suptitle(f'PCA-BO Visualization (Iteration {iteration})',
                          fontsize=16, y=0.98)

        # Add spacing between subplots and ensure proper layout
        self.fig.tight_layout(rect=[0, 0, 1, 0.95])

        # Draw the figure and save it as a frame
        self.canvas.draw()

        # Convert canvas to PIL Image
        buf = io.BytesIO()
        self.fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image = Image.open(buf)
        self.frames.append(image.copy())

        # Save the current frame as a snapshot if enabled
        if self.save_frames:
            self.fig.savefig(f"{self.output_dir}/pca_bo_iteration_{iteration:04d}.png", bbox_inches='tight')

    def _plot_acquisition_function(
            self,
            acquisition_function: Callable,
            z_evals: List[np.ndarray],
            current_best_index: int,
            reduced_space_dim_num: int
    ) -> None:
        """Plot the acquisition function along the first principal component.

        Args:
            acquisition_function: The acquisition function to visualize
            z_evals: List of points in the reduced space
            current_best_index: Index of the current best point
            reduced_space_dim_num: Number of dimensions in the reduced space
        """
        # Calculate bounds for the reduced space with extra margin
        if len(z_evals) > 1:
            z_array = np.vstack(z_evals)
            z_min = np.min(z_array, axis=0)
            z_max = np.max(z_array, axis=0)
            z_range = z_max - z_min
            # Add padding with specified margin
            reduced_space_bounds = np.vstack([
                z_min - self.margin * z_range,
                z_max + self.margin * z_range
            ]).T
        else:
            # Default bounds if we have only one point
            reduced_space_bounds = np.vstack([
                -np.ones(reduced_space_dim_num) * (1 + self.margin),
                np.ones(reduced_space_dim_num) * (1 + self.margin)
            ]).T

        # If reduced space is more than 1D, just use the first dimension
        if reduced_space_bounds.shape[0] > 1:
            bounds = reduced_space_bounds[0]  # Get bounds for the first PC
        else:
            bounds = reduced_space_bounds[0]

        # Create a grid of points along the first PC dimension to evaluate acquisition function
        z_grid = np.linspace(bounds[0], bounds[1], self.n_points)

        # Format grid for acquisition function (assuming it expects a torch tensor)
        def evaluate_acqf_1d(z_values):
            """Evaluate acquisition function for 1D points."""
            # Convert to tensor and reshape for batch processing
            if len(z_evals) > 0 and isinstance(z_evals[0], np.ndarray):
                pc_dim = z_evals[0].shape[0]
            else:
                pc_dim = 1

            # Create zeros for other dimensions if needed
            if pc_dim > 1:
                z_batch = np.zeros((len(z_values), pc_dim))
                z_batch[:, 0] = z_values  # Set first dimension to our grid values
            else:
                z_batch = z_values.reshape(-1, 1)

            # Convert to torch tensor
            try:
                device = acquisition_function.model.device
            except AttributeError:
                # Default to CPU if device not found
                device = torch.device("cpu")

            try:
                dtype = next(acquisition_function.model.parameters()).dtype
            except (AttributeError, StopIteration):
                # Default to float32 if dtype not found
                dtype = torch.float32

            # Create tensor and add required q=1 dimension for BoTorch acquisition functions
            z_tensor = torch.tensor(z_batch, device=device, dtype=dtype)
            z_tensor = z_tensor.unsqueeze(1)  # Add q=1 dimension [batch_size, 1, dim]

            # Evaluate acquisition function
            with torch.no_grad():
                try:
                    acq_values = acquisition_function(z_tensor).cpu().numpy()
                except Exception as e:
                    print(f"Error evaluating acquisition function: {e}")
                    acq_values = np.zeros(len(z_values))

            return acq_values

        try:
            # Try to evaluate acquisition function on grid
            acq_values = evaluate_acqf_1d(z_grid)
            self.acqf_history.append((z_grid, acq_values))
        except Exception as e:
            print(f"Warning: Could not evaluate acquisition function: {e}")
            # Use dummy values if evaluation fails
            acq_values = np.zeros_like(z_grid)
            self.acqf_history.append((z_grid, acq_values))

        # Plot acquisition function
        self.ax1.plot(z_grid, acq_values, color='blue', linewidth=2, alpha=0.8,
                      label='Acquisition Function')

        # Plot the points in reduced space
        if len(z_evals) > 0:
            z_points = np.array([z[0] if len(z.shape) > 0 else z for z in z_evals])  # Extract first component

            # Plot all evaluated points
            self.ax1.scatter(z_points, np.zeros_like(z_points), color='#888888', s=40, alpha=0.7,
                             label='Evaluated Points', zorder=3)

            # Highlight the best point
            if current_best_index < len(z_points):
                self.ax1.scatter(z_points[current_best_index], 0, color='#00aa00', s=120,
                                 edgecolor='black', linewidth=1.5,
                                 label='Best Point', zorder=5)

            # Highlight the latest point
            if len(z_points) > 0:
                self.ax1.scatter(z_points[-1], 0, color='#ff3333', s=100,
                                 edgecolor='black', linewidth=1.5,
                                 label='Latest Point', zorder=4)

        # Plot vertical lines for bounds
        bounds_original = None
        if len(z_evals) > 0 and isinstance(z_evals[0], np.ndarray) and len(z_evals[0]) > 0:
            # Find the min and max of the first PC from actual data points
            z_points_array = np.array([z[0] if len(z.shape) > 0 else z for z in z_evals])
            bounds_original = [np.min(z_points_array), np.max(z_points_array)]

        # Use original PC range if available, otherwise use calculated bounds
        if bounds_original is not None:
            bounds_to_plot = bounds_original
        else:
            # Default to removing some margin from calculated bounds
            bounds_to_plot = [bounds[0] * 0.9, bounds[1] * 0.9]

        self.ax1.axvline(bounds_to_plot[0], color='red', linestyle='--', alpha=0.6, label='Bounds')
        self.ax1.axvline(bounds_to_plot[1], color='red', linestyle='--', alpha=0.6)

        # Set labels and title with modern styling
        self.ax1.set_xlabel('First Principal Component', fontsize=12, fontweight='bold')
        self.ax1.set_ylabel('Acquisition Function Value', fontsize=12, fontweight='bold')
        self.ax1.set_title('Acquisition Function in PCA Space', fontsize=14, pad=10)

        # Add legend with modern styling
        handles, labels = self.ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax1.legend(by_label.values(), by_label.keys(),
                        loc='upper right', fontsize=10, framealpha=0.9,
                        edgecolor='lightgray')

        # Set y-axis limits with more padding so we can see the full function
        if len(acq_values) > 0:
            y_min = min(0, np.min(acq_values) * 1.1)
            y_max = max(0.1, np.max(acq_values) * 1.1)
            self.ax1.set_ylim(y_min, y_max)

        # Modern aesthetics
        self.ax1.spines['top'].set_visible(False)
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['left'].set_linewidth(0.5)
        self.ax1.spines['bottom'].set_linewidth(0.5)
        self.ax1.grid(True, linestyle='--', alpha=0.3)

    def _plot_objective_contour(
            self,
            x_evals: List[np.ndarray],
            f_evals: List[float],
            component_matrix: np.ndarray,
            current_best_index: int,
            bounds: np.ndarray
    ) -> None:
        """Plot a 2D contour of the objective function and the principal components.

        Args:
            x_evals: List of points in the original problem space
            f_evals: Function evaluations at each point
            component_matrix: PCA component matrix
            current_best_index: Index of the current best point
            bounds: Bounds in the original problem space [[min1, max1], [min2, max2], ...]
        """
        if len(x_evals) < 3:
            # Not enough points to create a meaningful contour
            self.ax2.text(0.5, 0.5, "Not enough points for contour plot",
                          ha='center', va='center', fontsize=14)
            return

        # Stack points and convert to numpy arrays
        x_array = np.vstack(x_evals)
        f_array = np.array(f_evals)

        # Use only the first two dimensions for the contour plot
        if x_array.shape[1] >= 2:
            x_plot = x_array[:, :2]  # First two dimensions
        else:
            x_plot = np.column_stack((x_array[:, 0], np.zeros(len(x_array))))

        # Calculate plot limits with margin
        if bounds is not None and bounds.shape[0] >= 2:
            x_min, x_max = bounds[0, 0], bounds[0, 1]
            y_min, y_max = bounds[1, 0], bounds[1, 1]
        else:
            x_min, x_max = np.min(x_plot[:, 0]), np.max(x_plot[:, 0])
            y_min, y_max = np.min(x_plot[:, 1]), np.max(x_plot[:, 1])

        # Add margin to the plot range
        x_range = x_max - x_min
        y_range = y_max - y_min

        # Check if any points are outside the bounds
        if np.any(x_plot[:, 0] < bounds[0, 0]) or np.any(x_plot[:, 0] > bounds[0, 1]) or \
                np.any(x_plot[:, 1] < bounds[1, 0]) or np.any(x_plot[:, 1] > bounds[1, 1]):
            # If points are outside bounds, extend the plot range to include them
            x_min = min(x_min, np.min(x_plot[:, 0]))
            x_max = max(x_max, np.max(x_plot[:, 0]))
            y_min = min(y_min, np.min(x_plot[:, 1]))
            y_max = max(y_max, np.max(x_plot[:, 1]))

            # Recalculate ranges
            x_range = x_max - x_min
            y_range = y_max - y_min

        # Add margin around the data
        x_min -= self.margin * x_range
        x_max += self.margin * x_range
        y_min -= self.margin * y_range
        y_max += self.margin * y_range

        # Create a grid for the contour plot
        xi = np.linspace(x_min, x_max, self.n_contour_points)
        yi = np.linspace(y_min, y_max, self.n_contour_points)
        xi_grid, yi_grid = np.meshgrid(xi, yi)

        # Interpolate the objective function values on the grid
        try:
            zi = griddata((x_plot[:, 0], x_plot[:, 1]), f_array, (xi_grid, yi_grid), method='cubic')

            # Fall back to linear interpolation if cubic fails
            if np.isnan(zi).all():
                zi = griddata((x_plot[:, 0], x_plot[:, 1]), f_array, (xi_grid, yi_grid), method='linear')

            # If we still have NaNs, use nearest neighbor
            if np.isnan(zi).any():
                zi_nearest = griddata((x_plot[:, 0], x_plot[:, 1]), f_array, (xi_grid, yi_grid), method='nearest')
                zi = np.where(np.isnan(zi), zi_nearest, zi)
        except Exception as e:
            print(f"Error creating contour plot: {e}")
            self.ax2.text(0.5, 0.5, "Error creating contour plot",
                          ha='center', va='center', fontsize=14)
            return

        # Create a custom colormap that transitions smoothly
        cmap = plt.cm.viridis

        # Plot the contour with smooth gradients and few levels for clarity
        contour = self.ax2.contourf(xi_grid, yi_grid, zi, 15, cmap=cmap, alpha=0.8)

        # Add a properly positioned and formatted colorbar
        cbar = self.fig.colorbar(contour, ax=self.ax2)
        cbar.set_label('Objective Function Value', fontsize=10)
        # Ensure proper positioning and formatting of the colorbar
        cbar.ax.tick_params(labelsize=8)

        # Draw problem bounds
        self.ax2.axvline(bounds[0, 0], color='red', linestyle='--', alpha=0.5)
        self.ax2.axvline(bounds[0, 1], color='red', linestyle='--', alpha=0.5)
        self.ax2.axhline(bounds[1, 0], color='red', linestyle='--', alpha=0.5)
        self.ax2.axhline(bounds[1, 1], color='red', linestyle='--', alpha=0.5)

        # Plot all evaluated points
        self.ax2.scatter(x_plot[:, 0], x_plot[:, 1], c='#555555', s=50, alpha=0.7,
                         label='Evaluated Points', zorder=3, edgecolor='white', linewidth=0.5)

        # Highlight the best point
        if current_best_index < len(x_plot):
            self.ax2.scatter(x_plot[current_best_index, 0], x_plot[current_best_index, 1],
                             c='#00aa00', s=120, label='Best Point', zorder=5,
                             edgecolor='black', linewidth=1.5)

        # Highlight the latest point
        if len(x_plot) > 0:
            self.ax2.scatter(x_plot[-1, 0], x_plot[-1, 1], c='#ff3333', s=100,
                             label='Latest Point', zorder=4, edgecolor='black', linewidth=1.5)

        # Plot the principal components if available
        if component_matrix is not None and component_matrix.shape[0] >= 2:
            # Origin for the PC vectors (use the mean of the data)
            origin_x = np.mean(x_plot[:, 0])
            origin_y = np.mean(x_plot[:, 1])

            # Scale factor for the PC vectors
            scale = min(x_range, y_range) * 0.2

            # Plot the first PC
            pc1 = component_matrix[0, :2]  # First two dimensions of first PC
            self.ax2.arrow(origin_x, origin_y, pc1[0] * scale, pc1[1] * scale,
                           head_width=scale * 0.07, head_length=scale * 0.1, fc='blue', ec='blue',
                           linewidth=2, zorder=6, alpha=0.8, label='PC1')

            # Plot the second PC if available
            if component_matrix.shape[0] >= 2:
                pc2 = component_matrix[1, :2]  # First two dimensions of second PC
                self.ax2.arrow(origin_x, origin_y, pc2[0] * scale, pc2[1] * scale,
                               head_width=scale * 0.07, head_length=scale * 0.1, fc='cyan', ec='cyan',
                               linewidth=2, zorder=6, alpha=0.8, label='PC2')

        # Set labels and title with modern styling
        self.ax2.set_xlabel('Dimension 1', fontsize=12, fontweight='bold')
        self.ax2.set_ylabel('Dimension 2', fontsize=12, fontweight='bold')
        self.ax2.set_title('Objective Function Contour & Principal Components', fontsize=14, pad=10)

        # Add legend with modern styling
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax2.legend(by_label.values(), by_label.keys(),
                        loc='upper right', fontsize=10, framealpha=0.9,
                        edgecolor='lightgray')

        # Set axis limits
        self.ax2.set_xlim(x_min, x_max)
        self.ax2.set_ylim(y_min, y_max)

        # Modern aesthetics
        self.ax2.spines['top'].set_visible(False)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['left'].set_linewidth(0.5)
        self.ax2.spines['bottom'].set_linewidth(0.5)

    def save_animation(self, filename: str = "pca_bo_animation.gif") -> None:
        """Save all frames as an animation.

        Args:
            filename: Name of the output file
        """
        if len(self.frames) == 0:
            print("No frames to save.")
            return

        # Save animation using PIL
        output_path = f"{self.output_dir}/{filename}"

        # Ensure consistent dimensions for all frames
        width, height = self.frames[0].size
        frames_resized = [frame.resize((width, height)) for frame in self.frames]

        # Save the first frame
        first_frame = frames_resized[0]

        # Save all frames as a GIF
        first_frame.save(
            output_path,
            save_all=True,
            append_images=frames_resized[1:],
            optimize=False,
            duration=1000 // self.fps,  # Convert fps to duration in ms
            loop=0  # 0 means loop indefinitely
        )

        print(f"Animation saved to {output_path}")

        # Delete any individual frames if they exist and we don't want to save them
        if not self.save_frames:
            for f in os.listdir(self.output_dir):
                if f.startswith("pca_bo_iteration_") and f.endswith(".png"):
                    try:
                        os.remove(os.path.join(self.output_dir, f))
                    except Exception as e:
                        print(f"Warning: Could not remove file {f}: {e}")
