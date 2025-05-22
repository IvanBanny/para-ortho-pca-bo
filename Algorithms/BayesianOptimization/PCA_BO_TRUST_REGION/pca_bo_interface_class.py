import dataclasses
import pickle
from abc import ABC
from typing import Union, Callable, Optional, Dict, Any, List



import numpy as np
import torch
from ioh.iohcpp.problem import RealSingleObjective, BBOB

from Algorithms.AbstractAlgorithm import AbstractAlgorithm
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import CleanPCABO, DOE, AcquisitionFunctionEnum, \
    PCBANumComponents, MyPCA

examplePath = r"example.pkl"

class CleanPCABOInterface(AbstractAlgorithm):
    def __init__(
            self,
            budget: int,
            n_DoE: int = 0,
            n_components: Optional[int] = None,
            var_threshold: float = 0.95,
            acquisition_function: str = "expected_improvement",
            random_seed: int = 43,
            torch_config: Optional[Dict[str, Any]] = None,
            visualize: bool = False,
            vis_output_dir: str = "./visualizations",
            save_logs: bool = False,
            log_dir: str = "./logs",
            **kwargs
    ):
        super().__init__(**kwargs)

        self.budget = budget
        self.n_DoE = n_DoE
        self.n_components = n_components
        self.var_threshold = var_threshold
        self.acquisition_function = AcquisitionFunctionEnum.from_name(acquisition_function)
        self.random_seed = random_seed
        self.torch_config = torch_config
        self.visualize = visualize
        self.vis_output_dir = vis_output_dir
        self.save_logs = save_logs

        self.pca_num_components = PCBANumComponents(
            num_components=self.n_components,
            var_threshold=self.var_threshold
        )

    def __call__(
            self,
            problem: Union[RealSingleObjective, BBOB, Callable],
            dim: Optional[int] = -1,
            bounds: Optional[np.ndarray] = None,
            **kwargs
    ):
        super().__call__(problem, dim, bounds, **kwargs)

        assert isinstance(problem, Callable)

        clean_pcabo = CleanPCABOWithLogging(
            problem=problem,
            budget=self.budget,
            bounds=self.bounds,
            doe=DOE(n=self.n_DoE),
            maximization=self.maximization,
            acquisition_function_class=self.acquisition_function.class_type,
            pca_num_components=self.pca_num_components,
        )

    def __str__(self):
        return "CleanPCABO_Interface"

    def reset(self):
        super().reset()


@dataclasses.dataclass
class IterationData:
    pca: Optional[MyPCA] = None
    points_x: Optional[np.ndarray] = None
    points_y: Optional[np.ndarray] = None
    bounds: Optional[np.ndarray] = None

    gpr_y: Optional[np.ndarray] = None
    gpr_std: Optional[np.ndarray] = None
    gpr_x: Optional[np.ndarray] = None

    acqf_y: Optional[np.ndarray] = None
    acqf_x: Optional[np.ndarray] = None


class CleanPCABOWithLogging(CleanPCABO):
    # overwrite every function which result we want to save

    iterations: List[IterationData]

    plot_X_grid: Optional[np.ndarray] = None
    plot_Y_grid: Optional[np.ndarray] = None
    plot_Z: Optional[np.ndarray] = None

    global_optimum_x: Optional[np.ndarray] = None

    def optimize(self):
        self.iterations = []
        # in this function y and z do not referr to the usual meanings of y and z in this class
        if self.bounds.shape[0] == 2:
            x_min = self.bounds[0, 0]
            x_max = self.bounds[0, 1]
            y_min = self.bounds[1, 0]
            y_max = self.bounds[1, 1]

            padding_x = 0.1 * (x_max - x_min)
            padding_y = 0.1 * (y_max - y_min)

            x = np.linspace(x_min - padding_x, x_max + padding_x, 100)
            y = np.linspace(y_min - padding_y, y_max + padding_y, 100)
            X_grid, Y_grid = np.meshgrid(x, y)
            XY = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
            Z = np.array([self.problem(point) for point in XY])
            self.plot_Z = Z.reshape(X_grid.shape)
            self.plot_X_grid = X_grid
            self.plot_Y_grid = Y_grid

            self.global_optimum_x = self.problem.optimum.x

        super().optimize()

    def create_gpr_model(self, points_z, z_bounds):
        gpr = super().create_gpr_model(points_z, z_bounds)

        assert z_bounds.shape == (2, 1), z_bounds.shape

        iteration = self.iterations[-1]
        iteration.gpr_x = np.linspace(z_bounds[0, 0], z_bounds[1, 0], 100).reshape(-1, 1)  # 100 evenly spaced points in the 1 dimensional reduced space

        # Predict the mean and variance of the gpr model at the 100 evenly spaced points
        with torch.no_grad():
            posterior = gpr.posterior(torch.from_numpy(iteration.gpr_x))
            mean = posterior.mean.numpy().flatten()
            # Extract the standard deviation (square root of variance)
            std = posterior.variance.sqrt().numpy().flatten()

        # Store mean and std for visualization
        iteration.gpr_y = mean
        iteration.gpr_std = std  # Add this to store standard deviation

        return gpr

    def create_penalized_acquisition(self, acquisition_function, gpr_model, pca):
        penalized_acqf = super().create_penalized_acquisition(acquisition_function, gpr_model, pca)

        iteration : IterationData = self.iterations[-1]

        # Create a grid of points in the reduced space for visualization
        if iteration.gpr_x is not None:
            # We already have a grid of points from the GPR visualization
            # Use the same grid for the acquisition function
            z_points = iteration.gpr_x

            # Get the acquisition function values at these points
            with torch.no_grad():
                # Convert to tensor
                z_tensor = torch.from_numpy(z_points.reshape(-1, 1, 1))
                # Get acqf values
                acqf_values = penalized_acqf(z_tensor).detach().numpy().flatten()

            # Store for visualization
            iteration.acqf_x = z_points
            iteration.acqf_y = acqf_values

        return penalized_acqf

    def iteration(self):
        self.iterations.append(IterationData())
        super().iteration()

        self.iterations[-1].points_x = self.X
        self.iterations[-1].points_y = self.fX
        self.iterations[-1].bounds = self.bounds

        with open(examplePath, 'wb') as f:
            pickle.dump(self, f)

        print(f'saved after iteration', len(self.iterations))

    def fit_pca(self):
        pca = super().fit_pca()

        self.iterations[-1].pca = pca

        return pca

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["problem"]
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)


def plot2d(pcabo: CleanPCABOWithLogging):
    """
    Generate a GIF visualization of the optimization process for a 2D problem.
    Creates three separate plots instead of subplots.

    Args:
        pcabo: CleanPCABOWithLogging object with optimization history
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.animation as animation
    import numpy as np
    from matplotlib import cm
    from matplotlib import gridspec
    import os

    # Check if the optimization was performed on a 2D problem
    if pcabo.bounds.shape[0] != 2:
        print("Visualization only supported for 2D problems")
        return

    # Create output directory if it doesn't exist
    os.makedirs("visualization_output", exist_ok=True)

    # Global x and y bounds
    x_min, x_max = pcabo.bounds[0, 0], pcabo.bounds[0, 1]
    y_min, y_max = pcabo.bounds[1, 0], pcabo.bounds[1, 1]

    # Create mesh grid for plotting
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X_grid, Y_grid = np.meshgrid(x, y)

    def create_objective_plot(i):
        """Create the objective function and search points plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        # Set plot limits with padding
        padding_x = 0.1 * (x_max - x_min)
        padding_y = 0.1 * (y_max - y_min)
        ax.set_xlim(x_min - padding_x, x_max + padding_x)
        ax.set_ylim(y_min - padding_y, y_max + padding_y)

        ax.set_title(f"Objective Function & Search Points\nIteration {i + 1}/{len(pcabo.iterations)}", fontsize=14)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_aspect('equal', adjustable='box')

        # Plot objective function contour
        if pcabo.plot_Z is not None:
            contour = ax.contourf(pcabo.plot_X_grid, pcabo.plot_Y_grid, pcabo.plot_Z,
                                  levels=50, cmap='plasma', alpha=0.7)
            cbar = plt.colorbar(contour, ax=ax, label='Objective function value')
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Objective function value', fontsize=14)

        # Plot search points until current iteration
        if iteration_data.points_x is not None:
            points_x = iteration_data.points_x[:, 0]
            points_y = iteration_data.points_x[:, 1]

            # Highlight the global optimum
            ax.scatter(pcabo.global_optimum_x[0], pcabo.global_optimum_x[1], color='gray', marker='*', s=400, label='Global Optimum')

            # Plot all points in black first
            ax.scatter(points_x, points_y, color='black', marker='o', s=50, alpha=0.5, label='Search points')

            # Highlight the most recent point
            ax.scatter(points_x[-1], points_y[-1], color='red', marker='*', s=400, label='Latest point')

            # Add best point found so far
            if len(points_x) > 0:
                best_idx = np.argmin(iteration_data.points_y) if not pcabo.maximization else np.argmax(
                    iteration_data.points_y)
                ax.scatter(points_x[best_idx], points_y[best_idx], color='orange', marker='X', s=150,
                           label='Best point')

            # Plot trust region bounds if available
            if iteration_data.bounds is not None and i > 0:
                local_x_min, local_x_max = iteration_data.bounds[0, 0], iteration_data.bounds[0, 1]
                local_y_min, local_y_max = iteration_data.bounds[1, 0], iteration_data.bounds[1, 1]

                rect = patches.Rectangle((local_x_min, local_y_min),
                                         local_x_max - local_x_min,
                                         local_y_max - local_y_min,
                                         linewidth=2, edgecolor='r', facecolor='none',
                                         label='Bounds')
                ax.add_patch(rect)

            # Add PC1 component visualization if PCA is available
            if iteration_data.pca is not None:
                # Calculate endpoints of PC1 in original space
                z_bounds = pcabo.calculate_reduced_space_bounds(iteration_data.pca)
                pc1_pos = iteration_data.pca.transform_to_original(z_bounds[0].reshape(1, 1))
                pc1_neg = iteration_data.pca.transform_to_original(z_bounds[1].reshape(1, 1))

                # Plot PC1 as a line
                ax.plot([pc1_neg[0, 0], pc1_pos[0, 0]], [pc1_neg[0, 1], pc1_pos[0, 1]],
                        'orange', linewidth=3, label='PC1 Component')

                # Add an arrow to indicate direction
                mid_point = (pc1_neg[0] + pc1_pos[0]) / 2
                direction = (pc1_pos[0] - pc1_neg[0]) / np.linalg.norm(pc1_pos[0] - pc1_neg[0])
                arrow_length = np.linalg.norm(pc1_pos[0] - pc1_neg[0]) * 0.1
                ax.arrow(mid_point[0], mid_point[1],
                         direction[0] * arrow_length, direction[1] * arrow_length,
                         head_width=0.1, head_length=0.2, fc='g', ec='g')

            # Add legend below the plot
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=14)

        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        # Adjust layout to accommodate legend below the plot
        plt.subplots_adjust(bottom=0.2)
        # Adjust layout to accommodate legend outside the plot
        plt.subplots_adjust(right=0.75)
        # Adjust layout to accommodate legend outside the plot
        plt.subplots_adjust(right=0.75)
        return fig

    def create_gp_plot(i):
        """Create the GP prediction plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        ax.set_title(f"GP Prediction in Reduced Space\nIteration {i + 1}/{len(pcabo.iterations)}", fontsize=14)
        ax.set_xlabel("Reduced Dimension (PC1)", fontsize=12)
        ax.set_ylabel("Predicted Value", fontsize=12)

        # Plot GP prediction if available
        if iteration_data.gpr_x is not None and iteration_data.gpr_y is not None and hasattr(iteration_data, 'gpr_std'):
            gpr_x_flat = iteration_data.gpr_x.flatten()

            # Plot mean prediction
            ax.plot(gpr_x_flat, iteration_data.gpr_y, 'b-', linewidth=2, label='Mean prediction')

            # Plot confidence intervals
            upper_bound = iteration_data.gpr_y + 2 * iteration_data.gpr_std
            lower_bound = iteration_data.gpr_y - 2 * iteration_data.gpr_std
            ax.fill_between(gpr_x_flat, lower_bound, upper_bound, color='b', alpha=0.2, label='95% confidence interval')

            # Plot the observed points in reduced space
            if iteration_data.pca is not None:
                points_z = iteration_data.pca.transform_to_reduced(iteration_data.points_x)
                points_z_1d = points_z[:, 0]
                ax.scatter(points_z_1d, iteration_data.points_y, color='black', marker='o', s=40,
                           label='Observed points')

                # Highlight the most recent point
                ax.scatter(points_z_1d[-1], iteration_data.points_y[-1], color='red',
                           marker='*', s=200, label='Latest point')

                # Add best point found so far
                best_idx = np.argmin(iteration_data.points_y) if not pcabo.maximization else np.argmax(
                    iteration_data.points_y)
                ax.scatter(points_z_1d[best_idx], iteration_data.points_y[best_idx],
                           color='orange', marker='X', s=150, label='Best point')

            # Add legend below the plot
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=14)

        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        return fig

    def create_acquisition_plot(i):
        """Create the acquisition function plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        ax.set_title(f"Acquisition Function in Reduced Space\nIteration {i + 1}/{len(pcabo.iterations)}", fontsize=14)
        ax.set_xlabel("Reduced Dimension (PC1)", fontsize=12)
        ax.set_ylabel("Acquisition Value", fontsize=12)

        # Plot acquisition function if available
        if iteration_data.acqf_x is not None and iteration_data.acqf_y is not None:
            acqf_x_flat = iteration_data.acqf_x.flatten()
            ax.plot(acqf_x_flat, iteration_data.acqf_y, 'g-', linewidth=2, label='Acquisition function')

            # Mark the next selected point if possible
            if i < len(pcabo.iterations) - 1 and iteration_data.pca is not None:
                next_iteration = pcabo.iterations[i + 1]
                latest_point_x = next_iteration.points_x[-1:]
                latest_point_z = iteration_data.pca.transform_to_reduced(latest_point_x)
                if latest_point_z.shape[1] > 0:
                    ax.axvline(x=latest_point_z[0, 0], color='red', linestyle='--',
                               linewidth=2, label='Next selected point')

            # Add legend below the plot
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=14)

        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        return fig

    # Create final static plots
    final_iteration = len(pcabo.iterations) - 1

    # Create and save objective function plot
    fig1 = create_objective_plot(final_iteration)
    fig1.savefig('visualization_output/objective_function.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # Create and save GP prediction plot
    fig2 = create_gp_plot(final_iteration)
    fig2.savefig('visualization_output/gp_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig2)

    # Create and save acquisition function plot
    fig3 = create_acquisition_plot(final_iteration)
    fig3.savefig('visualization_output/acquisition_function.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig3)

    # Create animations for each plot type
    frames = min(len(pcabo.iterations), 30)
    step = max(1, len(pcabo.iterations) // frames)
    frame_indices = list(range(0, len(pcabo.iterations), step))
    if len(pcabo.iterations) - 1 not in frame_indices:
        frame_indices.append(len(pcabo.iterations) - 1)

    # Create objective function animation
    fig_obj = plt.figure(figsize=(10, 8))

    def animate_objective(frame_idx):
        plt.clf()
        temp_fig = create_objective_plot(frame_indices[frame_idx])
        # Copy the plot content to the animation figure
        ax_temp = temp_fig.gca()
        ax_new = plt.gca()

        # Copy all the plot elements
        for child in ax_temp.get_children():
            if hasattr(child, 'get_data'):
                try:
                    ax_new.add_artist(child)
                except:
                    pass

        ax_new.set_xlim(ax_temp.get_xlim())
        ax_new.set_ylim(ax_temp.get_ylim())
        ax_new.set_title(ax_temp.get_title())
        ax_new.set_xlabel(ax_temp.get_xlabel())
        ax_new.set_ylabel(ax_temp.get_ylabel())

        plt.close(temp_fig)
        return ax_new.get_children()

    # Create separate animations (simplified approach)
    print("Creating individual frame images...")
    os.makedirs('visualization_output/frames_objective', exist_ok=True)
    os.makedirs('visualization_output/frames_gp', exist_ok=True)
    os.makedirs('visualization_output/frames_acquisition', exist_ok=True)

    for idx, frame_idx in enumerate(frame_indices):
        # Objective function frames
        fig = create_objective_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_objective/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # GP prediction frames
        fig = create_gp_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_gp/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Acquisition function frames
        fig = create_acquisition_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_acquisition/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f"Static plots saved:")
    print(f"  - visualization_output/objective_function.png")
    print(f"  - visualization_output/gp_prediction.png")
    print(f"  - visualization_output/acquisition_function.png")
    print(f"Frame sequences saved to:")
    print(f"  - visualization_output/frames_objective/")
    print(f"  - visualization_output/frames_gp/")
    print(f"  - visualization_output/frames_acquisition/")

    # Print optimization results
    best_idx = np.argmin(pcabo.fX) if not pcabo.maximization else np.argmax(pcabo.fX)
    print(f"\nOptimization Results:")
    print(f"Best point found: {pcabo.X[best_idx]}")
    print(f"Best value: {pcabo.fX[best_idx]}")
    print(f"Total evaluations: {len(pcabo.X)}")

if __name__ == "__main__":
    with open(examplePath, 'rb') as f:
        loaded_data : CleanPCABOWithLogging = pickle.load(f)
        print(len(loaded_data.iterations))
        print(loaded_data.X)
        plot2d(loaded_data)