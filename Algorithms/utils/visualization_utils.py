import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import torch
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def save_animation(frames, filename, fps=1):
    """Save frames as an animated GIF"""
    if frames:
        # Create output directory if it doesn't exist
        os.makedirs('optimization_gifs', exist_ok=True)

        imageio.mimsave(
            os.path.join('optimization_gifs', filename),
            frames,
            fps=fps
        )

class Visualizer:
    """Class for visualizing optimization progress"""

    def __init__(self):
        """Initialize the visualizer"""
        self.frames = {
            'initial_design': [],
            'weights': [],
            'pca': [],
            'moving': [],
            'acquisition': [],
            'progress': [],
            'gaussian_process': []
        }

        # Create output directory for GIFs if it doesn't exist
        os.makedirs('optimization_gifs', exist_ok=True)

    def visualize_initial_design(self, X, y, dim, bounds=None, latest_idx=None):
        """Visualize the initial design points and save frame"""
        plt.figure(figsize=(10, 6))

        if dim <= 3:
            if dim == 2:
                plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis')
                plt.colorbar(label='Objective Value')
                plt.xlabel('X1')
                plt.ylabel('X2')
                # Highlight the latest point if provided
                if latest_idx is not None and latest_idx < len(X):
                    plt.scatter(X[latest_idx, 0], X[latest_idx, 1],
                               s=100, edgecolor='red', facecolor='none',
                               linewidth=2, label='Latest Point')

                # Draw bounds if provided
                if bounds is not None:
                    # Draw a rectangle for the bounds
                    from matplotlib.patches import Rectangle
                    rect = Rectangle(
                        (bounds[0, 0], bounds[1, 0]),  # (x, y) of bottom-left corner
                        bounds[0, 1] - bounds[0, 0],   # width
                        bounds[1, 1] - bounds[1, 0],   # height
                        linewidth=2,
                        edgecolor='blue',
                        facecolor='none',
                        linestyle='--',
                        alpha=0.7,
                        label='Bounds'
                    )
                    plt.gca().add_patch(rect)

            elif dim == 3:
                ax = plt.axes(projection='3d')
                scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                                     c=y.flatten(), cmap='viridis')
                plt.colorbar(scatter, label='Objective Value')
                ax.set_xlabel('X1')
                ax.set_ylabel('X2')
                ax.set_zlabel('X3')
                # Highlight the latest point if provided
                if latest_idx is not None and latest_idx < len(X):
                    ax.scatter(X[latest_idx, 0], X[latest_idx, 1], X[latest_idx, 2],
                              s=100, edgecolor='red', facecolor='none',
                              linewidth=2, label='Latest Point')

                # For 3D, we can show the bounds as lines at the edges of the bounding box
                if bounds is not None:
                    # Get the bounds for each dimension
                    x_min, x_max = bounds[0, 0], bounds[0, 1]
                    y_min, y_max = bounds[1, 0], bounds[1, 1]
                    z_min, z_max = bounds[2, 0], bounds[2, 1]

                    # Create the 8 corners of the bounding box
                    corners = np.array([
                        [x_min, y_min, z_min],
                        [x_max, y_min, z_min],
                        [x_max, y_max, z_min],
                        [x_min, y_max, z_min],
                        [x_min, y_min, z_max],
                        [x_max, y_min, z_max],
                        [x_max, y_max, z_max],
                        [x_min, y_max, z_max]
                    ])

                    # Define the 12 edges of the bounding box
                    edges = [
                        # Bottom face
                        [0, 1], [1, 2], [2, 3], [3, 0],
                        # Top face
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        # Connecting edges
                        [0, 4], [1, 5], [2, 6], [3, 7]
                    ]

                    # Plot each edge
                    for edge in edges:
                        ax.plot3D(
                            [corners[edge[0], 0], corners[edge[1], 0]],
                            [corners[edge[0], 1], corners[edge[1], 1]],
                            [corners[edge[0], 2], corners[edge[1], 2]],
                            'blue', linestyle='--', alpha=0.5
                        )

        else:
            pd.plotting.parallel_coordinates(
                pd.DataFrame(np.hstack([X, y]),
                             columns=[f'X{i + 1}' for i in range(dim)] + ['y']),
                'y', colormap=plt.cm.viridis)
            # For high dimensions, we can't easily highlight the latest point or show bounds in parallel coordinates

        plt.title('Initial Design Points')

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['initial_design'].append(image)
        plt.close()

    def visualize_weights(self, weights, iteration):
        """Visualize the weights assigned to points and save frame"""
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(weights)), weights, alpha=0.6)
        plt.plot(range(len(weights)), weights, 'r--', alpha=0.3)
        plt.xlabel('Point Index')
        plt.ylabel('Weight')
        plt.title(f'Weights Distribution (Iteration {iteration})')

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['weights'].append(image)
        plt.close()

    def visualize_pca_components(self, X_reduced, y, pca):
        """Visualize PCA components and explained variance"""
        # Plot explained variance ratio
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Components')

        # Plot first two components if available
        if X_reduced.shape[1] >= 2:
            plt.subplot(1, 2, 2)
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y.flatten(), cmap='viridis')
            plt.colorbar(label='Objective Value')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('First Two Principal Components')

        plt.tight_layout()
        plt.show()

    def visualize_acquisition(self, X_reduced, acq_values, iteration, test_points=None, bounds=None, latest_idx=None):
        """Visualize acquisition function and save frame"""
        if test_points is None:
            return

        plt.figure(figsize=(10, 6))
        if X_reduced.shape[1] == 1:
            plt.scatter(X_reduced.flatten(), np.zeros_like(X_reduced.flatten()),
                        c='red', label='Observed points', alpha=0.6)
            plt.plot(test_points.numpy(), acq_values, 'b-', label='Acquisition function')
            # Highlight the latest point if provided
            if latest_idx is not None and latest_idx < len(X_reduced):
                plt.scatter(X_reduced[latest_idx].item(), 0,
                           s=100, edgecolor='yellow', facecolor='none',
                           linewidth=2, label='Latest Point')

            # Show bounds if provided
            if bounds is not None:
                plt.axvline(x=bounds[0, 0], color='r', linestyle='--', alpha=0.5, label='Bounds')
                plt.axvline(x=bounds[0, 1], color='r', linestyle='--', alpha=0.5)

            plt.xlabel('Reduced Dimension')
            plt.ylabel('Acquisition Value')
        else:
            test_x = test_points.numpy()
            plt.tricontourf(test_x[:, 0], test_x[:, 1], acq_values.flatten())
            plt.colorbar(label='Acquisition Value')
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='red',
                        label='Observed points', alpha=0.6)
            # Highlight the latest point if provided
            if latest_idx is not None and latest_idx < len(X_reduced):
                plt.scatter(X_reduced[latest_idx, 0], X_reduced[latest_idx, 1],
                           s=100, edgecolor='yellow', facecolor='none',
                           linewidth=2, label='Latest Point')

            # Show bounds if provided
            if bounds is not None and bounds.shape[0] >= 2:
                # Draw a rectangle for the bounds
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (bounds[0, 0], bounds[1, 0]),  # (x, y) of bottom-left corner
                    bounds[0, 1] - bounds[0, 0],   # width
                    bounds[1, 1] - bounds[1, 0],   # height
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.7,
                    label='Bounds'
                )
                plt.gca().add_patch(rect)

            plt.xlabel('PC1')
            plt.ylabel('PC2')

        plt.title(f'Acquisition Function Values (Iteration {iteration})')
        plt.legend()

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['acquisition'].append(image)
        plt.close()

    def visualize_optimization_progress(self, f_evals, iteration, maximisation=False):
        """Visualize optimization progress and save frame"""
        plt.figure(figsize=(15, 5))

        # Plot best value so far
        plt.subplot(1, 2, 1)
        if maximisation:
            best_so_far = np.maximum.accumulate(f_evals)
        else:
            best_so_far = np.minimum.accumulate(f_evals)

        plt.plot(range(len(best_so_far)), best_so_far, 'b-', label='Best Value')
        # Highlight the latest point
        if len(f_evals) > 0:
            latest_idx = len(f_evals) - 1
            plt.scatter(latest_idx, best_so_far[latest_idx],
                       s=100, edgecolor='red', facecolor='none',
                       linewidth=2, label='Latest Point')
        plt.xlabel('Iteration')
        plt.ylabel('Best Objective Value')
        plt.title(f'Optimization Progress (Iteration {iteration})')
        plt.legend()

        # Plot all evaluated points
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(f_evals)), f_evals, alpha=0.6)
        # Highlight the latest point
        if len(f_evals) > 0:
            latest_idx = len(f_evals) - 1
            plt.scatter(latest_idx, f_evals[latest_idx],
                       s=100, edgecolor='red', facecolor='none',
                       linewidth=2, label='Latest Point')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Value')
        plt.title('All Evaluated Points')
        plt.legend()

        plt.tight_layout()

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['progress'].append(image)
        plt.close()

    def visualize_pca_step(self, X, f_evals, pca, scaler, weights, obj_function, iteration, bounds=None, latest_idx=None):
        """Visualize PCA step with contour plot"""
        if X.shape[1] != 2:  # This visualization only works for 2D problems
            return

        plt.figure(figsize=(10, 8))

        # Determine plot bounds based on data points
        margin = 0.1
        if bounds is not None:
            x_min = float(bounds[0, 0] - margin)
            x_max = float(bounds[0, 1] + margin)
            y_min = float(bounds[1, 0] - margin)
            y_max = float(bounds[1, 1] + margin)
        else:
            x_min = float(min(X[:,0].min(), 0) - margin)
            x_max = float(max(X[:,0].max(), 0) + margin)
            y_min = float(min(X[:,1].min(), 0) - margin)
            y_max = float(max(X[:,1].max(), 0) + margin)

        # Create data for contour plot
        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X_grid, Y_grid = np.meshgrid(x, y)
        XY = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
        Z = np.array([obj_function(point) for point in XY])
        Z = Z.reshape(X_grid.shape)

        # Plot contour
        plt.contour(X_grid, Y_grid, Z, levels=15, colors='black', alpha=0.3)
        contourf = plt.contourf(X_grid, Y_grid, Z, levels=100, cmap='viridis_r')
        plt.colorbar(contourf, label='Objective Value')

        # Plot the optimal point (origin) with a circle
        plt.scatter([0], [0], color='black', marker='x', s=80, zorder=5)
        circle = plt.Circle((0, 0), 0.2, fill=False, color='red', linestyle='-')
        plt.gca().add_patch(circle)

        # Plot data points
        plt.scatter(X[:,0], X[:,1], color='white', edgecolor='black',
                    s=80, zorder=4)

        # Highlight the latest point if provided
        if latest_idx is not None and latest_idx < len(X):
            plt.scatter(X[latest_idx, 0], X[latest_idx, 1],
                       s=120, edgecolor='red', facecolor='none',
                       linewidth=2, label='Latest Point', zorder=6)

        # Draw bounds if provided
        if bounds is not None:
            # Draw a rectangle for the bounds
            from matplotlib.patches import Rectangle
            rect = Rectangle(
                (bounds[0, 0], bounds[1, 0]),  # (x, y) of bottom-left corner
                bounds[0, 1] - bounds[0, 0],   # width
                bounds[1, 1] - bounds[1, 0],   # height
                linewidth=2,
                edgecolor='blue',
                facecolor='none',
                linestyle='--',
                alpha=0.7,
                label='Bounds',
                zorder=3
            )
            plt.gca().add_patch(rect)

        # Calculate and plot PC directions
        if len(X) > 1:  # Need at least 2 points for PCA
            X_scaled = scaler.fit_transform(X)
            X_weighted = X_scaled * np.sqrt(weights[:, np.newaxis])
            pca.fit(X_weighted)

            # Get the components and mean
            components = pca.components_
            mean = scaler.mean_

            # Calculate endpoints for PCA vectors (scale them for visibility)
            scale = 2
            pc1_endpoints = np.vstack([mean - scale * components[0], mean + scale * components[0]])

            # Plot PC1 (solid line)
            plt.plot(pc1_endpoints[:, 0], pc1_endpoints[:, 1], 'r-',
                     linewidth=2, label='PC1')

        plt.xlabel(r'$x_1$')
        plt.ylabel(r'$x_2$')
        plt.title(f'Objective Function Contour Plot (Iteration {iteration})')
        plt.legend()
        plt.grid(False)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.tight_layout()

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['pca'].append(image)
        plt.close()

    def visualize_weighted_transform(self, X, weights, pca):
        """
        Visualize how weighted PCA transforms points compared to the original points.

        Args:
            X (np.ndarray): Original points in the input space
            weights (np.ndarray): Weights for each point
            pca (PCA): Fitted PCA object
        """
        plt.figure(figsize=(10, 8))

        # Plot original points in white
        plt.scatter(X[:,0], X[:,1], color='white', edgecolor='black',
                    s=80, label='Original Points', zorder=4)

        # Transform points using weights
        X_centered = X - np.mean(X, axis=0)
        weighted_X = X_centered * np.sqrt(weights[:, np.newaxis])

        # Plot weighted points in blue
        plt.scatter(weighted_X[:,0], weighted_X[:,1], color='blue', alpha=0.6,
                    s=80, label='Weighted Points', zorder=5)

        # Draw lines connecting original and weighted points
        for i in range(len(X)):
            plt.plot([X[i,0], weighted_X[i,0]], [X[i,1], weighted_X[i,1]],
                     'gray', alpha=0.3, linestyle='--', zorder=3)


        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('PCA Transformation with Weights')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add weight values as text annotations
        for i, (x, y, w) in enumerate(zip(X[:,0], X[:,1], weights)):
            plt.annotate(f'w={w:.2f}', (x, y),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=8, alpha=0.7)

        plt.tight_layout()

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['moving'].append(image)
        plt.close()


    def visualize_gaussian_process(self, model, X_reduced, y, test_points, iteration, bounds=None, latest_idx=None):
        """Visualize the Gaussian Process model's mean and uncertainty.

        Args:
            model: The GP model
            X_reduced: The observed points in reduced space
            y: The observed function values
            test_points: Points to evaluate the GP model at
            iteration: Current iteration number
            bounds: Bounds in the reduced space (optional)
            latest_idx: Index of the latest point to highlight
        """
        if test_points is None or model is None:
            return

        plt.figure(figsize=(15, 6))

        # Get predictions from the model
        with torch.no_grad():
            posterior = model.posterior(test_points)
            mean = posterior.mean.numpy().flatten()
            std = posterior.variance.sqrt().numpy().flatten()

        # Plot mean and uncertainty
        if X_reduced.shape[1] == 1:
            # 1D plot
            test_x = test_points.numpy().flatten()

            # Sort points for smooth plotting
            sort_indices = np.argsort(test_x)
            test_x = test_x[sort_indices]
            mean = mean[sort_indices]
            std = std[sort_indices]

            # Plot mean and confidence intervals
            plt.subplot(1, 2, 1)
            plt.plot(test_x, mean, 'b-', label='GP Mean')
            plt.fill_between(test_x, mean - 2*std, mean + 2*std, alpha=0.2, color='b', label='95% Confidence')
            plt.scatter(X_reduced.flatten(), y.flatten(), c='red', label='Observations', alpha=0.6)
            # Highlight the latest point if provided
            if latest_idx is not None and latest_idx < len(X_reduced):
                plt.scatter(X_reduced[latest_idx].item(), y[latest_idx].item(),
                           s=100, edgecolor='yellow', facecolor='none',
                           linewidth=2, label='Latest Point')

            # Show bounds if provided
            if bounds is not None:
                plt.axvline(x=bounds[0, 0], color='r', linestyle='--', alpha=0.5, label='Bounds')
                plt.axvline(x=bounds[0, 1], color='r', linestyle='--', alpha=0.5)

            plt.xlabel('Reduced Dimension')
            plt.ylabel('Objective Value')
            plt.title(f'Gaussian Process Model (Iteration {iteration})')
            plt.legend()

            # Plot standard deviation
            plt.subplot(1, 2, 2)
            plt.plot(test_x, std, 'r-', label='GP Uncertainty')
            plt.scatter(X_reduced.flatten(), np.zeros_like(X_reduced.flatten()), c='red', label='Observations', alpha=0.6)
            # Highlight the latest point if provided
            if latest_idx is not None and latest_idx < len(X_reduced):
                plt.scatter(X_reduced[latest_idx].item(), 0,
                           s=100, edgecolor='yellow', facecolor='none',
                           linewidth=2, label='Latest Point')

            # Show bounds if provided
            if bounds is not None:
                plt.axvline(x=bounds[0, 0], color='r', linestyle='--', alpha=0.5, label='Bounds')
                plt.axvline(x=bounds[0, 1], color='r', linestyle='--', alpha=0.5)

            plt.xlabel('Reduced Dimension')
            plt.ylabel('Standard Deviation')
            plt.title('Gaussian Process Uncertainty')
            plt.legend()

        else:
            # 2D plot - contour plots for mean and uncertainty
            test_x = test_points.numpy()

            # Plot mean
            plt.subplot(1, 2, 1)
            plt.tricontourf(test_x[:, 0], test_x[:, 1], mean, levels=50, cmap='viridis')
            plt.colorbar(label='Predicted Mean')
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y.flatten(), cmap='viridis',
                        edgecolors='white', label='Observations', alpha=0.8)
            # Highlight the latest point if provided
            if latest_idx is not None and latest_idx < len(X_reduced):
                plt.scatter(X_reduced[latest_idx, 0], X_reduced[latest_idx, 1],
                           s=100, edgecolor='yellow', facecolor='none',
                           linewidth=2, label='Latest Point')

            # Show bounds if provided
            if bounds is not None and bounds.shape[0] >= 2:
                # Draw a rectangle for the bounds
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (bounds[0, 0], bounds[1, 0]),  # (x, y) of bottom-left corner
                    bounds[0, 1] - bounds[0, 0],   # width
                    bounds[1, 1] - bounds[1, 0],   # height
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.7,
                    label='Bounds'
                )
                plt.gca().add_patch(rect)

            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title(f'GP Predicted Mean (Iteration {iteration})')
            plt.legend()

            # Plot uncertainty
            plt.subplot(1, 2, 2)
            plt.tricontourf(test_x[:, 0], test_x[:, 1], std, levels=50, cmap='plasma')
            plt.colorbar(label='Uncertainty (Std)')
            plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='white',
                        edgecolors='black', label='Observations', alpha=0.8)
            # Highlight the latest point if provided
            if latest_idx is not None and latest_idx < len(X_reduced):
                plt.scatter(X_reduced[latest_idx, 0], X_reduced[latest_idx, 1],
                           s=100, edgecolor='yellow', facecolor='none',
                           linewidth=2, label='Latest Point')

            # Show bounds if provided
            if bounds is not None and bounds.shape[0] >= 2:
                # Draw a rectangle for the bounds
                from matplotlib.patches import Rectangle
                rect = Rectangle(
                    (bounds[0, 0], bounds[1, 0]),  # (x, y) of bottom-left corner
                    bounds[0, 1] - bounds[0, 0],   # width
                    bounds[1, 1] - bounds[1, 0],   # height
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none',
                    linestyle='--',
                    alpha=0.7,
                    label='Bounds'
                )
                plt.gca().add_patch(rect)

            plt.xlabel('PC1')
            plt.ylabel('PC2')
            plt.title('GP Uncertainty')
            plt.legend()

        plt.tight_layout()

        # Capture the frame
        fig = plt.gcf()
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        self.frames['gaussian_process'].append(image)
        plt.close()

    def combine_gifs(self):
        """Combine all visualization frames into a single GIF"""
        combined_frames = []
        max_frames = max(len(frames) for frames in self.frames.values() if frames)

        if max_frames == 0:
            return None

        for i in range(max_frames):
            # Create a figure with subplots for each visualization
            fig, axes = plt.subplots(2, 3, figsize=(30, 20))
            fig.suptitle(f'Optimization Progress - Frame {i + 1}', fontsize=16)

            # List of frame types and their corresponding subplot positions
            frame_layout = [
                ('initial_design', (0, 0), 'Initial Design'),
                ('weights', (0, 1), 'Weights'),
                ('acquisition', (0, 2), 'Acquisition'),
                ('progress', (1, 0), 'Optimization Progress'),
                ('pca', (1, 1), 'PCA Visualization'),
                ('gaussian_process', (1, 2), 'Gaussian Process')
            ]

            # Plot each frame type
            for frame_type, (row, col), title in frame_layout:
                if self.frames[frame_type] and i < len(self.frames[frame_type]):
                    frame = self.frames[frame_type][i]
                    axes[row, col].imshow(frame)
                axes[row, col].set_title(title)
                axes[row, col].axis('off')

            # Capture the combined frame
            fig.canvas.draw()
            combined_frame = np.array(fig.canvas.renderer.buffer_rgba())
            combined_frames.append(combined_frame)
            plt.close()

        return combined_frames

    def save_all_animations(self):
        """Save all animations"""
        save_animation(self.frames['initial_design'], 'initial_design.gif')
        save_animation(self.frames['weights'], 'weights.gif')
        save_animation(self.frames['acquisition'], 'acquisition.gif')
        save_animation(self.frames['progress'], 'optimization_progress.gif')
        save_animation(self.frames['pca'], 'pca_visualization.gif')
        save_animation(self.frames['moving'], 'moving_visualization.gif')
        save_animation(self.frames['gaussian_process'], 'gaussian_process.gif')

        # Create and save combined animation
        #combined_frames = self.combine_gifs()
        #if combined_frames:
        #    save_animation(combined_frames, 'combined_visualization.gif')
