from typing import Tuple

import numpy as np

from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.lpca_bo_interface_class import CleanLPCABOWithLogging
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo import MyPCA, calculate_reduced_space_bounds, \
    calculate_weights
from Algorithms.BayesianOptimization.PCA_BO_TRUST_REGION.pca_bo_interface_class import CleanPCABOWithLogging


def plot2d(pcabo: CleanPCABOWithLogging | CleanLPCABOWithLogging):
    """
    Generate a GIF visualization of the optimization process for a 2D problem.
    Creates four separate plots instead of subplots.

    Args:
        pcabo: CleanPCABOWithLogging object with optimization history
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import os
    from PIL import Image
    import glob

    # Check if the optimization was performed on a 2D problem
    if pcabo.bounds.shape[0] != 2:
        print("Visualization only supported for 2D problems")
        return

    # Create output directory if it doesn't exist
    os.makedirs("visualization_output", exist_ok=True)

    # Global x and y bounds
    x_min, x_max = pcabo.bounds[0, 0], pcabo.bounds[0, 1]
    y_min, y_max = pcabo.bounds[1, 0], pcabo.bounds[1, 1]

    def create_objective_plot(i):
        """Create the objective function and search points plot with zoomed-in version"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        # Left plot: Full view
        # Set plot limits with padding
        padding_x = 0.1 * (x_max - x_min)
        padding_y = 0.1 * (y_max - y_min)
        ax1.set_xlim(x_min - padding_x, x_max + padding_x)
        ax1.set_ylim(y_min - padding_y, y_max + padding_y)

        ax1.set_title(f"Objective Function & Search Points (Full View)\nIteration {i + 1}/{len(pcabo.iterations)}",
                      fontsize=14)
        ax1.set_xlabel("x", fontsize=12)
        ax1.set_ylabel("y", fontsize=12)
        ax1.set_aspect('equal', adjustable='box')

        # Plot objective function contour on left plot
        if pcabo.plot_Z is not None:
            contour1 = ax1.contourf(pcabo.plot_X_grid, pcabo.plot_Y_grid, pcabo.plot_Z,
                                    levels=50, cmap='plasma', alpha=0.7)
            cbar1 = plt.colorbar(contour1, ax=ax1, label='Objective function value')
            cbar1.ax.tick_params(labelsize=12)
            cbar1.set_label('Objective function value', fontsize=12)

        # Right plot: Zoomed view
        ax2.set_title(f"Objective Function & Search Points (Zoomed View)\nIteration {i + 1}/{len(pcabo.iterations)}",
                      fontsize=14)
        ax2.set_xlabel("x", fontsize=12)
        ax2.set_ylabel("y", fontsize=12)
        ax2.set_aspect('equal', adjustable='box')

        # Set zoomed plot limits based on current iteration bounds
        if iteration_data.bounds is not None:
            local_x_min, local_x_max = iteration_data.bounds[0, 0], iteration_data.bounds[0, 1]
            local_y_min, local_y_max = iteration_data.bounds[1, 0], iteration_data.bounds[1, 1]

            # Add small padding for zoomed view
            zoom_padding_x = 0.05 * (local_x_max - local_x_min)
            zoom_padding_y = 0.05 * (local_y_max - local_y_min)
            ax2.set_xlim(local_x_min - zoom_padding_x, local_x_max + zoom_padding_x)
            ax2.set_ylim(local_y_min - zoom_padding_y, local_y_max + zoom_padding_y)
        else:
            # Fallback to full bounds if no local bounds available
            ax2.set_xlim(x_min - padding_x, x_max + padding_x)
            ax2.set_ylim(y_min - padding_y, y_max + padding_y)

        # Plot objective function contour on right plot (zoomed)
        if pcabo.plot_Z is not None:
            contour2 = ax2.contourf(pcabo.plot_X_grid, pcabo.plot_Y_grid, pcabo.plot_Z,
                                    levels=50, cmap='plasma', alpha=0.7)
            cbar2 = plt.colorbar(contour2, ax=ax2, label='Objective function value')
            cbar2.ax.tick_params(labelsize=12)
            cbar2.set_label('Objective function value', fontsize=12)

        # Plot search points on both plots
        for ax in [ax1, ax2]:
            if iteration_data.points_x is not None:
                points_x = iteration_data.points_x[:, 0]
                points_y = iteration_data.points_x[:, 1]

                # Highlight the global optimum
                ax.scatter(pcabo.global_optimum_x[0], pcabo.global_optimum_x[1], color='gray', marker='*', s=400,
                           label='Global Optimum')

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
                if iteration_data.bounds is not None:
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
                    z_bounds = calculate_reduced_space_bounds(iteration_data.bounds, iteration_data.pca)
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

            # Set tick label size
            ax.tick_params(axis='both', which='major', labelsize=12)

        # Add legend below the plots (shared)
        if iteration_data.points_x is not None:
            handles, labels = ax1.get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3, fontsize=12)

        plt.tight_layout()
        # Adjust layout to accommodate legend below the plot
        plt.subplots_adjust(bottom=0.15)
        return fig

    def create_weighted_pca_plot(i):
        """Create the weighted PCA transformation visualization plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        ax.set_title(f"PCA Transformation with Weights\nIteration {i + 1}/{len(pcabo.iterations)}", fontsize=14)
        ax.set_xlabel("X1", fontsize=12)
        ax.set_ylabel("X2", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Plot trust region bounds if available
        if iteration_data.bounds is not None:
            local_x_min, local_x_max = iteration_data.bounds[0, 0], iteration_data.bounds[0, 1]
            local_y_min, local_y_max = iteration_data.bounds[1, 0], iteration_data.bounds[1, 1]

            rect = patches.Rectangle((local_x_min, local_y_min),
                                     local_x_max - local_x_min,
                                     local_y_max - local_y_min,
                                     linewidth=2, edgecolor='r', facecolor='none',
                                     label='Bounds')
            ax.add_patch(rect)

        # Plot objective function contour
        if pcabo.plot_Z is not None:
            contour = ax.contourf(pcabo.plot_X_grid, pcabo.plot_Y_grid, pcabo.plot_Z,
                                  levels=50, cmap='plasma', alpha=0.1)
            cbar = plt.colorbar(contour, ax=ax, label='Objective function value')
            cbar.ax.tick_params(labelsize=14)
            cbar.set_label('Objective function value', fontsize=14)

        # Check if we have the necessary data
        if (iteration_data.points_x is not None and
                iteration_data.pca is not None):
            X = iteration_data.points_x[:-1]  # do not include the last point, as it is the new point
            mask = iteration_data.pca_points_mask if iteration_data.pca_points_mask is not None else np.full(len(X),
                                                                                                             True)
            X = X[mask]
            weights = calculate_weights(False, iteration_data.points_y[:-1][mask])

            # Plot original points in white with black edges
            ax.scatter(X[:, 0], X[:, 1], color='white', edgecolor='black',
                       s=80, label='Original Points', zorder=4)

            # Transform points using weights
            X_centered = X - np.mean(X, axis=0)
            weighted_X = X_centered * np.sqrt(weights[:, np.newaxis])

            # Shift weighted points to be around the mean for visualization
            weighted_X_shifted = weighted_X + np.mean(X, axis=0)

            # Plot weighted points in blue
            ax.scatter(weighted_X_shifted[:, 0], weighted_X_shifted[:, 1], color='blue', alpha=0.6,
                       s=80, label='Weighted Points', zorder=5)

            # Draw lines connecting original and weighted points
            for j in range(len(X)):
                ax.plot([X[j, 0], weighted_X_shifted[j, 0]],
                        [X[j, 1], weighted_X_shifted[j, 1]],
                        'gray', alpha=0.3, linestyle='--', zorder=3)

            # Add weight values as text annotations
            for j, (x, y, w) in enumerate(zip(X[:, 0], X[:, 1], weights)):
                ax.annotate(f'w={w:.2f}', (x, y),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, alpha=0.7)

            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3, fontsize=12)

        else:
            # If we don't have weighted PCA data, show a message
            ax.text(0.5, 0.5, 'Weighted PCA data not available\nfor this iteration',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)
        return fig

    def create_gp_plot(i):
        """Create the GP prediction plot"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        ax.set_title(f"GP Prediction in Reduced Space\nIteration {i + 1}/{len(pcabo.iterations)}", fontsize=14)
        ax.set_xlabel("Reduced Dimension (PC1)", fontsize=12)
        ax.set_ylabel("Predicted Value", fontsize=12)

        i1, i2 = calculate_pc1_bounds_intersection(iteration_data.bounds, iteration_data.pca)
        ax.vlines([i1, i2], 0, 1, transform=ax.get_xaxis_transform(), label="Intersection PC1 and Bounds")

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

        i1, i2 = calculate_pc1_bounds_intersection(iteration_data.bounds, iteration_data.pca)
        ax.vlines([i1, i2], 0, 1, transform=ax.get_xaxis_transform(), label="Intersection PC1 and Bounds")

        # Plot acquisition function if available
        if iteration_data.acqf_x is not None and iteration_data.acqf_y is not None:
            acqf_x_flat = iteration_data.acqf_x.flatten()
            ax.plot(acqf_x_flat, iteration_data.pacqf_y, 'r-', linewidth=2, label='Penalized Acquisition function')

            ax.plot(acqf_x_flat, iteration_data.acqf_y, 'g-', linewidth=2, label='Acquisition function')

            latest_point_x = iteration_data.points_x[-1:]
            latest_point_z = iteration_data.pca.transform_to_reduced(latest_point_x)
            if latest_point_z.shape[1] > 0:
                ax.axvline(x=latest_point_z[0, 0], color='red', linestyle='--',
                           linewidth=2, label='selected point')

            # Add legend below the plot
            ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2, fontsize=14)

        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.tight_layout()
        return fig

    def create_candidate_table(i):
        """Create a CSV table with candidate points information"""
        import pandas as pd

        iteration_data = pcabo.iterations[i] if i < len(pcabo.iterations) else pcabo.iterations[-1]

        if (iteration_data.acqf_x is None or iteration_data.acqf_y is None or
                iteration_data.pca is None):
            return

        # Get candidate points in reduced space
        candidates_reduced = iteration_data.acqf_x.flatten().reshape(-1, 1)

        # Transform to original space
        candidates_original = iteration_data.pca.transform_to_original(candidates_reduced)

        # Get acquisition function values and penalized values
        acqf_values = iteration_data.acqf_y
        pacqf_values = iteration_data.pacqf_y

        # Calculate penalty as the difference between penalized and original acquisition function
        penalty_values = pacqf_values - acqf_values

        # Determine if penalty applies (when there's a difference)
        penalized = np.abs(penalty_values) > 1e-6
        penalty_status = ['penalty' if p else 'no penalty' for p in penalized]

        # Create DataFrame
        table_data = {
            'Point candidates in reduced space': [f'[{x[0]:.7f}]' for x in candidates_reduced],
            'Point candidates in original space': [f'[{x[0]:.12f}, {x[1]:.12f}]' for x in candidates_original],
            'Total penalty': penalty_values,
            'Penalized': penalty_status,
            'EI value': acqf_values
        }

        df = pd.DataFrame(table_data)

        # Save to CSV
        csv_filename = f'visualization_output/candidate_points_iteration_{i + 1:03d}.csv'
        df.to_csv(csv_filename, index=True, float_format='%.6f')

        return csv_filename

    def create_gif_from_frames(frame_dir, output_path, duration=500):
        """
        Create a GIF from PNG frames in a directory.

        Args:
            frame_dir: Directory containing frame PNG files
            output_path: Path for the output GIF file
            duration: Duration between frames in milliseconds
        """
        # Get all PNG files and sort them
        frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))

        if not frame_files:
            print(f"No PNG files found in {frame_dir}")
            return

        # Load images
        images = []
        for frame_file in frame_files:
            img = Image.open(frame_file)
            images.append(img)

        # Save as GIF
        if images:
            images[0].save(
                output_path,
                save_all=True,
                append_images=images[1:],
                duration=duration,
                loop=0,  # 0 means infinite loop
                optimize=True
            )
            print(f"Created GIF: {output_path}")

    # Create final static plots
    final_iteration = len(pcabo.iterations) - 1

    # Create and save objective function plot
    fig1 = create_objective_plot(final_iteration)
    fig1.savefig('visualization_output/objective_function.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # Create and save weighted PCA plot
    fig2 = create_weighted_pca_plot(final_iteration)
    fig2.savefig('visualization_output/weighted_pca_transformation.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig2)

    # Create and save GP prediction plot
    fig3 = create_gp_plot(final_iteration)
    fig3.savefig('visualization_output/gp_prediction.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig3)

    # Create and save acquisition function plot
    fig4 = create_acquisition_plot(final_iteration)
    fig4.savefig('visualization_output/acquisition_function.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig4)

    # Create candidate points table for the final iteration
    final_csv = create_candidate_table(final_iteration)

    # Create animations for each plot type
    frames = min(len(pcabo.iterations), 30)
    step = max(1, len(pcabo.iterations) // frames)
    frame_indices = list(range(0, len(pcabo.iterations), step))
    if len(pcabo.iterations) - 1 not in frame_indices:
        frame_indices.append(len(pcabo.iterations) - 1)

    print("Creating individual frame images...")
    os.makedirs('visualization_output/frames_objective', exist_ok=True)
    os.makedirs('visualization_output/frames_weighted_pca', exist_ok=True)
    os.makedirs('visualization_output/frames_gp', exist_ok=True)
    os.makedirs('visualization_output/frames_acquisition', exist_ok=True)

    for idx, frame_idx in enumerate(frame_indices):
        # Objective function frames
        fig = create_objective_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_objective/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Weighted PCA frames
        fig = create_weighted_pca_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_weighted_pca/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # GP prediction frames
        fig = create_gp_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_gp/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Acquisition function frames
        fig = create_acquisition_plot(frame_idx)
        fig.savefig(f'visualization_output/frames_acquisition/frame_{idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Create candidate table for this frame
        csv_file = create_candidate_table(frame_idx)
        if csv_file and idx == len(frame_indices) - 1:  # Only print for the last frame
            print(f"Candidate points table saved: {csv_file}")

    print("Creating GIFs from frames...")

    # Create GIFs from the frame sequences
    create_gif_from_frames(
        'visualization_output/frames_objective',
        'visualization_output/objective_function_animation.gif',
        duration=800  # 800ms between frames
    )

    create_gif_from_frames(
        'visualization_output/frames_weighted_pca',
        'visualization_output/weighted_pca_animation.gif',
        duration=800
    )

    create_gif_from_frames(
        'visualization_output/frames_gp',
        'visualization_output/gp_prediction_animation.gif',
        duration=800
    )

    create_gif_from_frames(
        'visualization_output/frames_acquisition',
        'visualization_output/acquisition_function_animation.gif',
        duration=800
    )

    print(f"\nStatic plots saved:")
    print(f"  - visualization_output/objective_function.png")
    print(f"  - visualization_output/weighted_pca_transformation.png")
    print(f"  - visualization_output/gp_prediction.png")
    print(f"  - visualization_output/acquisition_function.png")
    if final_csv:
        print(f"  - {final_csv}")
    print(f"\nAnimated GIFs saved:")
    print(f"  - visualization_output/objective_function_animation.gif")
    print(f"  - visualization_output/weighted_pca_animation.gif")
    print(f"  - visualization_output/gp_prediction_animation.gif")
    print(f"  - visualization_output/acquisition_function_animation.gif")
    print(f"\nFrame sequences saved to:")
    print(f"  - visualization_output/frames_objective/")
    print(f"  - visualization_output/frames_weighted_pca/")
    print(f"  - visualization_output/frames_gp/")
    print(f"  - visualization_output/frames_acquisition/")
    print(f"\nCandidate points tables saved for each iteration in CSV format")

    # Print optimization results
    best_idx = np.argmin(pcabo.fX) if not pcabo.maximization else np.argmax(pcabo.fX)
    print(f"\nOptimization Results:")
    print(f"Best point found: {pcabo.X[best_idx]}")
    print(f"Best value: {pcabo.fX[best_idx]}")
    print(f"Total evaluations: {len(pcabo.X)}")


def calculate_pc1_bounds_intersection(bounds: np.ndarray, pca: MyPCA) -> Tuple[float, float]:
    """
    Calculate the intersection points of the PC1 line with the rectangular bounds.

    Args:
        bounds: 2D array of shape (2, 2) where bounds[0] = [x_min, x_max] and bounds[1] = [y_min, y_max]
        pca: PCA object with transform_to_reduced and transform_to_original methods

    Returns:
        Tuple of (z_min, z_max) representing the intersection points in PC1 space
    """

    # Extract bounds
    x_min, x_max = bounds[0, 0], bounds[0, 1]
    y_min, y_max = bounds[1, 0], bounds[1, 1]

    assert x_min < x_max
    assert y_min < y_max

    # Get the PC1 direction vector and center point
    # We'll use the PCA's mean as a reference point on the line
    center_original = pca.transform_to_original(np.array([[0]]))  # Transform z=0 back to original space
    center_x, center_y = center_original[0, 0], center_original[0, 1]

    # Get another point on the PC1 line to determine direction
    unit_point_original = pca.transform_to_original(np.array([[1]]))  # Transform z=1 back to original space
    direction_x = unit_point_original[0, 0] - center_x
    direction_y = unit_point_original[0, 1] - center_y

    # Normalize direction vector
    direction_norm = np.sqrt(direction_x ** 2 + direction_y ** 2)
    if direction_norm == 0:
        raise ValueError("PC1 direction vector has zero length")

    direction_x /= direction_norm
    direction_y /= direction_norm

    # Find intersections with each boundary line
    intersection_params = []

    # Left boundary (x = x_min)
    if True:  # Avoid division by zero
        t = (x_min - center_x) / direction_x
        y_intersect = center_y + t * direction_y
        intersection_params.append(t)

    # Right boundary (x = x_max)
    if abs(direction_x) > 1e-10:
        t = (x_max - center_x) / direction_x
        y_intersect = center_y + t * direction_y
        intersection_params.append(t)

    # Bottom boundary (y = y_min)
    if abs(direction_y) > 1e-10:  # Avoid division by zero
        t = (y_min - center_y) / direction_y
        x_intersect = center_x + t * direction_x
        intersection_params.append(t)

    # Top boundary (y = y_max)
    if abs(direction_y) > 1e-10:
        t = (y_max - center_y) / direction_y
        x_intersect = center_x + t * direction_x
        intersection_params.append(t)

    if len(intersection_params) < 2:
        raise ValueError(f"Found only {len(intersection_params)} intersection points, expected 2")

    # Remove duplicates and sort
    intersection_params = sorted(list(intersection_params))

    if len(intersection_params) < 2:
        raise ValueError("After removing duplicates, found less than 2 intersection points")

    # Take the two extreme intersection points
    t_min, t_max = intersection_params[1], intersection_params[2]

    # Convert parametric distances to PC1 coordinates
    # Since we normalized the direction vector, t represents the actual distance
    # We need to convert this back to PC1 space
    point_min_original = np.array([[center_x + t_min * direction_x, center_y + t_min * direction_y]])
    point_max_original = np.array([[center_x + t_max * direction_x, center_y + t_max * direction_y]])

    z_min = pca.transform_to_reduced(point_min_original)[0, 0]
    z_max = pca.transform_to_reduced(point_max_original)[0, 0]

    return float(z_min), float(z_max)