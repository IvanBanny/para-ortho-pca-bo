import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import os
from collections import defaultdict
from typing import Optional, Tuple, Callable
from PIL import Image
import io
import torch


class OPCABOVisualizer:
    """Visualizer class for Vanilla BO, PCA-BO, and O-PCA-BO.

    This class generates 2d gifs through sampling iterations.
    Primarily designed for a 2d problem with 1pc case.
    """

    VALID_MODES = ["bo", "pcabo", "pcabo2"]

    def __init__(self, output_dir: str = "./visualizations"):
        """Initialize the PCABOVisualizer.

        Args:
            output_dir: Directory to save the visualizations
        """
        # Create output directory if it doesn't exist
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # List to store plot images
        self.images: defaultdict = defaultdict(list)

    def visualize(
            self,
            mode: str,
            x: torch.Tensor,
            f: torch.Tensor,
            best: torch.Tensor,
            b: torch.Tensor,
            c: torch.Tensor,
            p: Callable,
            acqf: Optional[Callable] = None,
            component_matrix: Optional[torch.Tensor] = None,
            n_components: Optional[int] = None,
            mu: Optional[torch.Tensor] = None,
            model: Optional[Callable] = None,
            fig_size: Optional[Tuple[int, int]] = None,
            dpi: int = 100,
            font_size: int = 22,
            n_points: int = 100,
            margin: float = 0.1,
            x_scaled: Optional[torch.Tensor] = None,
            gpr_indices: Optional[torch.Tensor] = None
    ) -> None:
        """Visualize 2d landscape with sampled points and PCs.

        Args:
            mode (str): Visualization mode, either "bo", "pcabo", or "pcabo2"
            x (torch.Tensor): Sampled points
            f (torch.Tensor): Sampled point values
            best (torch.Tensor): Best solution so far
            b (torch.Tensor): Problem bounds
            c (torch.Tensor): Candidates selected on this iteration
            p (Callable): The problem object
            acqf (Callable): The acquisition function
            component_matrix (torch.Tensor): The PCA component matrix
            n_components (int): The number of components in use
            mu (torch.Tensor): mu + mu' - the center of PCA
            model (Callable): GP model
            fig_size (Tuple[int, int]): Figure size for the plots
            dpi (int): DPI for the plots
            font_size (int): Font size
            n_points (int): Number of points in each dimension for contour plot grid
            margin (float): Margin to add around the data (as a fraction of data range)
            x_scaled (torch.Tensor): Points scaled for PCA
            gpr_indices (torch.Tensor): Indices of points selected to fit GPR
        """

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")

        if mode == "bo":
            fig_size = (14, 10) if fig_size is None else fig_size
        if mode == "pcabo2":
            if model is None:
                raise ValueError("model must be provided for pcabo2 mode")
            fig_size = (16, 16) if fig_size is None else fig_size
        if mode in ["pcabo", "pcabo2"]:
            if acqf is None:
                raise ValueError("acqf must be provided for pcabo mode")
            if component_matrix is None:
                raise ValueError("component_matrix must be provided for pcabo mode")
            if n_components is None:
                raise ValueError("n_components must be provided for pcabo mode")
            if mu is None:
                raise ValueError("mu must be provided for pcabo mode")
            fig_size = (16, 10) if fig_size is None else fig_size

        if mode == "pcabo2":
            fig = plt.figure(figsize=fig_size, dpi=dpi)
            gs = fig.add_gridspec(2, 1, height_ratios=[10, 6], hspace=0.1,
                                  left=0.08, right=0.92, top=0.95, bottom=0.05)
            axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
            ax = axs[0]
        else:
            fig, axs = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)
            ax = axs

        # Calculate landscape points
        min_vis = torch.min(b[:2, 0], x[:, :2].min(dim=0).values)
        max_vis = torch.max(b[:2, 1], x[:, :2].max(dim=0).values)
        margin = (max_vis - min_vis) * margin
        min_vis, max_vis = min_vis - margin, max_vis + margin
        ax.set_xlim(min_vis[0], max_vis[0])
        ax.set_ylim(min_vis[1], max_vis[1])
        ax.set_aspect((max_vis[0] - min_vis[0]) / (max_vis[1] - min_vis[1]))

        vis_x = x.mean(dim=0).unsqueeze(dim=0).repeat(n_points * n_points, 1)
        ls0 = torch.linspace(min_vis[0], max_vis[0], n_points)
        ls1 = torch.linspace(min_vis[1], max_vis[1], n_points)
        i, j = torch.meshgrid(ls0, ls1, indexing="ij")
        vis_x[:, :2] = torch.stack([i.flatten(), j.flatten()], dim=1)

        # Plot landscape
        vis_p = p(vis_x)
        vis_p_shift = vis_p + 1.0 - vis_p.min()
        vis_p_log = vis_p_shift.log()

        im = ax.imshow(
            torch.reshape(vis_p_log, (n_points, n_points)).T,
            interpolation="spline16",
            origin="lower",
            extent=(min_vis[0].item(), max_vis[0].item(), min_vis[1].item(), max_vis[1].item()),
            aspect=(max_vis[0] - min_vis[0]) / (max_vis[1] - min_vis[1]),
            cmap="plasma",
            alpha=0.5,
            zorder=1
        )

        if mode in ["pcabo", "pcabo2"]:
            # Plot PCs
            def x_range(cen, bx, by, pc, line_margin):
                x_range_x = sorted([bx[k].item() for k in range(2)])
                x_range_y = sorted([(by[k] - cen[1]) * component_matrix[pc][0] / component_matrix[pc][1] + cen[0]
                                    for k in range(2)])
                return max(x_range_x[0], x_range_y[0]) - line_margin, min(x_range_x[1], x_range_y[1]) + line_margin

            pc1_x = torch.linspace(
                *x_range(mu,
                         # b[0], b[1],  # For within-constraint pacqf plotting
                         [min_vis[0], max_vis[0]], [min_vis[1], max_vis[1]],  # For within-vis pacqf plotting
                         0, 0), 500)
            pc1_x_plain = torch.linspace(*x_range(mu, [min_vis[0], max_vis[0]], [min_vis[1], max_vis[1]],
                                                  0, 0.1), 2)
            pc2_x = torch.linspace(*x_range(mu, [min_vis[0], max_vis[0]], [min_vis[1], max_vis[1]],
                                            1, 0.1), 2)
            pc3_x = torch.linspace(*x_range(c[0], [min_vis[0], max_vis[0]], [min_vis[1], max_vis[1]],
                                            1, 0.1), 2)

            pc1_y = (component_matrix[0][1] / component_matrix[0][0]) * (pc1_x - mu[0]) + mu[1]
            pc1_y_plain = (component_matrix[0][1] / component_matrix[0][0]) * (pc1_x_plain - mu[0]) + mu[1]
            pc2_y = (component_matrix[1][1] / component_matrix[1][0]) * (pc2_x - mu[0]) + mu[1]
            pc3_y = (component_matrix[1][1] / component_matrix[1][0]) * (pc3_x - c[0, 0]) + c[0, 1]

            ax.plot(pc1_x_plain, pc1_y_plain, color="green", alpha=0.4, zorder=2)
            ax.plot(pc2_x, pc2_y, color="green", alpha=0.4, zorder=2)
            ax.plot(pc3_x, pc3_y, color="green", alpha=0.2, zorder=2)

            pc1_p = torch.stack([pc1_x, pc1_y]).T.reshape(-1, 1, 2)
            pc1_segments = torch.cat([pc1_p[: -1], pc1_p[1:]], 1)

            pc1_p_r = (pc1_p.reshape(-1, 2) - mu) @ component_matrix[: n_components].T
            pc1_acqf_original = acqf(pc1_p_r.unsqueeze(1))
            pc1_acqf_log = (pc1_acqf_original - pc1_acqf_original.min() + 1.0 + 1e-8).log()

            # pc1_norm = LogNorm(vmin=pc1_acqf.min().item() - 1e-5, vmax=pc1_acqf.max().item())
            pc1_lc = LineCollection(pc1_segments, cmap="viridis", zorder=2)  # norm=pc1_norm,
            pc1_lc.set_array(pc1_acqf_log.detach().numpy())
            pc1_lc.set_linewidth(2)
            ax.add_collection(pc1_lc)

            if x_scaled is not None:
                scaled_scatter = ax.scatter(x_scaled[:, 0], x_scaled[:, 1], c=-(f - f.min() + 1.0).log(),
                                            cmap="viridis", s=12, label="Observed", zorder=4, alpha=0.6)

            if mode == "pcabo2":
                pc1_p_r_flat = pc1_p_r.squeeze().detach()

                # Plot acqf
                axs[1].plot(pc1_p_r_flat, pc1_acqf_log.detach(), c="red", label="acqf", zorder=2)

                # Plot model
                posterior = model.posterior(pc1_p_r_flat.unsqueeze(-1))
                posterior_mean = posterior.mean.squeeze()
                posterior_mean_log = (posterior_mean - posterior_mean.min() + 1.0 + 1e-8).log()
                posterior_var = posterior.variance.squeeze().log()/10
                # 95% confidence interval
                ci_lower = posterior_mean_log - 1.96 * posterior_var
                ci_upper = posterior_mean_log + 1.96 * posterior_var

                axs[1].plot(pc1_p_r_flat, posterior_mean_log.detach(), c="green", label="GP mean", zorder=1)
                axs[1].fill_between(
                    pc1_p_r_flat,
                    ci_lower.detach(),
                    ci_upper.detach(),
                    alpha=0.2,
                    color="green",
                    label="+-2std Pred"
                )

                # Plot p
                pc1_prob_original = p(pc1_p.reshape(-1, 2)).detach()
                min_prob = pc1_prob_original.min()
                pc1_prob_log = (pc1_prob_original - min_prob + 1.0 + 1e-8).log()
                axs[1].plot(pc1_p_r_flat, pc1_prob_log.squeeze(), c="blue", label="Problem", zorder=0)

                # Plot projected points
                pc1_x_proj = (x - mu) @ component_matrix[: n_components].T
                axs[1].scatter(pc1_x_proj, (f - min_prob + 1.0 + 1e-8).log(), zorder=3)

                # Plot points selected to fit the GPR
                pc1_x_proj = (x[gpr_indices] - mu) @ component_matrix[: n_components].T
                axs[1].scatter(pc1_x_proj, (f[gpr_indices] - min_prob + 1.0 + 1e-8).log(), c="green", zorder=4)

                # Plot candidates
                pc1_x_proj = (c - mu) @ component_matrix[: n_components].T
                axs[1].scatter(pc1_x_proj, (p(c).detach() - min_prob + 1.0 + 1e-8).log(), c="red", zorder=5)

                axs[1].autoscale()

        # Plot bounds
        ax.add_patch(patches.Rectangle((float(b[0, 0].item()), float(b[1, 0].item())),
                                       float(b[0, 1] - b[0, 0]), float(b[1, 1] - b[1, 0]),
                                       linewidth=2, edgecolor='r', facecolor="none", zorder=3))

        # Plot observed points with a viridis colormap based on value
        observed_scatter = ax.scatter(x[:, 0], x[:, 1], c=-(f - f.min() + 1.0).log(),
                                      cmap="viridis", s=36, label="Observed", zorder=4)

        # Plot candidates in red
        ax.scatter(c[:, 0], c[:, 1], color="red", s=42, label="New", zorder=5)
        # Plot the best sample as an orange X
        ax.scatter(best[0], best[1], color="blue", marker='X', s=100, label="Best", zorder=5)

        # Plot optimum
        if hasattr(p, "solution_x"):
            ax.scatter(p.solution_x[0], p.solution_x[1], color="blue", marker='*', s=160, label="Optimum", zorder=5)

        # Color bars with improved ticks

        # Colorbar for observed points (shows original f values)
        cbar3 = fig.colorbar(observed_scatter, ax=ax, fraction=0.05, pad=0.09, shrink=0.85)
        cbar3.set_label("Problem values", fontsize=font_size)
        cbar3.ax.yaxis.set_label_position("left")
        cbar3.ax.tick_params(labelsize=font_size * 0.64)

        # Generate ticks for observed scatter (which uses -(f - f.min() + 1.0).log())
        n_ticks = 5
        minimization = (1 if hasattr(p, "maximization") and p.maximization else -1)
        displayed_min = minimization * (f - f.min() + 1.0).log().min().item()
        displayed_max = minimization * (f - f.min() + 1.0).log().max().item()
        tick_values = torch.linspace(displayed_min, displayed_max, n_ticks).tolist()
        # Map back to original f values: if displayed = -(f - f.min() + 1.0).log(),
        # then f = exp(-displayed) + f.min() - 1.0
        f_min = f.min().item()
        tick_labels = [f"{torch.exp(-torch.tensor(val)).item() + f_min - 1.0:.1f}" for val in tick_values]
        cbar3.set_ticks(tick_values)
        cbar3.set_ticklabels(tick_labels)

        if mode in ["pcabo", "pcabo2"]:
            # Colorbar for PC line collection (shows original acqf values)
            cbar2 = fig.colorbar(pc1_lc, ax=ax, fraction=0.05, pad=0.11, shrink=0.85)
            cbar2.set_label("Pacqf values", fontsize=font_size)
            cbar2.ax.yaxis.set_label_position("left")

            # Generate ticks for PC line collection (log-normalized)
            n_ticks = 5
            log_min = pc1_acqf_log.min().item()
            log_max = pc1_acqf_log.max().item()
            tick_values = torch.linspace(log_min, log_max, n_ticks).tolist()
            # Map back to original acqf values
            shift_amount = 1.0 - pc1_acqf_original.min().item()
            tick_labels = [f"{torch.exp(torch.tensor(log_val)).item() - shift_amount:.3f}" for log_val in tick_values]
            cbar2.set_ticks(tick_values)
            cbar2.set_ticklabels(tick_labels)
            cbar2.ax.tick_params(labelsize=font_size * 0.64)

        # Landscape colorbar (shows original problem values)
        cbar1 = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.08, shrink=0.85)
        cbar1.set_label("Problem values", fontsize=font_size)
        cbar1.ax.yaxis.set_label_position("left")
        cbar1.ax.tick_params(labelsize=font_size * 0.64)

        # Generate 5 tick values logarithmically spaced in the shifted space
        log_min = vis_p_log.min().item()
        log_max = vis_p_log.max().item()
        log_ticks = torch.linspace(log_min, log_max, 5).tolist()
        # Map back: if log_val = log(vis_p_shift), then original = exp(log_val) - shift_amount
        shift_amount = 1.0 - vis_p.min().item()
        tick_labels = [f"{torch.exp(torch.tensor(log_val)).item() - shift_amount:.1f}" for log_val in log_ticks]
        cbar1.set_ticks(log_ticks)
        cbar1.set_ticklabels(tick_labels)

        # Make it pretty
        ax.tick_params(axis="both", labelsize=font_size)
        ax.legend(fontsize=font_size, markerscale=1.0)
        # ax.set_xlabel("X0")
        # ax.set_ylabel("X1")
        # ax.set_title(mode)

        if mode == "pcabo2":
            axs[1].tick_params(axis="both", labelsize=font_size)
            axs[1].legend(fontsize=font_size, markerscale=1.0)
        else:
            plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        img = Image.open(buf)
        self.images[mode].append(img)

        plt.close(fig)

    def save_gifs(self, prefix: str = "", postfix: Optional[str] = None, duration: int = 1000,
                  loop: int = 0, save_frames: bool = False) -> None:
        """Save the collected images as a GIF.

        Args:
            prefix (str): GIF name prefix e.g. "{prefix}{mode}_{postfix}.gif"
            postfix (str): GIF name postfix e.g. {prefix}{mode}_{postfix}.gif"
            duration (int): Duration per frame in ms
            loop (int): The number of times to loop the GIF (0 for infinite)
            save_frames (bool): Whether to save individual frames to a folder
        """
        for k in self.images.keys():
            if len(self.images[k]) > 0:
                self.images[k][0].save(
                    os.path.join(self.output_dir, f"{prefix}{k}{'' if postfix is None else '_' + postfix}.gif"),
                    save_all=True,
                    append_images=self.images[k][1:],
                    duration=duration,
                    loop=loop,
                    optimize=True
                )

                if save_frames:
                    frames_dir = os.path.join(self.output_dir, f"{k}{'' if postfix is None else '_' + postfix}")
                    os.makedirs(frames_dir, exist_ok=True)

                    for i, img in enumerate(self.images[k]):
                        img.save(os.path.join(frames_dir, f"{i:04d}.png"))

    def clear(self) -> None:
        """Clear all stored images."""
        self.images = defaultdict(list)

    def get_frame(self, mode: str, index: int) -> Image.Image:
        """Get a frame by index.

        Args:
            mode (str): Visualization mode, either "bo" or "pcabo"
            index (int): Frame index
        """
        if index < 0 or index >= len(self.images[mode]):
            raise IndexError(f"Index {index} out of range (0-{len(self.images[mode]) - 1})")

        return self.images[mode][index]

    def __len__(self, mode: str) -> int:
        """Get the number of saved frames.

        Args:
            mode (str): Visualization mode, either "bo" or "pcabo"
        """
        return len(self.images[mode])
