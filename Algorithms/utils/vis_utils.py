import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
import os
from collections import defaultdict
from typing import Optional, Tuple, Any
from PIL import Image
import io
import torch


class PCABOVisualizer:
    """Visualizer class for PCA-BO to create animations of the acquisition function.

    This class generates 2d visualizations of the PCA-BO sampling.
    Primarily designed for a 2d problem with 1pc case.
    """

    VALID_MODES = ["bo", "pcabo"]

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
            p: Any,
            acqf: Any,
            component_matrix: Optional[torch.Tensor] = None,
            n_components: Optional[int] = None,
            mu: Optional[torch.Tensor] = None,
            fig_size: Optional[Tuple[int, int]] = None,
            dpi: int = 100,
            font_size: int = 22,
            n_points: int = 100,
            margin: float = 0.2,
    ) -> None:
        """Visualize 2d landscape with sampled points and PCs.

        Args:
            mode (str): Visualization mode, either "bo" or "pcabo"
            x (torch.Tensor): Sampled points
            f (torch.Tensor): Sampled point values
            best (torch.Tensor): Best solution so far
            b (torch.Tensor): Problem bounds
            c (torch.Tensor): Candidates selected on this iteration
            component_matrix (torch.Tensor): The PCA component matrix
            n_components (int): The number of components in use
            mu (torch.Tensor): mu + mu' - the center of PCA
            p (Any): The problem object for mode = 'p'
            acqf (Any): The acquisition function
            fig_size (Tuple[int, int]): Figure size for the plots
            dpi (int): DPI for the plots
            font_size (int): Font size
            n_points (int): Number of points in each dimension for contour plot grid
            margin (float): Margin to add around the data (as a fraction of data range)
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")

        if mode == "bo":
            fig_size = (14, 10) if fig_size is None else fig_size
        elif mode == "pcabo":
            if component_matrix is None:
                raise ValueError("component_matrix must be provided for pcabo mode")
            if n_components is None:
                raise ValueError("n_components must be provided for pcabo mode")
            if mu is None:
                raise ValueError("mu must be provided for pcabo mode")
            fig_size = (16, 10) if fig_size is None else fig_size

        fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=dpi)

        # Calculate landscape points
        min_vis = torch.min(b[:2, 0], x[:, :2].min(dim=0).values)
        max_vis = torch.max(b[:2, 1], x[:, :2].max(dim=0).values)
        margin = (max_vis - min_vis) * margin
        min_vis, max_vis = min_vis - margin, max_vis + margin
        plt.xlim(min_vis[0], max_vis[0])
        plt.ylim(min_vis[1], max_vis[1])
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

        if mode == "pcabo":
            # Plot PCs
            line_margin = 0.1
            pc1_x = torch.linspace(
                max(min_vis[0].item() - line_margin,
                    (min_vis[1] - mu[1]) * component_matrix[0][0] / component_matrix[0][1] + mu[0]),
                min(max_vis[0].item() + line_margin,
                    (max_vis[1] - mu[1]) * component_matrix[0][0] / component_matrix[0][1] + mu[0]),
                500
            )
            pc2_x = torch.linspace(
                max(min_vis[0].item() - line_margin,
                    (min_vis[1] - mu[1]) * component_matrix[1][0] / component_matrix[1][1] + mu[0]),
                min(max_vis[0].item() + line_margin,
                    (max_vis[1] - mu[1]) * component_matrix[1][0] / component_matrix[1][1] + mu[0]),
                2
            )
            pc3_x = torch.linspace(
                max(min_vis[0].item() - line_margin,
                    (min_vis[1] - c[0, 1]) * component_matrix[1][0] / component_matrix[1][1] + c[0, 0]),
                min(max_vis[0].item() + line_margin,
                    (max_vis[1] - c[0, 1]) * component_matrix[1][0] / component_matrix[1][1] + c[0, 0]),
                2
            )

            pc1_y = (component_matrix[0][1] / component_matrix[0][0]) * (pc1_x - mu[0]) + mu[1]
            pc2_y = (component_matrix[1][1] / component_matrix[1][0]) * (pc2_x - mu[0]) + mu[1]
            pc3_y = (component_matrix[1][1] / component_matrix[1][0]) * (pc3_x - c[0, 0]) + c[0, 1]

            pc1_p = torch.stack([pc1_x, pc1_y]).T.reshape(-1, 1, 2)
            pc1_segments = torch.cat([pc1_p[: -1], pc1_p[1:]], 1)

            pc1_p_r = (pc1_p.reshape(-1, 2) - mu) @ component_matrix[: n_components].T
            pc1_acqf_original = acqf(pc1_p_r.unsqueeze(1))
            pc1_acqf = pc1_acqf_original + 1.0 - pc1_acqf_original.min()

            pc1_norm = LogNorm(vmin=pc1_acqf.min().item() - 1e-5, vmax=pc1_acqf.max().item())
            pc1_lc = LineCollection(pc1_segments, cmap="viridis", norm=pc1_norm, zorder=2)
            pc1_lc.set_array(pc1_acqf.detach().numpy())
            pc1_lc.set_linewidth(2)
            ax.add_collection(pc1_lc)

            plt.plot(pc2_x, pc2_y, color="green", alpha=0.4, zorder=2)
            plt.plot(pc3_x, pc3_y, color="green", alpha=0.2, zorder=2)

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

        if mode == "pcabo":
            # Colorbar for PC line collection (shows original acqf values)
            cbar2 = fig.colorbar(pc1_lc, ax=ax, fraction=0.05, pad=0.11, shrink=0.85)
            cbar2.set_label("Pacqf values", fontsize=font_size)
            cbar2.ax.yaxis.set_label_position("left")
            cbar2.ax.tick_params(labelsize=font_size * 0.64)

            # Generate ticks for PC line collection (log-normalized)
            n_ticks = 5
            pc1_shifted_min = pc1_acqf.min().item()
            pc1_shifted_max = pc1_acqf.max().item()
            tick_values = torch.logspace(
                torch.log10(torch.tensor(pc1_shifted_min)).item(),
                torch.log10(torch.tensor(pc1_shifted_max)).item(),
                n_ticks
            ).tolist()
            # Map back to original acqf values
            pc1_original_min = pc1_acqf_original.min().item()
            shift_amount = 1.0 - pc1_original_min
            tick_labels = [f"{val - shift_amount:.3f}" for val in tick_values]
            cbar2.set_ticks(tick_values)
            cbar2.set_ticklabels(tick_labels)

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
        plt.tick_params(axis='both', labelsize=font_size)
        ax.legend(fontsize=font_size, markerscale=1.0)
        # ax.set_xlabel("X0")
        # ax.set_ylabel("X1")
        # ax.set_title(mode)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        img = Image.open(buf)
        self.images[mode].append(img)

        plt.close(fig)

    def save_gifs(self, postfix: Optional[str] = None, duration: int = 1000, loop: int = 0) -> None:
        """Save the collected images as a GIF.

        Args:
            postfix (str): GIF name postfix e.g. "pcabo_{postfix}.gif"
            duration (int): Duration per frame in ms
            loop (int): The number of times to loop the GIF (0 for infinite)
        """
        for k in self.images.keys():
            if len(self.images[k]) > 0:
                self.images[k][0].save(
                    os.path.join(self.output_dir, f"{k}{'' if postfix is None else '_' + postfix}.gif"),
                    save_all=True,
                    append_images=self.images[k][1:],
                    duration=duration,
                    loop=loop,
                    optimize=True
                )

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
            raise IndexError(f"Index {index} out of range (0-{len(self.images[m]) - 1})")

        return self.images[mode][index]

    def __len__(self, mode: str) -> int:
        """Get the number of saved frames.

        Args:
            mode (str): Visualization mode, either "bo" or "pcabo"
        """
        return len(self.images[mode])
