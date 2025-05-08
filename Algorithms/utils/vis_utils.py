import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

    def visualize_pcabo(
        self,
        x: torch.Tensor,
        best: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        component_matrix: torch.Tensor,
        mu: torch.Tensor,
        p: Optional[Any] = None,
        fig_size: Tuple[int, int] = (13, 10),
        dpi: int = 100,
        n_points: int = 100,
        margin: float = 0.2,
    ) -> None:
        """Visualize 2d landscape with sampled points and PCs.

        Args:
            x (torch.Tensor): Sampled points
            best (torch.Tensor): Best solution so far
            b (torch.Tensor): Problem bounds
            c (torch.Tensor): Candidates selected on this iteration
            component_matrix (torch.Tensor): The PCA component matrix
            mu (torch.Tensor): mu + mu' - the center of PCA
            p (Optional[Any]): The problem object for mode = 'p'
            fig_size (Tuple[int, int]): Figure size for the plots
            dpi (int): DPI for the plots
            n_points (int): Number of points in each dimension for contour plot grid
            margin (float): Margin to add around the data (as a fraction of data range)
        """
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
        ls0 = torch.linspace(b[0, 0], b[0, 1], n_points)
        ls1 = torch.linspace(b[1, 0], b[1, 1], n_points)
        i, j = torch.meshgrid(ls0, ls1, indexing="ij")
        vis_x[:, :2] = torch.stack([i.flatten(), j.flatten()], dim=1)

        # Plot landscape
        vis_p = torch.tensor([p(k.tolist()) for k in vis_x])
        im = ax.imshow(
            torch.reshape(vis_p, (n_points, n_points)),
            interpolation="spline16",
            origin="lower",
            extent=[min_vis[0], max_vis[0], min_vis[1], max_vis[1]],
            aspect=(max_vis[0] - min_vis[0]) / (max_vis[1] - min_vis[1]),  # "auto",
            cmap="plasma",
            alpha=0.5,
            zorder=1
        )

        # Plot PCs
        line_margin = 0.1
        pc_x = torch.linspace(min_vis[0] - line_margin, max_vis[0] + line_margin, 2)
        pc1_y = (component_matrix[0][1] / component_matrix[0][0]) * (pc_x - mu[0]) + mu[1]
        pc2_y = (component_matrix[1][1] / component_matrix[1][0]) * (pc_x - mu[0]) + mu[1]
        plt.plot(pc_x, pc1_y, color='green', zorder=2)
        plt.plot(pc_x, pc2_y, color='green', alpha=0.2, zorder=2)

        # Plot bounds
        ax.add_patch(patches.Rectangle(b[:2, 0], b[0, 1] - b[0, 0], b[1, 1] - b[1, 0],
                                       linewidth=2, edgecolor='r', facecolor="none", zorder=3))

        # Plot observed points
        ax.scatter(x[:, 0], x[:, 1], color="blue", label="Observed", zorder=4)
        ax.scatter(c[:, 0], c[:, 1], color="red", label="New", zorder=5)
        ax.scatter(best[0], best[1], color="orange", label="Best", zorder=5)

        # Make it pretty
        ax.legend()
        # ax.set_xlabel("X0")
        # ax.set_ylabel("X1")
        # ax.set_title("PCABO")
        fig.colorbar(im, ax=ax)

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)

        img = Image.open(buf)
        self.images["pcabo"].append(img)

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

    def get_frame(self, m: str, index: int) -> Image.Image:
        """Get a frame by index.

        Args:
            m (str): visualization codename: "pcabo" or "acqf"
            index (int): Frame index
        """
        if index < 0 or index >= len(self.images[m]):
            raise IndexError(f"Index {index} out of range (0-{len(self.images[m]) - 1})")

        return self.images[m][index]

    def __len__(self, m: str) -> int:
        """Get the number of saved frames.

        Args:
            m (str): visualization codename: "pcabo" or "acqf"
        """
        return len(self.images[m])
