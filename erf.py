import os
import argparse
import torch
import numpy as np
from PIL import Image

from VMamba.analyze.utils import EffectiveReceiptiveField
from model.deit import deit_tiny


class visualize:
    @staticmethod
    def get_colormap(name):
        import matplotlib as mpl

        """Handle changes to matplotlib colormap interface in 3.6."""
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return mpl.cm.get_cmap(name)

    @staticmethod
    def draw_image_grid(image: Image, grid=[(0, 0, 1, 1)], **kwargs):
        # grid[0]: (x,y,w,h)
        default = dict(fill=None, outline="red", width=3)
        default.update(kwargs)
        assert (
            isinstance(grid, list) and isinstance(grid[0], tuple) and len(grid[0]) == 4
        )
        from PIL import ImageDraw

        a = ImageDraw.ImageDraw(image)
        for g in grid:
            a.rectangle([(g[0], g[1]), (g[0] + g[2], g[1] + g[3])], **default)
        return image

    @staticmethod
    def visualize_attnmap(
        attnmap,
        savefig="",
        figsize=(18, 16),
        cmap=None,
        sticks=True,
        dpi=400,
        fontsize=35,
        colorbar=True,
        **kwargs,
    ):
        import matplotlib.pyplot as plt

        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        # if isinstance(imgori, torch.Tensor):
        #     imgori = imgori.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        im = ax.imshow(attnmap, cmap=cmap)
        # ax.set_title(title)
        if not sticks:
            ax.set_axis_off()
        if colorbar:
            cbar = ax.figure.colorbar(im, ax=ax)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()

    @staticmethod
    def visualize_attnmaps(
        attnmaps,
        savefig="",
        figsize=(18, 16),
        rows=1,
        cmap=None,
        dpi=400,
        fontsize=35,
        linewidth=2,
        **kwargs,
    ):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt

        vmin = min(
            [
                np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a))
                for a, t in attnmaps
            ]
        )
        vmax = max(
            [
                np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a))
                for a, t in attnmaps
            ]
        )
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        figsize = (cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(
            rows,
            cols,
            squeeze=False,
            sharex="all",
            sharey="all",
            figsize=figsize,
            dpi=dpi,
        )
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                im = axs[i, j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
                axs[i, j].set_title(title)
                axs[i, j].set_yticks([])
                axs[i, j].set_xticks([])
                print(title, "max", np.max(image), "min", np.min(image), end=" | ")
            print("")
        axs[0, 0].figure.colorbar(im, ax=axs)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()
        print("")

    @staticmethod
    def seanborn_heatmap(
        data,
        *,
        vmin=None,
        vmax=None,
        cmap=None,
        center=None,
        robust=False,
        annot=None,
        fmt=".2g",
        annot_kws=None,
        linewidths=0,
        linecolor="white",
        cbar=True,
        cbar_kws=None,
        cbar_ax=None,
        square=False,
        xticklabels="auto",
        yticklabels="auto",
        mask=None,
        ax=None,
        **kwargs,
    ):
        from matplotlib import pyplot as plt
        from seaborn.matrix import _HeatMapper

        # Initialize the plotter object
        plotter = _HeatMapper(
            data,
            vmin,
            vmax,
            cmap,
            center,
            robust,
            annot,
            fmt,
            annot_kws,
            cbar,
            cbar_kws,
            xticklabels,
            yticklabels,
            mask,
        )

        # Add the pcolormesh kwargs here
        kwargs["linewidths"] = linewidths
        kwargs["edgecolor"] = linecolor

        # Draw the plot and return the Axes
        if ax is None:
            ax = plt.gca()
        if square:
            ax.set_aspect("equal")
        plotter.plot(ax, cbar_ax, kwargs)
        mesh = ax.pcolormesh(plotter.plot_data, cmap=plotter.cmap, **kwargs)
        return ax, mesh

    @classmethod
    def visualize_snsmap(
        cls,
        attnmap,
        savefig="",
        figsize=(18, 16),
        cmap=None,
        sticks=True,
        dpi=80,
        fontsize=35,
        linewidth=2,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()

        # 使用高对比度颜色，并强制数值 0 显示为纯黑
        base_cmap_name = cmap if cmap is not None else "inferno"
        base_cmap = cls.get_colormap(base_cmap_name)
        colors = base_cmap(np.linspace(0.0, 1.0, 256))
        colors[0, :3] = 0.0  # 0 -> 纯黑
        colors[0, 3] = 1.0
        cmap = mcolors.ListedColormap(colors)

        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        _, mesh = cls.seanborn_heatmap(
            attnmap,
            xticklabels=sticks,
            yticklabels=sticks,
            cmap=cmap,
            linewidths=0,
            center=0,
            annot=False,
            ax=ax,
            cbar=False,
            annot_kws={"size": 24},
            fmt=".2f",
        )
        cb = ax.figure.colorbar(mesh, ax=ax)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()

    @classmethod
    def visualize_snsmaps(
        cls,
        attnmaps,
        savefig="",
        figsize=(18, 16),
        rows=1,
        cmap=None,
        sticks=True,
        dpi=80,
        fontsize=35,
        linewidth=2,
        **kwargs,
    ):
        # attnmaps: [(map, title), (map, title),...]
        import math
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        vmin = min(
            [
                np.min((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a))
                for a, t in attnmaps
            ]
        )
        vmax = max(
            [
                np.max((a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a))
                for a, t in attnmaps
            ]
        )
        cols = math.ceil(len(attnmaps) / rows)

        base_cmap_name = cmap if cmap is not None else "inferno"
        base_cmap = cls.get_colormap(base_cmap_name)
        colors = base_cmap(np.linspace(0.0, 1.0, 256))
        colors[0, :3] = 0.0
        colors[0, 3] = 1.0
        cmap = mcolors.ListedColormap(colors)

        plt.rcParams["font.size"] = fontsize
        figsize = (cols * figsize[0], rows * figsize[1])
        fig, axs = plt.subplots(
            rows,
            cols,
            squeeze=False,
            sharex="all",
            sharey="all",
            figsize=figsize,
            dpi=dpi,
        )
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    image = np.zeros_like(image)
                    title = "pad"
                else:
                    image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                im = axs[i, j].imshow(image, vmin=vmin, vmax=vmax, cmap=cmap)
                axs[i, j].set_title(title)
        cb = axs[0, 0].figure.colorbar(im, ax=axs)
        cb.outline.set_linewidth(0)
        if savefig == "":
            plt.show()
        else:
            plt.savefig(savefig)
        plt.close()


class FeatureMapWrapper(torch.nn.Module):
    """
    Wrap a ViT-based model to return a 2D feature map [B, C, H, W]
    from the token features after blocks & norm, reshaped to grid.
    """

    def __init__(self, base_model: torch.nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.base
        # replicate forward_features up to norm
        x = b.patch_embed(x)
        # x = b.init_norm(x)
        # for blk in b.blocks:
        #     x = blk(x)
        x = b.blocks(x)
        x = b.norm(x)
        # x: [B, N, C] where N = H*W
        B, N, C = x.shape
        H = int(np.sqrt(N))
        W = H
        assert (
            H * W == N
        ), f"Token count {N} is not a square; cannot reshape to HxW grid."
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x


def load_trained_model(
    ckpt_path: str, img_size: int = 224, drop_path_rate: float = 0.0
) -> torch.nn.Module:
    base = deit_tiny(img_size=img_size, drop_path_rate=drop_path_rate)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = base.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    return base


def build_init_model(
    img_size: int = 224, drop_path_rate: float = 0.0
) -> torch.nn.Module:
    return deit_tiny(img_size=img_size, drop_path_rate=drop_path_rate)


def main():
    parser = argparse.ArgumentParser(
        "ERF visualization for deit_tiny (before vs after training)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/home/data/imagenet",
        help="ImageNet root path (expects val/ under it)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="deit_tiny_patch16_224-a1311bcf.pth",
        help="Path to trained checkpoint",
    )
    parser.add_argument(
        "--img-size", type=int, default=224, help="Input image size for ERF computation"
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=50,
        help="Number of random validation images to average",
    )
    parser.add_argument(
        "--out", type=str, default="show/erf.jpg", help="Output figure path"
    )
    parser.add_argument(
        "--cmap", type=str, default="RdYlGn", help="Matplotlib colormap"
    )
    parser.add_argument(
        "--erf-thresh",
        type=float,
        default=0,
        help="Threshold: ERF values below this are set to 0 before visualization",
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Build models
    base_init = build_init_model(img_size=args.img_size)
    base_trained = load_trained_model(args.ckpt, img_size=args.img_size)

    # Wrap to expose feature map
    model_init_map = FeatureMapWrapper(base_init)
    model_trained_map = FeatureMapWrapper(base_trained)

    # Compute ERF maps
    simpnorm = EffectiveReceiptiveField.simpnorm
    data_path = args.data_path

    print("Computing ERF (before training)...")
    erf_before = EffectiveReceiptiveField.get_input_grad_avg(
        model_init_map,
        # size=args.img_size,
        data_path=data_path,
        num_images=args.num_images,
        norms=simpnorm,
    )

    print("Computing ERF (after training)...")
    erf_after = EffectiveReceiptiveField.get_input_grad_avg(
        model_trained_map,
        # size=args.img_size,
        data_path=data_path,
        num_images=args.num_images,
        norms=simpnorm,
    )

    thresh = float(args.erf_thresh)
    erf_before = np.where(erf_before < thresh, 0.0, erf_before)
    erf_after = np.where(erf_after < thresh, 0.0, erf_after)

    # Visualize side-by-side
    results = [(erf_before, ""), (erf_after, "")]
    visualize.visualize_snsmaps(
        results,
        savefig=args.out,
        rows=2,
        sticks=False,
        # figsize=(10, 5),
        cmap=args.cmap,
        # dpi=120,
        fontsize=70,
    )
    print(f"Saved ERF comparison to: {args.out}")


if __name__ == "__main__":
    main()
