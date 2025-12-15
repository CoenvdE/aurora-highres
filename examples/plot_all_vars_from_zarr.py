#!/usr/bin/env python3
"""Decode and plot ALL variables from regional latents to diagnose quality.

This script reads regional latents saved by `run_year_forward_with_latents_zarr.py`,
decodes all surface and atmospheric variables, and produces diagnostic plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from aurora.model.util import unpatchify
from aurora.normalisation import unnormalise_atmos_var, unnormalise_surf_var
from examples.init_exploring.utils import load_model
from examples.plot_region_latents_from_zarr import load_latents_from_zarr


# Variable metadata for plotting
SURF_VARS = ("2t", "10u", "10v", "msl")
ATMOS_VARS = ("z", "u", "v", "t", "q")

VAR_LABELS = {
    "2t": "2m Temperature (K)",
    "10u": "10m U-wind (m/s)",
    "10v": "10m V-wind (m/s)",
    "msl": "Mean Sea Level Pressure (Pa)",
    "z": "Geopotential (m²/s²)",
    "u": "U-wind (m/s)",
    "v": "V-wind (m/s)",
    "t": "Temperature (K)",
    "q": "Specific Humidity (kg/kg)",
}

CMAPS = {
    "2t": "RdBu_r",
    "10u": "RdBu_r",
    "10v": "RdBu_r",
    "msl": "viridis",
    "z": "viridis",
    "u": "RdBu_r",
    "v": "RdBu_r",
    "t": "RdBu_r",
    "q": "Blues",
}


def decode_surface_var(
    model,
    latents: torch.Tensor,
    var_name: str,
    patch_rows: int,
    patch_cols: int,
) -> torch.Tensor:
    """Decode a single surface variable."""
    device = next(model.parameters()).device
    latents = latents.to(device)

    patch_size = model.decoder.patch_size
    total_height = patch_rows * patch_size
    total_width = patch_cols * patch_size

    # Get prediction from head
    head_output = model.decoder.surf_heads[var_name](latents)
    head_output = head_output.reshape(*head_output.shape[:3], -1)

    # Unpatchify
    preds = unpatchify(
        head_output,
        1,  # Single variable
        total_height,
        total_width,
        patch_size,
    )

    # Unnormalize
    preds = unnormalise_surf_var(preds, var_name)

    return preds.detach().cpu()


def decode_atmos_var(
    model,
    latents: torch.Tensor,
    var_name: str,
    patch_rows: int,
    patch_cols: int,
    atmos_levels: torch.Tensor,
) -> torch.Tensor:
    """Decode a single atmospheric variable."""
    device = next(model.parameters()).device
    latents = latents.to(device)

    patch_size = model.decoder.patch_size
    total_height = patch_rows * patch_size
    total_width = patch_cols * patch_size

    # Get prediction from head
    head_output = model.decoder.atmos_heads[var_name](latents)
    head_output = head_output.reshape(*head_output.shape[:3], -1)

    # Unpatchify
    preds = unpatchify(
        head_output,
        1,  # Single variable
        total_height,
        total_width,
        patch_size,
    )

    # Unnormalize - needs squeeze for surface-like treatment
    preds = unnormalise_atmos_var(preds.squeeze(2), var_name, atmos_levels)

    return preds.detach().cpu()


def plot_surface_variables(
    model,
    latents: torch.Tensor,
    patch_rows: int,
    patch_cols: int,
    extent: tuple,
    actual_time,
    output_dir: Path,
):
    """Decode and plot all surface variables."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()

    for idx, var_name in enumerate(SURF_VARS):
        ax = axes[idx]

        # Decode
        decoded = decode_surface_var(
            model, latents, var_name, patch_rows, patch_cols)
        field = decoded[0, 0]  # (height, width)

        # Handle extra dim
        if field.dim() > 2:
            field = field.squeeze(0)

        field_np = field.numpy()

        # Plot
        im = ax.imshow(
            field_np,
            extent=extent,
            origin="upper",
            cmap=CMAPS.get(var_name, "viridis"),
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_title(f"{var_name}: {VAR_LABELS[var_name]}")

        # Stats annotation
        stats_text = f"min={field_np.min():.2f}, max={field_np.max():.2f}\nmean={field_np.mean():.2f}, std={field_np.std():.2f}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    fig.suptitle(f"Surface Variables — {actual_time}", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / \
        f"surface_all_vars_{actual_time.strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_atmos_variables(
    model,
    latents: torch.Tensor,
    patch_rows: int,
    patch_cols: int,
    extent: tuple,
    atmos_levels: torch.Tensor,
    level_to_plot: int,
    actual_time,
    output_dir: Path,
):
    """Decode and plot all atmospheric variables at a given level."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()

    level_idx = list(atmos_levels.numpy()).index(level_to_plot)

    for idx, var_name in enumerate(ATMOS_VARS):
        ax = axes[idx]

        # Decode
        decoded = decode_atmos_var(
            model, latents, var_name, patch_rows, patch_cols, atmos_levels)
        field = decoded[0, 0, level_idx]  # Select level
        field_np = field.numpy()

        # Plot
        im = ax.imshow(
            field_np,
            extent=extent,
            origin="upper",
            cmap=CMAPS.get(var_name, "viridis"),
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.set_title(f"{var_name}: {VAR_LABELS[var_name]}")

        # Stats annotation
        stats_text = f"min={field_np.min():.2f}, max={field_np.max():.2f}\nmean={field_np.mean():.2f}, std={field_np.std():.2f}"
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    # Hide last axis (we have 5 vars, 6 subplots)
    axes[-1].axis("off")

    fig.suptitle(
        f"Atmospheric Variables @ {level_to_plot} hPa — {actual_time}", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / \
        f"atmos_all_vars_{level_to_plot}hPa_{actual_time.strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_atmos_levels(
    model,
    latents: torch.Tensor,
    patch_rows: int,
    patch_cols: int,
    extent: tuple,
    atmos_levels: torch.Tensor,
    var_name: str,
    actual_time,
    output_dir: Path,
):
    """Plot a single atmospheric variable across all pressure levels."""
    n_levels = len(atmos_levels)
    ncols = 4
    nrows = (n_levels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows),
                             subplot_kw={"projection": ccrs.PlateCarree()})
    axes = axes.flatten()

    # Decode
    decoded = decode_atmos_var(
        model, latents, var_name, patch_rows, patch_cols, atmos_levels)

    for level_idx, level in enumerate(atmos_levels.numpy()):
        ax = axes[level_idx]
        field = decoded[0, 0, level_idx]
        field_np = field.numpy()

        im = ax.imshow(
            field_np,
            extent=extent,
            origin="upper",
            cmap=CMAPS.get(var_name, "viridis"),
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=0.5)
        ax.set_title(f"{int(level)} hPa")
        plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)

    # Hide unused axes
    for idx in range(n_levels, len(axes)):
        axes[idx].axis("off")

    fig.suptitle(
        f"{var_name} ({VAR_LABELS[var_name]}) — All Levels — {actual_time}", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / \
        f"atmos_{var_name}_all_levels_{actual_time.strftime('%Y%m%d_%H%M')}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all decoded variables from latent Zarr.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("/projects/prjs1858/latents_europe_2018_2020.zarr"),
        help="Path to the latents Zarr store",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=0,
        help="Integer index of timestep to plot (0-based)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots"),
        help="Directory to save plots",
    )
    parser.add_argument(
        "--atmos-level",
        type=int,
        default=850,
        help="Pressure level for atmospheric comparison plots (default: 850 hPa)",
    )
    parser.add_argument(
        "--plot-all-levels",
        action="store_true",
        help="Also plot each atmospheric variable across all pressure levels",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {args.zarr_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    model.eval()

    # Load surface latents
    print(f"\nLoading surface latents from {args.zarr_path}...")
    (
        surf_latents,
        patch_rows,
        patch_cols,
        region_bounds,
        extent,
        atmos_levels,
        actual_time,
    ) = load_latents_from_zarr(args.zarr_path, time_idx=args.time_idx, mode="surface")

    print(f"  Time: {actual_time}")
    print(f"  Surface latents shape: {surf_latents.shape}")
    print(f"  Patch grid: {patch_rows} x {patch_cols}")
    print(f"  Extent: {extent}")

    # Plot surface variables
    print("\nDecoding and plotting surface variables...")
    plot_surface_variables(
        model, surf_latents, patch_rows, patch_cols, extent, actual_time, args.output_dir
    )

    # Load atmospheric latents
    print(f"\nLoading atmospheric latents...")
    (
        atmos_latents,
        _,
        _,
        _,
        _,
        atmos_levels,
        _,
    ) = load_latents_from_zarr(args.zarr_path, time_idx=args.time_idx, mode="atmos")

    print(f"  Atmospheric latents shape: {atmos_latents.shape}")
    print(f"  Levels: {atmos_levels.numpy()}")

    # Plot atmospheric variables at specified level
    print(
        f"\nDecoding and plotting atmospheric variables @ {args.atmos_level} hPa...")
    plot_atmos_variables(
        model, atmos_latents, patch_rows, patch_cols, extent, atmos_levels,
        args.atmos_level, actual_time, args.output_dir
    )

    # Optionally plot all levels for each variable
    if args.plot_all_levels:
        print("\nPlotting atmospheric variables across all levels...")
        for var_name in ATMOS_VARS:
            print(f"  Processing {var_name}...")
            plot_atmos_levels(
                model, atmos_latents, patch_rows, patch_cols, extent, atmos_levels,
                var_name, actual_time, args.output_dir
            )

    print(f"\n✓ All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# usage: python examples/plot_all_vars_from_zarr.py --zarr-path /projects/prjs1858/latents_europe_2018_2020.zarr --time-idx 0 --output-dir plots --atmos-level 850 --plot-all-levels