#!/usr/bin/env python3
"""Verify latents test pipeline by loading and visualizing the Zarr dataset.

This script:
1. Loads the extracted latents Zarr dataset
2. Prints dataset info and statistics
3. Creates visualization plots of latent channels

Memory-optimized to avoid OOM issues on cluster nodes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify latents test pipeline")
    parser.add_argument("--latents-zarr", type=Path, required=True, help="Path to latents Zarr")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LATENTS TEST PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Load latents dataset
    print("\n--- Loading Latents Zarr ---")
    if not args.latents_zarr.exists():
        print(f"ERROR: Latents Zarr not found at {args.latents_zarr}")
        return
    
    ds = xr.open_zarr(str(args.latents_zarr))
    print(f"Dataset dimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # Check processed timesteps - load only this small array
    processed = ds["processed"].values
    n_processed = int(processed.sum())
    print(f"\nProcessed timesteps: {n_processed} / {len(processed)}")
    
    if n_processed == 0:
        print("ERROR: No timesteps were processed!")
        return
    
    # Print coordinate info (small arrays, safe to load)
    print(f"\nLat centers: {ds.lat.values[0]:.2f} to {ds.lat.values[-1]:.2f} ({len(ds.lat)} patches)")
    print(f"Lon centers: {ds.lon.values[0]:.2f} to {ds.lon.values[-1]:.2f} ({len(ds.lon)} patches)")
    print(f"Levels: {ds.level.values}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Find first processed timestep
    first_idx = int(np.argmax(processed))
    time_val = ds.time.values[first_idx]
    
    # Load only ONE timestep at a time to avoid OOM
    print(f"\nLoading first processed timestep (idx={first_idx})...")
    surf_data = ds["surface_latents"].isel(time=first_idx).values
    print(f"Surface latents shape: {surf_data.shape}")
    print(f"  Min: {np.nanmin(surf_data):.4f}, Max: {np.nanmax(surf_data):.4f}")
    print(f"  Mean: {np.nanmean(surf_data):.4f}, Std: {np.nanstd(surf_data):.4f}")
    
    pres_data = ds["pressure_latents"].isel(time=first_idx).values
    print(f"\nPressure latents shape: {pres_data.shape}")
    print(f"  Min: {np.nanmin(pres_data):.4f}, Max: {np.nanmax(pres_data):.4f}")
    print(f"  Mean: {np.nanmean(pres_data):.4f}, Std: {np.nanstd(pres_data):.4f}")
    
    # Create visualization plots
    print("\n--- Creating Visualizations ---")
    
    # Plot 1: Surface latent channels
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    n_channels = min(8, surf_data.shape[-1])
    
    for i in range(n_channels):
        data = surf_data[:, :, i]
        im = axes[i].imshow(data, cmap="RdBu_r", origin="upper")
        axes[i].set_title(f"Surface Channel {i}")
        plt.colorbar(im, ax=axes[i])
    
    plt.suptitle(f"Surface Latents - {time_val}", fontsize=12)
    plt.tight_layout()
    plot_path = args.output_dir / "latents_surface_channels.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Pressure latents at different levels
    n_levels = pres_data.shape[0]
    levels = ds.level.values
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    level_indices = [0, 2, 4, 6, 8, 10]
    for ax, lev_idx in zip(axes, level_indices):
        if lev_idx < n_levels:
            # Show first channel at each level
            data = pres_data[lev_idx, :, :, 0]
            im = ax.imshow(data, cmap="RdBu_r", origin="upper")
            ax.set_title(f"Level {levels[lev_idx]} hPa (Channel 0)")
            plt.colorbar(im, ax=ax)
    
    plt.suptitle(f"Pressure Latents - {time_val}", fontsize=12)
    plt.tight_layout()
    plot_path = args.output_dir / "latents_pressure_levels.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close()
    
    # Free memory
    del surf_data, pres_data
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION COMPLETE")
    print(f"  Plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

