#!/usr/bin/env python3
"""Verify latents test pipeline by loading and visualizing the Zarr dataset.

This script:
1. Loads the extracted latents Zarr dataset
2. Prints dataset info and statistics
3. Creates visualization plots of latent channels
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
    print(f"Dataset dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # Check processed timesteps
    processed = ds["processed"].values
    n_processed = processed.sum()
    print(f"\nProcessed timesteps: {n_processed} / {len(processed)}")
    
    if n_processed == 0:
        print("ERROR: No timesteps were processed!")
        return
    
    # Print coordinate info
    print(f"\nLat centers: {ds.lat.values[0]:.2f} to {ds.lat.values[-1]:.2f} ({len(ds.lat)} patches)")
    print(f"Lon centers: {ds.lon.values[0]:.2f} to {ds.lon.values[-1]:.2f} ({len(ds.lon)} patches)")
    print(f"Levels: {ds.level.values}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Stats for surface latents
    surf = ds["surface_latents"].values
    print(f"\nSurface latents shape: {surf.shape}")
    print(f"  Min: {np.nanmin(surf):.4f}, Max: {np.nanmax(surf):.4f}")
    print(f"  Mean: {np.nanmean(surf):.4f}, Std: {np.nanstd(surf):.4f}")
    print(f"  NaN ratio: {np.isnan(surf).mean():.2%}")
    
    # Stats for pressure latents
    pres = ds["pressure_latents"].values
    print(f"\nPressure latents shape: {pres.shape}")
    print(f"  Min: {np.nanmin(pres):.4f}, Max: {np.nanmax(pres):.4f}")
    print(f"  Mean: {np.nanmean(pres):.4f}, Std: {np.nanstd(pres):.4f}")
    print(f"  NaN ratio: {np.isnan(pres).mean():.2%}")
    
    # Create visualization plots
    print("\n--- Creating Visualizations ---")
    
    # Find first processed timestep
    first_idx = np.argmax(processed)
    time_val = ds.time.values[first_idx]
    
    # Plot 1: Surface latent channels
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    surf_data = ds["surface_latents"].isel(time=first_idx).values
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
    pres_data = ds["pressure_latents"].isel(time=first_idx).values
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
    
    # Plot 3: Time series of mean latent values
    if n_processed > 1:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get processed indices
        proc_idx = np.where(processed)[0]
        times = ds.time.values[proc_idx]
        
        # Surface latent mean over time
        surf_means = [np.nanmean(ds["surface_latents"].isel(time=i).values) for i in proc_idx]
        axes[0].plot(range(len(proc_idx)), surf_means, 'b-o', markersize=3)
        axes[0].set_xlabel("Timestep index")
        axes[0].set_ylabel("Mean latent value")
        axes[0].set_title("Surface Latents - Mean over Time")
        axes[0].grid(True, alpha=0.3)
        
        # Pressure latent mean over time
        pres_means = [np.nanmean(ds["pressure_latents"].isel(time=i).values) for i in proc_idx]
        axes[1].plot(range(len(proc_idx)), pres_means, 'r-o', markersize=3)
        axes[1].set_xlabel("Timestep index")
        axes[1].set_ylabel("Mean latent value")
        axes[1].set_title("Pressure Latents - Mean over Time")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = args.output_dir / "latents_time_series.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved: {plot_path}")
        plt.close()
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION COMPLETE")
    print(f"  Plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
