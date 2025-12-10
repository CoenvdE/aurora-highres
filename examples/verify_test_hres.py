#!/usr/bin/env python3
"""Verify HRES test pipeline by loading and visualizing the Zarr datasets.

This script:
1. Loads the converted HRES Zarr dataset
2. Loads the static variables Zarr
3. Prints dataset info and statistics
4. Creates visualization plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify HRES test pipeline")
    parser.add_argument("--hres-zarr", type=Path, required=True, help="Path to HRES Zarr")
    parser.add_argument("--static-zarr", type=Path, required=True, help="Path to static Zarr")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("HRES TEST PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Load HRES dataset
    print("\n--- Loading HRES Zarr ---")
    if not args.hres_zarr.exists():
        print(f"ERROR: HRES Zarr not found at {args.hres_zarr}")
        return
    
    ds_hres = xr.open_zarr(str(args.hres_zarr))
    print(f"HRES Dataset: {dict(ds_hres.dims)}")
    print(f"Variables: {list(ds_hres.data_vars)}")
    print(f"Time range: {ds_hres.time.values[0]} to {ds_hres.time.values[-1]}")
    print(f"Timesteps: {len(ds_hres.time)}")
    
    # Print stats for each variable
    for var in ds_hres.data_vars:
        data = ds_hres[var].values
        print(f"\n{var}:")
        print(f"  Shape: {data.shape}")
        print(f"  Min: {np.nanmin(data):.2f}, Max: {np.nanmax(data):.2f}")
        print(f"  Mean: {np.nanmean(data):.2f}, Std: {np.nanstd(data):.2f}")
        print(f"  NaN count: {np.isnan(data).sum()}")
    
    # Load static dataset
    print("\n--- Loading Static Zarr ---")
    if not args.static_zarr.exists():
        print(f"ERROR: Static Zarr not found at {args.static_zarr}")
        return
    
    ds_static = xr.open_zarr(str(args.static_zarr))
    print(f"Static Dataset: {dict(ds_static.dims)}")
    print(f"Variables: {list(ds_static.data_vars)}")
    
    # Create visualization plots
    print("\n--- Creating Visualizations ---")
    
    # Plot 1: First timestep of 2t (2m temperature)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    t2m = ds_hres["2t"].isel(time=0).values
    im1 = axes[0].imshow(t2m, cmap="RdBu_r", origin="upper")
    axes[0].set_title(f"2m Temperature\n{ds_hres.time.values[0]}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="K")
    
    msl = ds_hres["msl"].isel(time=0).values
    im2 = axes[1].imshow(msl / 100, cmap="viridis", origin="upper")
    axes[1].set_title(f"Mean Sea Level Pressure\n{ds_hres.time.values[0]}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label="hPa")
    
    plt.tight_layout()
    plot_path = args.output_dir / "hres_first_timestep.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close()
    
    # Plot 2: Atmospheric temperature at different levels
    if "t" in ds_hres.data_vars:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        t_atmos = ds_hres["t"].isel(time=0)
        levels = t_atmos.coords.get("isobaricInhPa", t_atmos.coords.get("level", None))
        
        if levels is not None:
            level_vals = levels.values
            for i, (ax, lev_idx) in enumerate(zip(axes, [0, 2, 4, 6, 8, 10])):
                if lev_idx < len(level_vals):
                    lev = level_vals[lev_idx]
                    data = t_atmos.isel({levels.dims[0]: lev_idx}).values
                    im = ax.imshow(data, cmap="RdBu_r", origin="upper")
                    ax.set_title(f"Temperature at {lev} hPa")
                    plt.colorbar(im, ax=ax, label="K")
            
            plt.tight_layout()
            plot_path = args.output_dir / "hres_atmos_temp_levels.png"
            plt.savefig(plot_path, dpi=150)
            print(f"  Saved: {plot_path}")
            plt.close()
    
    # Plot 3: Static variables
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for ax, var in zip(axes, ["z", "lsm", "slt"]):
        if var in ds_static.data_vars:
            data = ds_static[var].values
            im = ax.imshow(data, origin="upper")
            ax.set_title(f"Static: {var}")
            plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plot_path = args.output_dir / "static_variables.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved: {plot_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION COMPLETE")
    print(f"  Plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
