#!/usr/bin/env python3
"""Plot HRES data from Zarr to verify structure matches Aurora latents.

This script:
1. Loads HRES data from Zarr (created by convert_hres_to_zarr.py)
2. Plots surface and atmospheric variables with coastlines
3. Shows region extent to verify alignment with latent data

Usage:
    python examples/plot_hres_from_zarr.py \
        --zarr-path examples/hres_europe_2018_2020.zarr \
        --time-idx 0

    # List available timesteps:
    python examples/plot_hres_from_zarr.py --list-timesteps
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr


def plot_surface_variable(
    ds: xr.Dataset,
    var_name: str,
    time_idx: int,
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
) -> None:
    """Plot a surface variable on the given axes."""
    data = ds[var_name].isel(time=time_idx).values
    lat = ds.latitude.values if "latitude" in ds.coords else ds.lat.values
    lon = ds.longitude.values if "longitude" in ds.coords else ds.lon.values
    
    # Handle coordinate naming
    if hasattr(ds[var_name], "latitude"):
        lat = ds[var_name].latitude.values
        lon = ds[var_name].longitude.values
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # Use imshow for gridded data (faster than pcolormesh)
    im = ax.imshow(
        data,
        origin="upper",  # HRES data typically stored north-to-south
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
    )
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Colorbar
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    
    # Title with units
    units = {"2t": "K", "msl": "Pa"}
    unit = units.get(var_name, "")
    ax.set_title(f"{var_name} [{unit}]")


def plot_atmospheric_variable(
    ds: xr.Dataset,
    var_name: str,
    time_idx: int,
    level_idx: int,
    ax: plt.Axes,
    extent: tuple[float, float, float, float],
) -> None:
    """Plot an atmospheric variable at a specific pressure level."""
    data = ds[var_name].isel(time=time_idx)
    
    # Get level dimension name
    level_dim = "isobaricInhPa" if "isobaricInhPa" in data.dims else "level"
    data = data.isel({level_dim: level_idx}).values
    
    # Get pressure level value for title
    if level_dim in ds.coords:
        level_val = int(ds[level_dim].values[level_idx])
    else:
        level_val = level_idx
    
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    im = ax.imshow(
        data,
        origin="upper",
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
    )
    
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
    ax.set_title(f"{var_name} @ {level_val} hPa [K]")


def plot_hres_overview(ds: xr.Dataset, time_idx: int, output_dir: Path) -> None:
    """Create overview plot of all HRES variables."""
    # Get ACTUAL extent from data coordinates (not attrs!)
    # This ensures correct alignment with coastlines
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    lat = ds[lat_name].values
    lon = ds[lon_name].values
    
    # Compute extent from actual data bounds
    # Extent for imshow: (lon_min, lon_max, lat_min, lat_max)
    # Account for grid cell size (half cell on each edge)
    lat_res = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.1
    lon_res = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.1
    
    extent = (
        float(lon.min() - lon_res/2),
        float(lon.max() + lon_res/2),
        float(lat.min() - lat_res/2),
        float(lat.max() + lat_res/2),
    )
    
    # Get timestamp
    timestamp = ds.time.values[time_idx]
    timestamp_str = np.datetime64(timestamp, 'us').astype(datetime).strftime("%Y-%m-%d %H:%M")
    
    # Create figure with 2x2 grid: 2t, msl, t@850, t@500
    fig, axes = plt.subplots(
        2, 2,
        figsize=(16, 14),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    
    # Surface: 2t
    plot_surface_variable(ds, "2t", time_idx, axes[0, 0], extent)
    
    # Surface: msl
    plot_surface_variable(ds, "msl", time_idx, axes[0, 1], extent)
    
    # Atmospheric: t at two levels (if available)
    if "t" in ds.data_vars:
        level_dim = "isobaricInhPa" if "isobaricInhPa" in ds.t.dims else "level"
        n_levels = ds[level_dim].size
        
        # Plot at ~850 hPa (low) and ~500 hPa (mid) if available
        # Typically levels are ordered from top to bottom (low pressure to high)
        if n_levels >= 2:
            plot_atmospheric_variable(ds, "t", time_idx, -1, axes[1, 0], extent)  # Lowest level
            plot_atmospheric_variable(ds, "t", time_idx, n_levels // 2, axes[1, 1], extent)  # Mid level
        else:
            plot_atmospheric_variable(ds, "t", time_idx, 0, axes[1, 0], extent)
            axes[1, 1].set_visible(False)
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
    
    fig.suptitle(f"HRES Data - {timestamp_str}\nRegion: [{extent[2]:.1f}°, {extent[3]:.1f}°]N × [{extent[0]:.1f}°, {extent[1]:.1f}°]E", fontsize=14)
    plt.tight_layout()
    
    output_path = output_dir / f"hres_overview_t{time_idx}_{timestamp_str.replace(' ', '_').replace(':', '-')}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def list_available_timesteps(ds: xr.Dataset, n_show: int = 20) -> None:
    """Print available timesteps."""
    n_total = len(ds.time)
    print(f"\n{'='*60}")
    print(f"Total timesteps: {n_total}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"{'='*60}")
    
    print(f"\nFirst {min(n_show, n_total)} timesteps:")
    for i in range(min(n_show, n_total)):
        time_val = ds.time.values[i]
        print(f"  [{i:4d}] {time_val}")
    
    if n_total > n_show:
        print(f"  ... ({n_total - n_show} more timesteps)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HRES data from Zarr to verify structure."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("examples/hres_europe_2018_2020.zarr"),
        help="Path to the HRES Zarr store",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=0,
        help="Timestep index to plot",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of timesteps to plot",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples/hres_vis"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--list-timesteps",
        action="store_true",
        help="List available timesteps and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {args.zarr_path}")
    
    print(f"Opening {args.zarr_path}...")
    ds = xr.open_zarr(str(args.zarr_path))
    print("\n📋 ds.coords:")
    print(ds.coords)
    print("\n📋 Full Dataset:")
    print(ds)
    
    print(f"Dataset dimensions:")
    for dim, size in ds.dims.items():
        print(f"  {dim}: {size}")
    
    if args.list_timesteps:
        list_available_timesteps(ds)
        return
    
    # Plot requested timesteps
    for offset in range(args.n_samples):
        idx = args.time_idx + offset
        if idx >= len(ds.time):
            print(f"Warning: Timestep {idx} out of range, stopping.")
            break
        print(f"\nPlotting timestep {idx}...")
        plot_hres_overview(ds, idx, args.output_dir)
    
    print(f"\n✓ Done! Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

# Usage examples:
#
# List timesteps:
#   python examples/plot_hres_from_zarr.py --list-timesteps
#
# Plot first timestep:
#   python examples/plot_hres_from_zarr.py --time-idx 0
#
# Plot multiple timesteps:
#   python examples/plot_hres_from_zarr.py --time-idx 0 --n-samples 4
