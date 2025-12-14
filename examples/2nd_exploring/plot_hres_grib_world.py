#!/usr/bin/env python3
"""Plot full-world HRES GRIB data with region box overlay.

This script:
1. Loads raw GRIB files from downloads (full global coverage)
2. Plots the data with coastlines
3. Draws a box showing the region that will be cropped

Usage:
    python examples/plot_hres_grib_world.py \
        --hres-dir examples/downloads/hres \
        --date 2018-01-01 --hour 0
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr


# Default region bounds (same as convert_hres_to_zarr.py)
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -30.0
DEFAULT_LON_MAX = 50.0


def load_surface_grib(hres_dir: Path, date: datetime, var_name: str) -> xr.DataArray | None:
    """Load surface variable from GRIB (full world)."""
    day_dir = hres_dir / f"{date:%Y/%m/%d}"
    file_path = day_dir / date.strftime(f"surf_{var_name}_%Y-%m-%d.grib")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    try:
        ds = xr.open_dataset(str(file_path), engine="cfgrib")
        grib_names = {"2t": ["t2m", "2t"], "msl": ["msl"]}
        for name in grib_names.get(var_name, [var_name]):
            if name in ds.data_vars:
                return ds[name]
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def load_atmos_grib(hres_dir: Path, date: datetime, var_name: str, hour: int) -> xr.DataArray | None:
    """Load atmospheric variable from GRIB (full world)."""
    day_dir = hres_dir / f"{date:%Y/%m/%d}"
    file_path = day_dir / f"atmos_{var_name}_{date:%Y-%m-%d}_{hour:02d}.grib"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    try:
        ds = xr.open_dataset(
            str(file_path),
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
        if var_name in ds.data_vars:
            return ds[var_name]
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def draw_region_box(ax, lat_min, lat_max, lon_min, lon_max, color='red', linewidth=2):
    """Draw a rectangle showing the region bounds."""
    # Handle negative longitudes (convert to 0-360 if needed for display)
    rect = mpatches.Rectangle(
        (lon_min, lat_min),
        lon_max - lon_min,
        lat_max - lat_min,
        linewidth=linewidth,
        edgecolor=color,
        facecolor='none',
        transform=ccrs.PlateCarree(),
        zorder=10,
    )
    ax.add_patch(rect)


def plot_world_with_region(
    data: xr.DataArray,
    var_name: str,
    hour: int,
    date: datetime,
    region_bounds: dict,
    output_dir: Path,
    level_idx: int | None = None,
) -> None:
    """Plot full world data with region box overlay."""
    fig, ax = plt.subplots(
        figsize=(16, 8),
        subplot_kw={'projection': ccrs.PlateCarree()},
    )
    
    # Get coordinates
    lat_name = "latitude" if "latitude" in data.coords else "lat"
    lon_name = "longitude" if "longitude" in data.coords else "lon"
    lat = data[lat_name].values
    lon = data[lon_name].values
    
    # Select hour if surface (multiple steps) or level if atmos
    plot_data = data
    if "step" in plot_data.dims:
        hour_to_idx = {0: 0, 6: 1, 12: 2, 18: 3}
        plot_data = plot_data.isel(step=hour_to_idx.get(hour, 0))
    elif "time" in plot_data.dims and plot_data.sizes.get("time", 1) > 1:
        hour_to_idx = {0: 0, 6: 1, 12: 2, 18: 3}
        plot_data = plot_data.isel(time=hour_to_idx.get(hour, 0))
    
    level_str = ""
    if level_idx is not None and "isobaricInhPa" in plot_data.dims:
        level_val = int(plot_data.isobaricInhPa.values[level_idx])
        plot_data = plot_data.isel(isobaricInhPa=level_idx)
        level_str = f" @ {level_val} hPa"
    
    values = plot_data.values
    
    # Set up map
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.3)
    
    # Plot data
    im = ax.pcolormesh(
        lon, lat, values,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        shading='auto',
    )
    
    # Draw region box
    draw_region_box(
        ax,
        region_bounds["lat"][0],
        region_bounds["lat"][1],
        region_bounds["lon"][0],
        region_bounds["lon"][1],
        color='lime',
        linewidth=3,
    )
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    
    # Colorbar
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.6)
    
    # Title
    ax.set_title(
        f"HRES {var_name}{level_str} - {date:%Y-%m-%d} {hour:02d}:00 UTC\n"
        f"Region box: [{region_bounds['lat'][0]}°, {region_bounds['lat'][1]}°]N × "
        f"[{region_bounds['lon'][0]}°, {region_bounds['lon'][1]}°]E",
        fontsize=12,
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"hres_world_{var_name}_{date:%Y%m%d}_{hour:02d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hres-dir", type=Path, default=Path("examples/downloads/hres"))
    parser.add_argument("--date", type=str, default="2018-01-01", help="Date YYYY-MM-DD")
    parser.add_argument("--hour", type=int, default=0, choices=[0, 6, 12, 18])
    parser.add_argument("--output-dir", type=Path, default=Path("examples/hres_vis"))
    parser.add_argument("--lat-min", type=float, default=DEFAULT_LAT_MIN)
    parser.add_argument("--lat-max", type=float, default=DEFAULT_LAT_MAX)
    parser.add_argument("--lon-min", type=float, default=DEFAULT_LON_MIN)
    parser.add_argument("--lon-max", type=float, default=DEFAULT_LON_MAX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = datetime.strptime(args.date, "%Y-%m-%d")
    region_bounds = {
        "lat": (args.lat_min, args.lat_max),
        "lon": (args.lon_min, args.lon_max),
    }
    
    print(f"Loading HRES data for {date:%Y-%m-%d} {args.hour:02d}:00...")
    
    # Plot 2t (surface temperature)
    data = load_surface_grib(args.hres_dir, date, "2t")
    if data is not None:
        plot_world_with_region(data, "2t", args.hour, date, region_bounds, args.output_dir)
    
    # Plot msl (mean sea level pressure)
    data = load_surface_grib(args.hres_dir, date, "msl")
    if data is not None:
        plot_world_with_region(data, "msl", args.hour, date, region_bounds, args.output_dir)
    
    # Plot t (atmospheric temperature) at ~850 hPa
    data = load_atmos_grib(args.hres_dir, date, "t", args.hour)
    if data is not None:
        n_levels = data.isobaricInhPa.size
        plot_world_with_region(data, "t", args.hour, date, region_bounds, args.output_dir, level_idx=n_levels-1)
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
