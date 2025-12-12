#!/usr/bin/env python3
"""Slice global static HRES zarr to a specific geographic region and interpolate to HRES grid.

The Aurora static pickle contains global 0.1° data (1801x3600).
This script:
1. Slices to the specified region (default: Europe)
2. Interpolates to match the exact HRES zarr grid coordinates
3. Saves as a new zarr file aligned with HRES data

Static variables:
    - z: Surface geopotential (m²/s²)
    - slt: Soil type (categorical, 0-7)
    - lsm: Land-sea mask (0-1)

Usage:
    python examples/slice_static_to_region.py
    python examples/slice_static_to_region.py --hres-zarr /path/to/hres.zarr
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def convert_lon_to_minus180_180(ds: xr.Dataset) -> xr.Dataset:
    """Convert dataset from 0-360° to -180° to 180° longitude format."""
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    lon_vals = ds[lon_name].values
    
    if lon_vals.min() >= 0 and lon_vals.max() > 180:
        print("Converting lon from 0-360 to -180/180 format...")
        # Roll the dataset to center on 0° longitude
        lon_180_idx = np.argmin(np.abs(lon_vals - 180))
        ds = ds.roll({lon_name: -lon_180_idx}, roll_coords=True)
        
        # Update longitude values
        new_lon = ds[lon_name].values.copy()
        new_lon = np.where(new_lon > 180, new_lon - 360, new_lon)
        ds = ds.assign_coords({lon_name: new_lon})
        ds = ds.sortby(lon_name)
        
    return ds


def interpolate_to_hres_grid(
    ds_static: xr.Dataset,
    hres_lat: np.ndarray,
    hres_lon: np.ndarray,
) -> xr.Dataset:
    """Interpolate static dataset to match exact HRES grid coordinates.
    
    First slices static data to a region slightly larger than HRES (with buffer
    for interpolation), then interpolates. This avoids loading entire global
    static data when HRES region is small.
    
    Args:
        ds_static: Static dataset (may be global or regional)
        hres_lat: Target latitude coordinates from HRES zarr
        hres_lon: Target longitude coordinates from HRES zarr
    
    Returns:
        Static dataset interpolated to HRES grid
    """
    lat_name = "latitude" if "latitude" in ds_static.coords else "lat"
    lon_name = "longitude" if "longitude" in ds_static.coords else "lon"
    
    static_lat = ds_static[lat_name].values
    static_lon = ds_static[lon_name].values
    
    print(f"Static grid: lat=[{static_lat.min():.2f}, {static_lat.max():.2f}] ({len(static_lat)} pts)")
    print(f"             lon=[{static_lon.min():.2f}, {static_lon.max():.2f}] ({len(static_lon)} pts)")
    print(f"Target grid: lat=[{hres_lat.min():.2f}, {hres_lat.max():.2f}] ({len(hres_lat)} pts)")
    print(f"             lon=[{hres_lon.min():.2f}, {hres_lon.max():.2f}] ({len(hres_lon)} pts)")
    
    # Pre-slice static to region with buffer (for interpolation at edges)
    # Buffer of 1 degree should be enough for linear interpolation
    buffer = 1.0
    lat_min_with_buffer = max(hres_lat.min() - buffer, static_lat.min())
    lat_max_with_buffer = min(hres_lat.max() + buffer, static_lat.max())
    lon_min_with_buffer = max(hres_lon.min() - buffer, static_lon.min())
    lon_max_with_buffer = min(hres_lon.max() + buffer, static_lon.max())
    
    # Check if lat is descending
    lat_descending = static_lat[0] > static_lat[-1] if len(static_lat) > 1 else False
    
    if lat_descending:
        lat_slice = slice(lat_max_with_buffer, lat_min_with_buffer)
    else:
        lat_slice = slice(lat_min_with_buffer, lat_max_with_buffer)
    
    lon_slice = slice(lon_min_with_buffer, lon_max_with_buffer)
    
    ds_sliced = ds_static.sel({lat_name: lat_slice, lon_name: lon_slice})
    
    sliced_shape = (len(ds_sliced[lat_name]), len(ds_sliced[lon_name]))
    print(f"Pre-sliced static to {sliced_shape} (with {buffer}° buffer for interpolation)")
    
    # Interpolate to HRES grid
    print(f"Interpolating to HRES grid: ({len(hres_lat)}, {len(hres_lon)})")
    ds_interp = ds_sliced.interp(
        {lat_name: hres_lat, lon_name: hres_lon},
        method='linear',
    )
    
    # Rename coordinates to match HRES convention
    if lat_name != 'latitude':
        ds_interp = ds_interp.rename({lat_name: 'latitude'})
    if lon_name != 'longitude':
        ds_interp = ds_interp.rename({lon_name: 'longitude'})
    
    return ds_interp


def save_to_zarr(ds: xr.Dataset, output_path: Path, overwrite: bool = False) -> None:
    """Save dataset to Zarr format."""
    output_path = Path(output_path)
    
    if output_path.exists():
        if overwrite:
            print(f"Removing existing: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"{output_path} exists. Use --overwrite.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # No chunking needed for static data (loaded once)
    print(f"\nSaving to Zarr: {output_path}")
    ds.to_zarr(output_path, mode="w", consolidated=True)
    
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Done! Size: {total_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Slice global static HRES zarr and interpolate to HRES grid"
    )
    parser.add_argument(
        "--input", "-i", 
        type=Path, 
        default=Path("/projects/prjs1858/static_hres.zarr"),
        help="Input global static zarr"
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path, 
        default=Path("/projects/prjs1858/static_hres_europe.zarr"),
        help="Output sliced static zarr"
    )
    parser.add_argument(
        "--hres-zarr",
        type=Path,
        default=Path("/projects/prjs1858/hres_europe_2018_2020.zarr"),
        help="HRES zarr to get target grid coordinates from"
    )
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Slice Static HRES and Interpolate to HRES Grid")
    print("=" * 60)
    
    # Load HRES zarr to get target grid
    print(f"\nLoading HRES zarr for target grid: {args.hres_zarr}")
    hres_ds = xr.open_zarr(args.hres_zarr, consolidated=True)
    
    # Get HRES coordinates
    hres_lat_name = "latitude" if "latitude" in hres_ds.coords else "lat"
    hres_lon_name = "longitude" if "longitude" in hres_ds.coords else "lon"
    hres_lat = hres_ds[hres_lat_name].values
    hres_lon = hres_ds[hres_lon_name].values
    
    print(f"HRES grid: lat=[{hres_lat.min():.2f}, {hres_lat.max():.2f}] ({len(hres_lat)} pts)")
    print(f"           lon=[{hres_lon.min():.2f}, {hres_lon.max():.2f}] ({len(hres_lon)} pts)")
    
    # Load global static zarr
    print(f"\nLoading static zarr: {args.input}")
    ds_static = xr.open_zarr(args.input, consolidated=True)
    
    print(f"Variables: {list(ds_static.data_vars)}")
    for var_name, var in ds_static.data_vars.items():
        print(f"  {var_name}: {var.dims} {var.shape}")
    
    # Convert to -180/180 longitude format
    ds_static = convert_lon_to_minus180_180(ds_static)
    
    # Interpolate to HRES grid
    ds_interp = interpolate_to_hres_grid(ds_static, hres_lat, hres_lon)
    
    # Add metadata
    ds_interp.attrs["interpolated_to"] = "HRES grid"
    ds_interp.attrs["source_file"] = str(args.input)
    ds_interp.attrs["hres_source"] = str(args.hres_zarr)
    
    print(f"\nInterpolated static data:")
    for var_name, var in ds_interp.data_vars.items():
        print(f"  {var_name}: {var.dims} {var.shape}")
    
    # Save
    save_to_zarr(ds_interp, args.output, args.overwrite)
    
    print(f"\n✓ Complete! Output: {args.output}")
    print(f"  Static data is now on same grid as: {args.hres_zarr}")


if __name__ == "__main__":
    main()
