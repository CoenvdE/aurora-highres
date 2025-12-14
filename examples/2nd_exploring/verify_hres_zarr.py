#!/usr/bin/env python3
"""Verify HRES Zarr file integrity during or after conversion.

Usage:
    python examples/verify_hres_zarr.py --zarr-path /path/to/hres_europe_2018.zarr
    python examples/verify_hres_zarr.py --year 2018  # Uses default path
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify HRES Zarr file integrity.")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=None,
        help="Path to Zarr file to verify",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Year to verify (uses default path /projects/prjs1858/hres_europe_{year}.zarr)",
    )
    parser.add_argument(
        "--sample-times",
        type=int,
        default=5,
        help="Number of random timesteps to sample for detailed stats",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Determine Zarr path
    if args.zarr_path:
        zarr_path = args.zarr_path
    elif args.year:
        zarr_path = Path(f"/projects/prjs1858/hres_europe_{args.year}.zarr")
    else:
        print("Error: Must specify --zarr-path or --year")
        return
    
    if not zarr_path.exists():
        print(f"❌ Zarr file not found: {zarr_path}")
        return
    
    print(f"{'='*70}")
    print(f"HRES Zarr Verification: {zarr_path}")
    print(f"Verification time: {datetime.now().isoformat()}")
    print(f"{'='*70}\n")
    
    # Open the dataset
    try:
        ds = xr.open_zarr(str(zarr_path))
    except Exception as e:
        print(f"❌ Failed to open Zarr: {e}")
        return
    
    # === Basic Info ===
    print("📊 DATASET OVERVIEW")
    print("-" * 40)
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    print(f"Dimensions: {dict(ds.dims)}")
    print()
    
    # === Time Verification ===
    print("⏰ TIME COORDINATE")
    print("-" * 40)
    times = ds["time"].values
    n_times = len(times)
    print(f"Number of timesteps: {n_times}")
    
    if n_times > 0:
        print(f"First timestep: {times[0]}")
        print(f"Last timestep: {times[-1]}")
        
        # Check time deltas
        if n_times > 1:
            time_diffs = np.diff(times.astype("datetime64[h]").astype(int))
            unique_diffs = np.unique(time_diffs)
            print(f"Time step intervals (hours): {unique_diffs}")
            
            if len(unique_diffs) == 1 and unique_diffs[0] == 6:
                print("✅ Time steps are consistently 6-hourly")
            else:
                print("⚠️  Irregular time steps detected!")
                
            # Check for gaps
            expected_steps = (n_times - 1) * 6  # Expected hours
            actual_hours = (times[-1] - times[0]).astype("timedelta64[h]").astype(int)
            if expected_steps == actual_hours:
                print("✅ No gaps in time series")
            else:
                missing = (actual_hours - expected_steps) // 6
                print(f"⚠️  Missing ~{missing} timesteps (gaps detected)")
    print()
    
    # === Spatial Verification ===
    print("🌍 SPATIAL COORDINATES")
    print("-" * 40)
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    
    if lat_name in ds.coords:
        lats = ds[lat_name].values
        print(f"Latitude range: [{lats.min():.2f}, {lats.max():.2f}]")
        print(f"Latitude points: {len(lats)}")
    
    if lon_name in ds.coords:
        lons = ds[lon_name].values
        print(f"Longitude range: [{lons.min():.2f}, {lons.max():.2f}]")
        print(f"Longitude points: {len(lons)}")
    print()
    
    # === Variable Statistics ===
    print("📈 VARIABLE STATISTICS")
    print("-" * 40)
    
    for var in ds.data_vars:
        print(f"\n  Variable: {var}")
        print(f"  Shape: {ds[var].shape}")
        print(f"  Dtype: {ds[var].dtype}")
        print(f"  Chunks: {ds[var].encoding.get('chunks', 'N/A')}")
        
        # Sample a few timesteps to check for NaNs and get stats
        sample_size = min(args.sample_times, n_times)
        if sample_size > 0:
            sample_indices = np.linspace(0, n_times - 1, sample_size, dtype=int)
            
            nan_counts = []
            mins = []
            maxs = []
            means = []
            
            for idx in sample_indices:
                data = ds[var].isel(time=idx).values
                nan_count = np.isnan(data).sum()
                nan_counts.append(nan_count)
                
                valid_data = data[~np.isnan(data)]
                if len(valid_data) > 0:
                    mins.append(valid_data.min())
                    maxs.append(valid_data.max())
                    means.append(valid_data.mean())
            
            total_nans = sum(nan_counts)
            print(f"  Sampled {sample_size} timesteps:")
            print(f"    Total NaNs in samples: {total_nans}")
            
            if total_nans > 0:
                nan_pct = 100 * total_nans / (sample_size * np.prod(ds[var].shape[1:]))
                print(f"    ⚠️  NaN percentage: {nan_pct:.2f}%")
            else:
                print(f"    ✅ No NaNs detected in samples")
            
            if mins:
                print(f"    Value range: [{min(mins):.4f}, {max(maxs):.4f}]")
                print(f"    Mean: {np.mean(means):.4f}")
    print()
    
    # === Metadata ===
    print("📝 METADATA")
    print("-" * 40)
    for key, value in ds.attrs.items():
        print(f"  {key}: {value}")
    print()
    
    # === Summary ===
    print(f"{'='*70}")
    print("✅ Verification complete!")
    print(f"{'='*70}")
    
    ds.close()


if __name__ == "__main__":
    main()
