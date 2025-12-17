#!/usr/bin/env python3
"""Process multiple years of ERA5 samples and store Aurora PREDICTIONS as Zarr.

This script is similar to run_year_forward_with_latents_zarr.py, but instead of
capturing internal latent representations, it saves the actual Aurora model
predictions (output observation space) for comparison experiments.

The predictions are stored with dimensions:
- surface_predictions: (time, lat, lon) per variable (2t, 10u, 10v, msl)
- atmos_predictions: (time, level, lat, lon) per variable (z, u, v, t, q)

This enables comparison between:
- Decoder from Aurora latents (current approach)
- Encoder→Decoder from Aurora predictions (alternative approach)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr
import zarr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from examples.init_exploring.load_era_batch_flexible import (
    load_batch_for_timestep,
    iterate_timesteps,
)
from examples.init_exploring.utils import (
    ensure_static_dataset,
    load_model,
)

DEFAULT_ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/"
    "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)

# Default region (roughly Europe) - same as HRES target region
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -30.0
DEFAULT_LON_MAX = 50.0

# Aurora surface and atmospheric variables
SURFACE_VARS = ["2t", "10u", "10v", "msl"]
ATMOS_VARS = ["z", "u", "v", "t", "q"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Aurora forward passes for multiple years and store "
            "predictions (output observation space) as a Zarr dataset."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=DEFAULT_ZARR_PATH,
        help="Path to the ERA5 Zarr archive",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First year (inclusive) to process",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="Last year (inclusive) to process",
    )
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("/projects/prjs1858/predictions_europe_2018_2020.zarr"),
        help="Output Zarr path for predictions dataset",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/projects/prjs1858/run_year_forward_predictions_zarr"),
        help="Root directory for downloads and cache",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_LAT_MIN,
        help="Minimum latitude of region (degrees)",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_LAT_MAX,
        help="Maximum latitude of region (degrees)",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=DEFAULT_LON_MIN,
        help="Minimum longitude of region (degrees)",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=DEFAULT_LON_MAX,
        help="Maximum longitude of region (degrees)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip timesteps that already exist in the Zarr",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after processing at most this many timesteps",
    )
    return parser.parse_args()


def get_all_timesteps(start_year: int, end_year: int) -> list[datetime]:
    """Get all 6-hourly timesteps for the year range."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    return list(iterate_timesteps(start, end))


def slice_region(
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice data to the specified region.
    
    Args:
        data: Input data array with lat/lon as last two dimensions
        lats: 1D latitude array (decreasing order, 90 to -90)
        lons: 1D longitude array (0 to 360)
        lat_min, lat_max: Latitude bounds
        lon_min, lon_max: Longitude bounds (can be negative for -180 to 180)
    
    Returns:
        Sliced data, sliced lats, sliced lons
    """
    # Convert lon bounds to 0-360 if negative
    if lon_min < 0:
        lon_min_360 = lon_min + 360
    else:
        lon_min_360 = lon_min
    if lon_max < 0:
        lon_max_360 = lon_max + 360
    else:
        lon_max_360 = lon_max
    
    # Handle longitude wrap-around (e.g., -30 to 50 becomes 330 to 50)
    if lon_min_360 > lon_max_360:
        # Region crosses the 0° meridian
        lon_mask = (lons >= lon_min_360) | (lons <= lon_max_360)
    else:
        lon_mask = (lons >= lon_min_360) & (lons <= lon_max_360)
    
    # Latitude mask (lats are typically decreasing: 90 to -90)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    
    # Get indices
    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]
    
    # Sort lon_indices to handle wrap-around correctly
    if lon_min_360 > lon_max_360:
        # Split into two parts and concatenate
        lon_part1 = lon_indices[lons[lon_indices] >= lon_min_360]
        lon_part2 = lon_indices[lons[lon_indices] <= lon_max_360]
        lon_indices = np.concatenate([lon_part1, lon_part2])
    
    # Slice data (assuming last two dims are lat, lon)
    sliced_data = data[..., lat_indices, :]
    sliced_data = sliced_data[..., lon_indices]
    
    # Slice coordinates
    sliced_lats = lats[lat_indices]
    sliced_lons = lons[lon_indices]
    
    # Convert longitudes to -180 to 180 for consistency
    sliced_lons = np.where(sliced_lons > 180, sliced_lons - 360, sliced_lons)
    
    return sliced_data, sliced_lats, sliced_lons


def initialize_zarr_store(
    zarr_path: Path,
    n_timesteps: int,
    region_lats: np.ndarray,
    region_lons: np.ndarray,
    atmos_levels: tuple,
    timesteps: list[datetime],
    region_bounds: dict,
) -> zarr.Group:
    """Initialize Zarr store for Aurora predictions.
    
    Creates separate arrays for each surface and atmospheric variable.
    """
    store = zarr.open(str(zarr_path), mode="w")
    
    n_lat = len(region_lats)
    n_lon = len(region_lons)
    n_levels = len(atmos_levels)
    
    # Create lat coordinate
    lat_arr = store.create_dataset(
        "lat",
        data=region_lats.astype(np.float32),
        dtype="float32",
    )
    lat_arr.attrs["_ARRAY_DIMENSIONS"] = ["lat"]
    
    # Create lon coordinate
    lon_arr = store.create_dataset(
        "lon",
        data=region_lons.astype(np.float32),
        dtype="float32",
    )
    lon_arr.attrs["_ARRAY_DIMENSIONS"] = ["lon"]
    
    # Create time coordinate
    time_values = np.array([np.datetime64(t) for t in timesteps], dtype="datetime64[ns]")
    time_arr = store.create_dataset(
        "time",
        data=time_values,
        dtype="datetime64[ns]",
        chunks=(n_timesteps,),
    )
    time_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    
    # Create level coordinate (pressure levels)
    level_data = np.array([int(lev) for lev in atmos_levels], dtype=np.int32)
    level_arr = store.create_dataset(
        "level",
        data=level_data,
        dtype="int32",
        fill_value=None,
    )
    level_arr.attrs["_ARRAY_DIMENSIONS"] = ["level"]
    
    # Create surface variable arrays
    for var in SURFACE_VARS:
        arr = store.create_dataset(
            var,
            shape=(n_timesteps, n_lat, n_lon),
            chunks=(1, n_lat, n_lon),
            dtype="float32",
            fill_value=np.nan,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]
    
    # Create atmospheric variable arrays
    for var in ATMOS_VARS:
        arr = store.create_dataset(
            var,
            shape=(n_timesteps, n_levels, n_lat, n_lon),
            chunks=(1, n_levels, n_lat, n_lon),
            dtype="float32",
            fill_value=np.nan,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ["time", "level", "lat", "lon"]
    
    # Create tracking array for processed timesteps
    processed_arr = store.create_dataset(
        "processed",
        shape=(n_timesteps,),
        dtype=bool,
        fill_value=False,
    )
    processed_arr.attrs["_ARRAY_DIMENSIONS"] = ["time"]
    
    # Store metadata
    store.attrs["region_bounds"] = json.dumps(region_bounds)
    store.attrs["region_lat_min"] = region_bounds["lat"][0]
    store.attrs["region_lat_max"] = region_bounds["lat"][1]
    store.attrs["region_lon_min"] = region_bounds["lon"][0]
    store.attrs["region_lon_max"] = region_bounds["lon"][1]
    store.attrs["atmos_levels"] = list(atmos_levels)
    store.attrs["n_timesteps"] = n_timesteps
    store.attrs["surface_vars"] = SURFACE_VARS
    store.attrs["atmos_vars"] = ATMOS_VARS
    store.attrs["start_time"] = str(timesteps[0])
    store.attrs["end_time"] = str(timesteps[-1])
    store.attrs["description"] = "Aurora model predictions (observation space)"
    
    print(f"✓ Initialized Zarr store at {zarr_path}")
    print(f"  Lat: {n_lat} values from {region_lats[0]:.2f} to {region_lats[-1]:.2f}")
    print(f"  Lon: {n_lon} values from {region_lons[0]:.2f} to {region_lons[-1]:.2f}")
    print(f"  Levels: {list(atmos_levels)}")
    print(f"  Surface vars: {SURFACE_VARS}")
    print(f"  Atmos vars: {ATMOS_VARS}")
    
    zarr.consolidate_metadata(str(zarr_path))
    
    return store


def main() -> None:
    args = parse_args()
    
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")
    
    work_dir = args.work_dir.expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    static_path = ensure_static_dataset(work_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model once
    model = load_model(device)
    model.eval()
    
    # Region bounds
    requested_bounds = {
        "lat": (args.lat_min, args.lat_max),
        "lon": (args.lon_min, args.lon_max),
    }
    print(
        f"Region: lat=[{args.lat_min}, {args.lat_max}], "
        f"lon=[{args.lon_min}, {args.lon_max}]"
    )
    
    # Pre-compute all timesteps
    all_timesteps = get_all_timesteps(args.start_year, args.end_year)
    n_timesteps = len(all_timesteps)
    print(f"Total timesteps to process: {n_timesteps}")
    
    # Create timestamp to index mapping
    time_to_idx = {t: i for i, t in enumerate(all_timesteps)}
    
    zarr_store = None
    region_lats = None
    region_lons = None
    
    try:
        print("Opening ERA5 Zarr dataset...")
        dataset = xr.open_zarr(args.zarr_path, consolidated=True)
        
        print("Opening static dataset...")
        try:
            static_dataset = xr.open_dataset(static_path, engine="netcdf4")
        except OSError:
            print("Falling back to scipy engine for static dataset...")
            static_dataset = xr.open_dataset(static_path, engine="scipy")
        
        processed_samples = 0
        failed_samples = 0
        skipped_samples = 0
        
        for target_time in all_timesteps:
            time_idx = time_to_idx[target_time]
            
            # Check if already processed
            if args.output_zarr.exists() and zarr_store is None:
                zarr_store = zarr.open(str(args.output_zarr), mode="a")
                if args.skip_existing and zarr_store["processed"][time_idx]:
                    print(f"  ⊘ Skipping {target_time} (already processed)")
                    skipped_samples += 1
                    continue
            
            # Check max samples
            if args.max_samples and processed_samples >= args.max_samples:
                print("\nReached max sample limit, stopping early.")
                break
            
            print(f"  ➤ Processing {target_time.strftime('%Y-%m-%d %H:%M')} (idx={time_idx})...")
            
            try:
                batch = load_batch_for_timestep(
                    target_time=target_time,
                    zarr_path=args.zarr_path,
                    static_path=str(static_path),
                    dataset=dataset,
                    static_dataset=static_dataset,
                )
            except Exception as exc:
                print(f"    ✗ Failed to load batch: {exc}")
                failed_samples += 1
                continue
            
            # Run forward pass
            batch_device = batch.to(device)
            try:
                with torch.no_grad():
                    prediction = model(batch_device)
            except Exception as exc:
                print(f"    ✗ Forward pass failed: {exc}")
                failed_samples += 1
                continue
            
            # Extract predictions and metadata
            pred_lats = prediction.metadata.lat.cpu().numpy()
            pred_lons = prediction.metadata.lon.cpu().numpy()
            atmos_levels = prediction.metadata.atmos_levels
            
            # Initialize Zarr on first successful sample
            if zarr_store is None:
                # Slice to get region coordinates for first sample
                sample_data = prediction.surf_vars["2t"][0, 0].cpu().numpy()
                _, region_lats, region_lons = slice_region(
                    sample_data, pred_lats, pred_lons,
                    args.lat_min, args.lat_max, args.lon_min, args.lon_max
                )
                
                zarr_store = initialize_zarr_store(
                    args.output_zarr,
                    n_timesteps=n_timesteps,
                    region_lats=region_lats,
                    region_lons=region_lons,
                    atmos_levels=atmos_levels,
                    timesteps=all_timesteps,
                    region_bounds=requested_bounds,
                )
            
            # Write surface variables
            try:
                for var in SURFACE_VARS:
                    # Shape: (batch=1, time_history=1, lat, lon)
                    var_data = prediction.surf_vars[var][0, 0].cpu().numpy()
                    sliced_data, _, _ = slice_region(
                        var_data, pred_lats, pred_lons,
                        args.lat_min, args.lat_max, args.lon_min, args.lon_max
                    )
                    zarr_store[var][time_idx] = sliced_data.astype(np.float32)
                
                # Write atmospheric variables
                for var in ATMOS_VARS:
                    # Shape: (batch=1, time_history=1, level, lat, lon)
                    var_data = prediction.atmos_vars[var][0, 0].cpu().numpy()
                    sliced_data, _, _ = slice_region(
                        var_data, pred_lats, pred_lons,
                        args.lat_min, args.lat_max, args.lon_min, args.lon_max
                    )
                    zarr_store[var][time_idx] = sliced_data.astype(np.float32)
                
                # Mark as processed
                zarr_store["processed"][time_idx] = True
                
                print(f"    ✓ Saved timestep {time_idx}/{n_timesteps}")
                processed_samples += 1
                
            except Exception as exc:
                print(f"    ✗ Failed to write to Zarr: {exc}")
                failed_samples += 1
        
    finally:
        # Close datasets
        if "static_dataset" in locals():
            static_close = getattr(static_dataset, "close", None)
            if callable(static_close):
                static_close()
        if "dataset" in locals():
            dataset_close = getattr(dataset, "close", None)
            if callable(dataset_close):
                dataset_close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("Processing Summary:")
    print(f"  ✓ Successfully processed: {processed_samples} samples")
    print(f"  ⊘ Skipped (existing):     {skipped_samples} samples")
    print(f"  ✗ Failed:                 {failed_samples} samples")
    print(f"  Output: {args.output_zarr}")
    print(f"{'='*60}")
    
    # Print example usage
    print(f"""
Example usage with xarray:

    import xarray as xr
    ds = xr.open_zarr("{args.output_zarr}")
    
    # Access coordinates
    lat = ds.lat.values
    lon = ds.lon.values
    levels = ds.level.values
    
    # Get surface variable (2m temperature)
    t2m = ds['2t'].isel(time=0).values  # (lat, lon)
    
    # Get atmospheric variable (temperature at all levels)
    t_atmos = ds['t'].isel(time=0).values  # (level, lat, lon)
    
    # Get by datetime
    t2m = ds['2t'].sel(time="2018-01-01T00:00:00").values
    
    # DataLoader usage
    n_samples = len(ds.time)
    for idx in range(n_samples):
        t2m = ds['2t'].isel(time=idx).values
        msl = ds['msl'].isel(time=idx).values
        # etc.
""")


if __name__ == "__main__":
    main()

# Usage examples:
#
# Process ALL timesteps for years 2018-2020:
#   python examples/run_year_forward_with_predictions_zarr.py \
#       --start-year 2018 --end-year 2020 \
#       --output-zarr /projects/prjs1858/predictions_europe_2018_2020.zarr
#
# Test with a few samples:
#   python examples/run_year_forward_with_predictions_zarr.py \
#       --start-year 2018 --end-year 2018 \
#       --max-samples 10
#
# Resume interrupted processing:
#   python examples/run_year_forward_with_predictions_zarr.py \
#       --start-year 2018 --end-year 2020 \
#       --skip-existing
