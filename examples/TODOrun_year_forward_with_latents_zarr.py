#!/usr/bin/env python3
"""Process multiple years of ERA5 samples and store region-specific Aurora latents as Zarr.

This script creates a single Zarr dataset with proper time indexing, optimized for
DataLoader usage in downstream training. Latents are stored with dimensions:
- surface_latents: (time, patch_rows, patch_cols, channels)
- pressure_latents: (time, levels, patch_rows, patch_cols, channels)

The time dimension allows simple integer indexing for efficient batch loading.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import xarray as xr
import zarr

from examples.extract_latents import register_latent_hooks
from examples.init_exploring.load_era_batch_flexible import (
    load_batch_for_timestep,
    iterate_timesteps,
)
from examples.init_exploring.utils import (
    compute_patch_grid,
    ensure_static_dataset,
    load_model,
)
from examples.init_exploring.region_selection import (
    prepare_region_for_capture,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Aurora forward passes for multiple years and store "
            "region-specific latents as a single Zarr dataset."
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
        default=Path("examples/latents_europe_2018_2020.zarr"),
        help="Output Zarr path for latents dataset",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/run_year_forward_latents_zarr"),
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


def initialize_zarr_store(
    zarr_path: Path,
    n_timesteps: int,
    surface_shape: Tuple[int, int, int],  # (rows, cols, channels)
    pressure_shape: Tuple[int, int, int, int],  # (levels, rows, cols, channels)
    region_bounds: dict,
    patch_grid: dict,
    atmos_levels: tuple,
    timesteps: list[datetime],
) -> zarr.Group:
    """Initialize Zarr store with proper structure for latents storage.
    
    Aurora produces two types of latents:
    - surface_latents: (time, lat, lon, channels) - 1 "level" for surface variables
    - pressure_latents: (time, level, lat, lon, channels) - 13 pressure levels
    
    These are stored SEPARATELY (not stacked) because they have different shapes.
    The surface latents correspond to surface variables (2t, 10u, 10v, msl).
    The pressure latents correspond to atmospheric variables (z, u, v, t, q) at 13 levels.
    
    Coordinates:
        lat: 1D array of patch center latitudes
        lon: 1D array of patch center longitudes
        level: 1D array of pressure levels [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
        
    Bounds (for each patch):
        lat_bounds: (n_lat, 2) with [lat_min, lat_max] for each row
        lon_bounds: (n_lon, 2) with [lon_min, lon_max] for each col
    
    Usage:
        ds.surface_latents.sel(lat=55.0, lon=10.0, method="nearest")
        ds.lat.values  # patch center latitudes
        ds.lat_bounds.values  # [[lat_min, lat_max], ...] for each lat
    """
    
    store = zarr.open(str(zarr_path), mode="w")
    
    # Extract patch center coordinates as 1D arrays
    # patch_grid["centres"] is (n_patches, 2) with [lat, lon]
    s_rows, s_cols, s_channels = surface_shape
    centres_2d = patch_grid["centres"].numpy().reshape(s_rows, s_cols, 2)
    lat_centres = centres_2d[:, 0, 0]  # Take first column's lat values
    lon_centres = centres_2d[0, :, 1]  # Take first row's lon values
    
    # Create lat coordinate (patch center latitudes)
    store.create_dataset(
        "lat",
        data=lat_centres.astype(np.float32),
        dtype="float32",
    )
    
    # Create lon coordinate (patch center longitudes)
    store.create_dataset(
        "lon",
        data=lon_centres.astype(np.float32),
        dtype="float32",
    )
    
    # Create lat/lon bounds for each patch
    # Estimate bounds from spacing between centers
    lat_spacing = np.abs(np.diff(lat_centres).mean()) if len(lat_centres) > 1 else 4.0
    lon_spacing = np.abs(np.diff(lon_centres).mean()) if len(lon_centres) > 1 else 4.0
    
    lat_bounds = np.stack([
        lat_centres - lat_spacing / 2,
        lat_centres + lat_spacing / 2
    ], axis=-1).astype(np.float32)
    
    lon_bounds = np.stack([
        lon_centres - lon_spacing / 2,
        lon_centres + lon_spacing / 2
    ], axis=-1).astype(np.float32)
    
    store.create_dataset(
        "lat_bounds",
        data=lat_bounds,
        dtype="float32",
    )
    
    store.create_dataset(
        "lon_bounds",
        data=lon_bounds,
        dtype="float32",
    )
    
    # Create time coordinate (as int64 nanoseconds for xarray compatibility)
    time_values = np.array([np.datetime64(t) for t in timesteps], dtype="datetime64[ns]")
    store.create_dataset(
        "time",
        data=time_values,
        dtype="datetime64[ns]",
        chunks=(n_timesteps,),  # Single chunk for the time coordinate
    )
    
    # Create levels coordinate for pressure data
    # Aurora uses 13 levels: 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000 hPa
    store.create_dataset(
        "level",
        data=np.array(list(atmos_levels), dtype=np.int32),
        dtype="int32",
    )
    
    # Surface latents: (time, lat, lon, channels) - SEPARATE from pressure
    # This contains latents for surface-level variables (2t, 10u, 10v, msl)
    store.create_dataset(
        "surface_latents",
        shape=(n_timesteps, s_rows, s_cols, s_channels),
        chunks=(1, s_rows, s_cols, s_channels),  # 1 timestep per chunk
        dtype="float32",
        fill_value=np.nan,
    )
    
    # Pressure latents: (time, level, lat, lon, channels) - 13 pressure levels
    # This contains latents for atmospheric variables (z, u, v, t, q) at each pressure level
    p_levels, p_rows, p_cols, p_channels = pressure_shape
    store.create_dataset(
        "pressure_latents",
        shape=(n_timesteps, p_levels, p_rows, p_cols, p_channels),
        chunks=(1, p_levels, p_rows, p_cols, p_channels),  # 1 timestep per chunk
        dtype="float32",
        fill_value=np.nan,
    )
    
    # Create tracking array for which timesteps are processed
    store.create_dataset(
        "processed",
        shape=(n_timesteps,),
        dtype=bool,
        fill_value=False,
    )
    
    # Store metadata as attributes (consistent naming with HRES zarr)
    store.attrs["region_bounds"] = json.dumps(region_bounds)
    store.attrs["region_lat_min"] = region_bounds["lat"][0]
    store.attrs["region_lat_max"] = region_bounds["lat"][1]
    store.attrs["region_lon_min"] = region_bounds["lon"][0]
    store.attrs["region_lon_max"] = region_bounds["lon"][1]
    store.attrs["atmos_levels"] = list(atmos_levels)
    store.attrs["n_timesteps"] = n_timesteps
    store.attrs["surface_shape"] = list(surface_shape)
    store.attrs["pressure_shape"] = list(pressure_shape)
    store.attrs["start_time"] = str(timesteps[0])
    store.attrs["end_time"] = str(timesteps[-1])
    
    # Store patch grid info for reference
    store.attrs["patch_row_start"] = int(patch_grid["row_start"])
    store.attrs["patch_row_end"] = int(patch_grid["row_end"])
    store.attrs["patch_col_start"] = int(patch_grid["col_start"])
    store.attrs["patch_col_end"] = int(patch_grid["col_end"])
    store.attrs["patch_shape"] = list(patch_grid["patch_shape"])
    
    print(f"✓ Initialized Zarr store at {zarr_path}")
    print(f"  Lat centers: {len(lat_centres)} values from {lat_centres[0]:.2f} to {lat_centres[-1]:.2f}")
    print(f"  Lon centers: {len(lon_centres)} values from {lon_centres[0]:.2f} to {lon_centres[-1]:.2f}")
    print(f"  Levels: {list(atmos_levels)}")
    print(f"  Surface latents: (time={n_timesteps}, lat={s_rows}, lon={s_cols}, channels={s_channels})")
    print(f"  Pressure latents: (time={n_timesteps}, level={p_levels}, lat={p_rows}, lon={p_cols}, channels={p_channels})")
    
    return store


def main() -> None:
    args = parse_args()
    
    # Validate year range
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    work_dir = args.work_dir.expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)
    
    output_zarr = args.output_zarr.expanduser()
    
    static_path = ensure_static_dataset(work_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model once
    model = load_model(device)
    model.eval()

    # Register hooks once
    captures: dict[str, torch.Tensor] = {}
    handles, decoder_cleanup = register_latent_hooks(model, captures)

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
    
    # Create timestamp to index mapping for fast lookup
    time_to_idx = {t: i for i, t in enumerate(all_timesteps)}

    # Track if Zarr is initialized
    zarr_store = None
    patch_grid = None
    region_info = None
    
    try:
        print("Opening ERA5 Zarr dataset once...")
        dataset = xr.open_zarr(args.zarr_path, consolidated=True)

        print("Opening static dataset once...")
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
            
            # Check if Zarr exists and this timestep is already processed
            if output_zarr.exists() and zarr_store is None:
                zarr_store = zarr.open(str(output_zarr), mode="a")
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

            # Compute patch grid on first sample
            if patch_grid is None:
                patch_grid = compute_patch_grid(
                    batch.metadata.lat,
                    batch.metadata.lon,
                    model.patch_size,
                )

            # Run forward pass
            batch_device = batch.to(device)
            try:
                with torch.no_grad():
                    prediction = model(batch_device).to("cpu")
            except Exception as exc:
                print(f"    ✗ Forward pass failed: {exc}")
                failed_samples += 1
                captures.clear()
                continue

            # Extract captured latents
            deagg_latents = captures.get("decoder.deaggregated_atmospheric_latents")
            surface_latents = captures.get("decoder.surface_latents")
            
            if deagg_latents is None or surface_latents is None:
                print("    ✗ Latent captures are missing")
                failed_samples += 1
                captures.clear()
                continue

            deagg_latents = deagg_latents.detach().cpu()
            surface_latents = surface_latents.detach().cpu()

            # Prepare region-specific latents
            try:
                (
                    surf_region_latents,
                    _,
                    _,
                    _,
                    row_start,
                    row_end,
                    col_start,
                    col_end,
                ) = prepare_region_for_capture(
                    mode="surface",
                    requested_bounds=requested_bounds,
                    patch_grid=patch_grid,
                    surface_latents=surface_latents,
                    atmos_latents=None,
                )
                
                (
                    atmos_region_latents,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = prepare_region_for_capture(
                    mode="atmos",
                    requested_bounds=requested_bounds,
                    patch_grid=patch_grid,
                    surface_latents=None,
                    atmos_latents=deagg_latents,
                )
            except Exception as exc:
                print(f"    ✗ Region extraction failed: {exc}")
                failed_samples += 1
                captures.clear()
                continue

            # Initialize Zarr on first successful sample
            if zarr_store is None:
                # Get shapes from first sample
                # Surface: (1, patches, channels) -> reshape to (rows, cols, channels)
                patch_rows = row_end - row_start + 1
                patch_cols = col_end - col_start + 1
                
                # surf_region_latents shape: (1, n_patches, channels) where n_patches = rows*cols
                surf_np = surf_region_latents.numpy()
                s_channels = surf_np.shape[-1]
                
                # atmos_region_latents shape: (1, levels, n_patches, channels)
                atmos_np = atmos_region_latents.numpy()
                p_levels = atmos_np.shape[1]
                p_channels = atmos_np.shape[-1]
                
                # Get patch centres for the region
                centres = patch_grid["centres"]
                patch_shape = patch_grid["patch_shape"]
                all_indices = np.arange(centres.shape[0])
                lat_ids, lon_ids = np.unravel_index(all_indices, patch_shape)
                mask = (
                    (lat_ids >= row_start)
                    & (lat_ids <= row_end)
                    & (lon_ids >= col_start)
                    & (lon_ids <= col_end)
                )
                region_centres = centres[mask]
                
                region_info = {
                    "row_start": row_start,
                    "row_end": row_end,
                    "col_start": col_start,
                    "col_end": col_end,
                    "patch_shape": patch_shape,
                    "centres": region_centres,
                }
                
                zarr_store = initialize_zarr_store(
                    output_zarr,
                    n_timesteps=n_timesteps,
                    surface_shape=(patch_rows, patch_cols, s_channels),
                    pressure_shape=(p_levels, patch_rows, patch_cols, p_channels),
                    region_bounds=requested_bounds,
                    patch_grid=region_info,
                    atmos_levels=batch.metadata.atmos_levels,
                    timesteps=all_timesteps,
                )

            # Reshape and write to Zarr
            try:
                # Reshape surface: (1, rows*cols, channels) -> (rows, cols, channels)
                surf_np = surf_region_latents.numpy()
                surf_reshaped = surf_np[0].reshape(patch_rows, patch_cols, -1)
                zarr_store["surface_latents"][time_idx] = surf_reshaped.astype(np.float32)
                
                # Reshape pressure: (1, levels, rows*cols, channels) -> (levels, rows, cols, channels)
                atmos_np = atmos_region_latents.numpy()
                n_levels = atmos_np.shape[1]
                atmos_reshaped = atmos_np[0].reshape(n_levels, patch_rows, patch_cols, -1)
                zarr_store["pressure_latents"][time_idx] = atmos_reshaped.astype(np.float32)
                
                # Mark as processed
                zarr_store["processed"][time_idx] = True
                
                print(f"    ✓ Saved timestep {time_idx}/{n_timesteps}")
                processed_samples += 1
                
            except Exception as exc:
                print(f"    ✗ Failed to write to Zarr: {exc}")
                failed_samples += 1
            
            captures.clear()

    finally:
        # Clean up hooks
        for handle in handles:
            handle.remove()
        decoder_cleanup()
        
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
    print(f"  Output: {output_zarr}")
    print(f"{'='*60}")
    
    # Print example usage
    print(f"""
Example usage with xarray:

    import xarray as xr
    ds = xr.open_zarr("{output_zarr}")
    
    # Access patch center coordinates
    lat_centres = ds.lat.values  # 1D array of patch center latitudes
    lon_centres = ds.lon.values  # 1D array of patch center longitudes
    levels = ds.level.values     # Pressure levels [50, 100, ..., 1000]
    
    # Get first timestep
    surface = ds.surface_latents.isel(time=0).values  # (lat, lon, channels)
    pressure = ds.pressure_latents.isel(time=0).values  # (level, lat, lon, channels)
    
    # Get by datetime
    surface = ds.surface_latents.sel(time="2018-01-01T00:00:00").values
    
    # Get by coordinate (nearest neighbor)
    surface = ds.surface_latents.sel(lat=55.0, lon=10.0, method="nearest")
    
    # DataLoader usage
    n_samples = len(ds.time)
    for idx in range(n_samples):
        surface = ds.surface_latents.isel(time=idx).values
        pressure = ds.pressure_latents.isel(time=idx).values
        # lat_centres and lon_centres give you the position of each latent
""")


if __name__ == "__main__":
    main()

# Usage examples:
# 
# Process ALL timesteps (00, 06, 12, 18 UTC) for years 2018-2020:
#   python examples/run_year_forward_with_latents_zarr.py \
#       --start-year 2018 --end-year 2020 \
#       --output-zarr examples/latents_europe_2018_2020.zarr
#
# Test with a few samples:
#   python examples/run_year_forward_with_latents_zarr.py \
#       --start-year 2018 --end-year 2018 \
#       --max-samples 10
#
# Resume interrupted processing:
#   python examples/run_year_forward_with_latents_zarr.py \
#       --start-year 2018 --end-year 2020 \
#       --skip-existing
