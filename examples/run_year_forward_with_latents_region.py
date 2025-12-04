#!/usr/bin/env python3
"""Process multiple years of ERA5 samples and store region-specific Aurora latents.

This script combines the region-based latent capture from 
`run_single_forward_with_latents_region.py` with the multi-year capability 
from `run_year_forward_with_latents.py`, with improved error handling and 
progress tracking.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import h5py
import torch
import xarray as xr

from examples.extract_latents import register_latent_hooks
from examples.init_exploring.load_era_batch_snellius import load_batch_from_zarr
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

# Default region as in `run_single_forward_with_latents_region.py` (roughly Europe)
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -30.0
DEFAULT_LON_MAX = 50.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Aurora forward passes for multiple years while capturing "
            "region-specific latents efficiently."
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
        "--work-dir",
        type=Path,
        default=Path("examples/run_year_forward_latents_region"),
        help="Root directory for downloads, tensors, and latents",
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
        help="Skip samples whose latents dataset already exists",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after processing at most this many time steps",
    )
    return parser.parse_args()


def iterate_days(year: int):
    """Generate all days in a given year."""
    current = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    delta = timedelta(days=1)
    while current <= end:
        yield current
        current += delta


def latent_exists(latents_h5: Path, label: str) -> bool:
    """Check if a specific timestamp's latents already exist in the H5 file."""
    if not latents_h5.exists():
        return False

    try:
        with h5py.File(latents_h5, "r") as handle:
            latents_group = handle.get("pressure_latents_region")
            if latents_group is None:
                return False
            return label in latents_group
    except Exception:
        return False


def save_patch_grid_file(
    destination: Path,
    patch_grid: dict[str, torch.Tensor | int | Tuple[int, int]],
) -> None:
    """Save the patch grid to a .pt file for later use."""
    if destination.exists():
        return

    serialisable: dict[str, torch.Tensor | int | Tuple[int, int]] = {}
    for key, value in patch_grid.items():
        if isinstance(value, torch.Tensor):
            serialisable[key] = value.detach().cpu()
        else:
            serialisable[key] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialisable, destination)
    print(f"✓ Patch grid saved to {destination}")


def save_region_bounds_once(
    latents_h5: Path,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> None:
    """Save the region bounds to the H5 file attributes (once per file)."""
    with h5py.File(latents_h5, "a") as handle:
        # Only set these if they haven't been set yet
        if "row_start" not in handle.attrs:
            handle.attrs["row_start"] = row_start
            handle.attrs["row_end"] = row_end
            handle.attrs["col_start"] = col_start
            handle.attrs["col_end"] = col_end
            print(
                f"✓ Region bounds saved: rows [{row_start}:{row_end}], "
                f"cols [{col_start}:{col_end}]"
            )


def main() -> None:
    args = parse_args()
    
    # Validate year range
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    work_dir = args.work_dir.expanduser()
    latents_dir = work_dir / "latents_dataset"
    latents_dir.mkdir(parents=True, exist_ok=True)

    static_path = ensure_static_dataset(work_dir) #TODO: check this

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model once
    model = load_model(device)
    model.eval()

    patch_grid: dict[str, torch.Tensor | int | Tuple[int, int]] | None = None
    patch_grid_cache_path = latents_dir / "patch_grid.pt"

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

    # Keep track of region indices (extracted from first sample)
    region_indices: Tuple[int, int, int, int] | None = None

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

        for year in range(args.start_year, args.end_year + 1):
            print(f"\n{'='*60}")
            print(f"Processing year {year}")
            print(f"{'='*60}")
            
            year_latents_dir = latents_dir / str(year)
            year_latents_dir.mkdir(parents=True, exist_ok=True)

            latents_h5 = year_latents_dir / "pressure_surface_latents_region.h5"

            for day in iterate_days(year):
                date_str = day.strftime("%Y-%m-%d")
                expected_time = datetime(day.year, day.month, day.day, 12)
                expected_label = expected_time.strftime("%Y-%m-%dT%H-%M-%S")

                # Skip if already processed
                if args.skip_existing and latent_exists(latents_h5, expected_label):
                    print(f"  ⊘ Skipping {date_str} (already exists)")
                    skipped_samples += 1
                    continue

                print(f"  ➤ Processing {date_str}...")
                
                try:
                    batch = load_batch_from_zarr(
                        zarr_path=args.zarr_path,
                        static_path=str(static_path),
                        date_str=date_str, #TODO: fix that it loads 2 dates before this and it predicts this date
                        dataset=dataset,
                        static_dataset=static_dataset,
                    )
                except Exception as exc:
                    print(f"    ✗ Failed to load batch: {exc}")
                    failed_samples += 1
                    continue

                # Compute patch grid once
                if patch_grid is None:
                    patch_grid = compute_patch_grid(
                        batch.metadata.lat,
                        batch.metadata.lon,
                        model.patch_size,
                    )
                    save_patch_grid_file(patch_grid_cache_path, patch_grid)

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

                actual_time = prediction.metadata.time[0]
                timestamp_label = actual_time.strftime("%Y-%m-%dT%H-%M-%S") #TODO: look at best way to save this to index later with dataset

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

                # Save region bounds once per file
                if region_indices is None:
                    region_indices = (row_start, row_end, col_start, col_end)
                    save_region_bounds_once( #TODO: check this
                        latents_h5, row_start, row_end, col_start, col_end
                    )

                # Write latents to H5 file
                try:
                    with h5py.File(latents_h5, "a") as handle:
                        # Atmospheric latents
                        levels_group = handle.require_group("pressure_latents_region")
                        if timestamp_label in levels_group:
                            del levels_group[timestamp_label]
                        pressure_dataset = levels_group.create_dataset(
                            timestamp_label,
                            data=atmos_region_latents.numpy(),
                            compression="gzip",
                        )
                        pressure_dataset.attrs["timestamp"] = actual_time.isoformat()
                        pressure_dataset.attrs["shape"] = atmos_region_latents.shape

                        # Surface latents
                        surface_group = handle.require_group("surface_latents_region")
                        if timestamp_label in surface_group:
                            del surface_group[timestamp_label]
                        surface_dataset = surface_group.create_dataset(
                            timestamp_label,
                            data=surf_region_latents.numpy(),
                            compression="gzip",
                        )
                        surface_dataset.attrs["timestamp"] = actual_time.isoformat()
                        surface_dataset.attrs["shape"] = surf_region_latents.shape

                    print(f"    ✓ Saved {timestamp_label} to {latents_h5.name}")
                    processed_samples += 1
                    
                except Exception as exc:
                    print(f"    ✗ Failed to write to H5: {exc}")
                    failed_samples += 1
                
                captures.clear()

                # Check if we've reached max samples
                if args.max_samples is not None and processed_samples >= args.max_samples:
                    print("\nReached max sample limit, stopping early.")
                    break
            
            # Break outer loop if max samples reached
            if args.max_samples is not None and processed_samples >= args.max_samples:
                break

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
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# Usage examples:
# 
# Process years 2018-2020 with default region (Europe):
#   python examples/run_year_forward_with_latents_region.py \
#       --start-year 2018 --end-year 2020
#
# Process with custom region (North America):
#   python examples/run_year_forward_with_latents_region.py \
#       --start-year 2018 --end-year 2020 \
#       --lat-min 25.0 --lat-max 50.0 \
#       --lon-min -130.0 --lon-max -60.0
#
# Process with limited samples for testing:
#   python examples/run_year_forward_with_latents_region.py \
#       --start-year 2018 --end-year 2018 \
#       --max-samples 5 --skip-existing
