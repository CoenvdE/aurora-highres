#!/usr/bin/env python3
"""Merge multiple yearly HRES Zarr files into a single dataset.

This script opens yearly Zarr files lazily (minimal memory) and 
concatenates them along the time dimension.

Usage:
    python examples/merge_hres_zarr.py \
        --input-dir /projects/prjs1858 \
        --years 2018 2019 2020 \
        --output /projects/prjs1858/hres_europe_2018_2020.zarr
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge yearly HRES Zarr files into a single dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/projects/prjs1858"),
        help="Directory containing yearly Zarr files",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2018, 2019, 2020],
        help="Years to merge (e.g., 2018 2019 2020)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/projects/prjs1858/hres_europe_2018_2020.zarr"),
        help="Output merged Zarr path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Check if output exists
    if args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}")
        print("Use --overwrite to replace it")
        return
    
    # Find and validate input files
    input_files = []
    for year in sorted(args.years):
        zarr_path = args.input_dir / f"hres_europe_{year}.zarr"
        if not zarr_path.exists():
            raise FileNotFoundError(f"Missing yearly Zarr file: {zarr_path}")
        input_files.append(zarr_path)
    
    print(f"Merging {len(input_files)} yearly files:")
    for f in input_files:
        print(f"  - {f}")
    
    # Open all datasets lazily
    print("\nOpening datasets lazily...")
    datasets = []
    for path in input_files:
        ds = xr.open_zarr(str(path))
        print(f"  {path.name}: {len(ds.time)} timesteps")
        datasets.append(ds)
    
    # Concatenate along time dimension
    print("\nConcatenating along time dimension...")
    merged = xr.concat(datasets, dim="time")
    
    # Update metadata
    merged.attrs["years"] = str(sorted(args.years))
    merged.attrs["merged_from"] = [str(f) for f in input_files]
    merged.attrs["created"] = datetime.now().isoformat()
    merged.attrs["description"] = f"HRES analysis data for {min(args.years)}-{max(args.years)}"
    
    # Remove single-year metadata
    if "year" in merged.attrs:
        del merged.attrs["year"]
    
    print(f"\nMerged dataset:")
    print(f"  Time range: {merged.time.values[0]} to {merged.time.values[-1]}")
    print(f"  Total timesteps: {len(merged.time)}")
    print(f"  Variables: {list(merged.data_vars)}")
    
    # Set up encoding for efficient chunked access
    encoding = {}
    for var in merged.data_vars:
        encoding[var] = {
            "dtype": "float32",
            "chunks": tuple(
                1 if dim == "time" else merged[var].sizes[dim]
                for dim in merged[var].dims
            ),
        }
    
    # Save to Zarr
    print(f"\nSaving to {args.output}...")
    mode = "w" if args.overwrite or not args.output.exists() else None
    if mode:
        merged.to_zarr(str(args.output), mode=mode, encoding=encoding)
    else:
        merged.to_zarr(str(args.output), encoding=encoding)
    
    print(f"\n{'='*60}")
    print(f"✓ Merge complete!")
    print(f"  Output: {args.output}")
    print(f"  Years: {sorted(args.years)}")
    print(f"  Total timesteps: {len(merged.time)}")
    print(f"{'='*60}")
    
    # Print usage example
    print(f"""
Example usage:

    import xarray as xr
    ds = xr.open_zarr("{args.output}")
    
    # Get data
    print(ds)
    t2m = ds["2t"].isel(time=0).values
""")


if __name__ == "__main__":
    main()
