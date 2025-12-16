#!/usr/bin/env python3
"""Convert Zarr v2 datasets to Zarr v3 using xarray.

This is the simple approach - xarray handles everything:
- Preserves dimension names and metadata
- Automatically handles coordinates
- Full xarray compatibility

Usage:
    python examples/convert_zarr_v3_xarray.py \
        --input /projects/prjs1858/hres_europe_2018_2020.zarr \
        --output /projects/prjs1858/hres_europe_2018_2020_v3.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zarr v2 to v3 using xarray (preserves all metadata)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input Zarr v2 dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Zarr v3 dataset path",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Check zarr version
    import zarr
    from zarr.codecs import BloscCodec
    
    print(f"Using zarr {zarr.__version__}")
    print(f"Using xarray {xr.__version__}")
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    
    if args.output.exists():
        if args.overwrite:
            import shutil
            print(f"Removing existing output: {args.output}")
            shutil.rmtree(args.output)
        else:
            print(f"Output already exists: {args.output}")
            print("Use --overwrite to replace it")
            return
    
    print(f"{'='*60}")
    print(f"Converting to Zarr v3 using xarray")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()
    
    # Open input dataset
    print("Opening input dataset...")
    ds = xr.open_zarr(str(args.input), consolidated=True)
    
    print(f"\nDataset info:")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.sizes)}")
    if "time" in ds.dims:
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        print(f"  Timesteps: {len(ds.time)}")
    
    # Build encoding to override the old numcodecs compressor with zarr v3 codec
    # Use zarr.codecs.BloscCodec instead of numcodecs.blosc.Blosc
    compressor = BloscCodec(cname="lz4", clevel=5)
    encoding = {}
    for var in list(ds.data_vars) + list(ds.coords):
        encoding[var] = {"compressor": compressor}
    
    # Convert to v3 - xarray handles everything!
    print("\nWriting to Zarr v3...")
    ds.to_zarr(
        str(args.output),
        zarr_format=3,
        mode="w",
        encoding=encoding,
    )
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    
    # Verify it can be loaded
    print("\nVerifying output...")
    ds_v3 = xr.open_zarr(str(args.output), zarr_format=3)
    print(f"  ✓ xarray can load the v3 dataset")
    print(f"  Variables: {list(ds_v3.data_vars)}")
    print(f"  Dimensions: {dict(ds_v3.sizes)}")


if __name__ == "__main__":
    main()
