#!/usr/bin/env python3
"""Rechunk Zarr datasets with optimized chunks for ML DataLoader access.

This script rechunks existing Zarr v2 datasets to use smaller temporal chunks,
which improves random access performance during training.

Optimized for:
- Random timestep access (DataLoader with shuffle=True)
- Prefetching nearby timesteps (temporal locality)
- Reduced I/O per sample

Usage:
    # Rechunk HRES dataset
    python examples/rechunk_zarr_v2.py \
        --input /projects/prjs1858/hres_europe_2018_2020.zarr \
        --output /projects/prjs1858/hres_europe_2018_2020_v2opt.zarr \
        --time-chunk 8

    # Rechunk latents dataset  
    python examples/rechunk_zarr_v2.py \
        --input /projects/prjs1858/latents_europe_2018_2020.zarr \
        --output /projects/prjs1858/latents_europe_2018_2020_v2opt.zarr \
        --time-chunk 8

    # Rechunk static dataset (no time dimension)
    python examples/rechunk_zarr_v2.py \
        --input /projects/prjs1858/static_hres_europe.zarr \
        --output /projects/prjs1858/static_hres_europe_v2opt.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import xarray as xr
import zarr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rechunk Zarr datasets for optimized ML training access."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input Zarr dataset path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output Zarr dataset path",
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        default=8,
        help="Number of timesteps per chunk (default: 8). Use 1 for static datasets.",
    )
    parser.add_argument(
        "--spatial-chunk",
        type=int,
        default=None,
        help="Optional: chunk size for spatial dims (default: keep full spatial extent)",
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="blosc",
        choices=["blosc", "zstd", "lz4", "none"],
        help="Compression algorithm (default: blosc)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    return parser.parse_args()


def get_compressor(name: str):
    """Get numcodecs compressor by name."""
    import numcodecs
    
    if name == "blosc":
        return numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
    elif name == "zstd":
        return numcodecs.Zstd(level=3)
    elif name == "lz4":
        return numcodecs.LZ4(acceleration=1)
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown compressor: {name}")


def compute_optimal_chunks(ds: xr.Dataset, time_chunk: int, spatial_chunk: int | None) -> dict:
    """Compute optimal chunk sizes for each variable."""
    encoding = {}
    
    for var in ds.data_vars:
        da = ds[var]
        chunks = {}
        
        for dim in da.dims:
            dim_size = da.sizes[dim]
            
            if dim == "time":
                # Use specified time chunk size
                chunks[dim] = min(time_chunk, dim_size)
            elif dim in ("lat", "latitude", "lon", "longitude", "level", "channel"):
                # Keep full extent for these dims (or use spatial_chunk if specified)
                if spatial_chunk and dim in ("lat", "latitude", "lon", "longitude"):
                    chunks[dim] = min(spatial_chunk, dim_size)
                else:
                    chunks[dim] = dim_size
            else:
                # Unknown dimension - keep full
                chunks[dim] = dim_size
        
        encoding[var] = {"chunks": tuple(chunks[d] for d in da.dims)}
    
    return encoding


def main() -> None:
    args = parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")
    
    if args.output.exists() and not args.overwrite:
        print(f"Output already exists: {args.output}")
        print("Use --overwrite to replace it")
        return
    
    print(f"{'='*60}")
    print(f"Rechunking Zarr Dataset (v2 Optimized)")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Time chunk size: {args.time_chunk}")
    print(f"Compressor: {args.compressor}")
    print()
    
    # Open input dataset
    print("Opening input dataset...")
    ds = xr.open_zarr(str(args.input), consolidated=True)
    
    print(f"\nInput dataset info:")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    if "time" in ds.dims:
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Get original chunks for comparison
    print(f"\nOriginal chunks:")
    for var in ds.data_vars:
        if hasattr(ds[var], 'encoding') and 'chunks' in ds[var].encoding:
            print(f"  {var}: {ds[var].encoding.get('chunks', 'unknown')}")
        else:
            print(f"  {var}: (from zarr metadata)")
    
    # Compute new chunk sizes
    encoding = compute_optimal_chunks(ds, args.time_chunk, args.spatial_chunk)
    
    # Add compressor and dtype to encoding
    compressor = get_compressor(args.compressor)
    for var in encoding:
        encoding[var]["compressor"] = compressor
        encoding[var]["dtype"] = "float32"
    
    # Add coordinate encoding
    for coord in ds.coords:
        if coord == "time":
            encoding[coord] = {
                "dtype": "float64",
                "units": "hours since 1970-01-01T00:00:00",
                "calendar": "proleptic_gregorian",
            }
        elif coord in ("level",):
            encoding[coord] = {"dtype": "int32"}
        elif coord in ("channel",):
            encoding[coord] = {"dtype": "int32"}
    
    print(f"\nNew chunks:")
    for var in ds.data_vars:
        print(f"  {var}: {encoding[var]['chunks']}")
    
    # Calculate estimated sizes
    print(f"\nEstimated chunk sizes:")
    for var in ds.data_vars:
        chunk_shape = encoding[var]["chunks"]
        chunk_size_bytes = 4  # float32
        for s in chunk_shape:
            chunk_size_bytes *= s
        chunk_size_mb = chunk_size_bytes / (1024 * 1024)
        print(f"  {var}: {chunk_shape} = {chunk_size_mb:.2f} MB per chunk")
    
    # Update attributes
    ds.attrs["rechunked_from"] = str(args.input)
    ds.attrs["rechunk_time"] = datetime.now().isoformat()
    ds.attrs["time_chunk_size"] = args.time_chunk
    ds.attrs["compressor"] = args.compressor
    
    # Write to new Zarr store
    print(f"\nWriting to {args.output}...")
    print("This may take a while for large datasets...")
    
    # Use compute=True to force immediate computation (no lazy writing)
    ds.to_zarr(
        str(args.output),
        mode="w" if args.overwrite else None,
        encoding=encoding,
        consolidated=True,
    )
    
    # Verify output
    print(f"\nVerifying output...")
    ds_out = xr.open_zarr(str(args.output), consolidated=True)
    
    print(f"\n{'='*60}")
    print(f"✓ Rechunking complete!")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"Variables: {list(ds_out.data_vars)}")
    if "time" in ds_out.dims:
        print(f"Timesteps: {len(ds_out.time)}")
    
    # Print actual chunks from output
    print(f"\nActual output chunks:")
    store = zarr.open(str(args.output), mode="r")
    for var in ds_out.data_vars:
        if var in store:
            print(f"  {var}: {store[var].chunks}")
    
    print(f"""
Usage in training:
    # Update your config to use the new dataset:
    latent_zarr_path: {args.output}
    
    # The new chunks are optimized for:
    # - Random timestep access (DataLoader shuffle)
    # - Prefetching nearby timesteps
    # - Reduced I/O overhead per sample
""")


if __name__ == "__main__":
    main()
