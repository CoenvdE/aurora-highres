#!/usr/bin/env python3
"""Convert Zarr v2 datasets to Zarr v3 with sharding for optimal ML access.

Zarr v3 sharding groups multiple chunks into single files ("shards"), which:
- Reduces file count (thousands of chunks → few shard files)
- Improves random access via chunk indexing within shards
- Better filesystem performance on HPC storage

IMPORTANT: This requires zarr>=3.0.0 and uses the new zarr-python v3 API.
xarray support for zarr v3 is experimental - test before production use.

Usage:
    # Convert HRES dataset to v3 with sharding
    python examples/convert_zarr_v3_sharded.py \
        --input /projects/prjs1858/hres_europe_2018_2020.zarr \
        --output /projects/prjs1858/hres_europe_2018_2020_v3.zarr \
        --shard-time 64

    # Convert latents dataset
    python examples/convert_zarr_v3_sharded.py \
        --input /projects/prjs1858/latents_europe_2018_2020.zarr \
        --output /projects/prjs1858/latents_europe_2018_2020_v3.zarr \
        --shard-time 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Zarr v2 to v3 with sharding for optimized ML access."
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
        "--shard-time",
        type=int,
        default=64,
        help="Number of timesteps per shard (default: 64). Inner chunk is 1 timestep.",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=1,
        help="Inner chunk size for time dimension (default: 1 for random access)",
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="blosc",
        choices=["blosc", "zstd", "gzip", "none"],
        help="Compression codec (default: blosc)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )
    return parser.parse_args()


def check_zarr_version():
    """Check that zarr v3 is available."""
    import zarr
    version = tuple(int(x) for x in zarr.__version__.split(".")[:2])
    if version < (3, 0):
        raise RuntimeError(
            f"This script requires zarr>=3.0.0, but found {zarr.__version__}. "
            f"Install with: pip install 'zarr>=3.0.0'"
        )
    print(f"Using zarr {zarr.__version__}")


def get_codec_pipeline(compressor: str):
    """Get zarr v3 codec pipeline."""
    from zarr.codecs import BytesCodec, BloscCodec, ZstdCodec, GzipCodec
    
    codecs = [BytesCodec()]
    
    if compressor == "blosc":
        codecs.append(BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle"))
    elif compressor == "zstd":
        codecs.append(ZstdCodec(level=3))
    elif compressor == "gzip":
        codecs.append(GzipCodec(level=5))
    # "none" = no compression codec added
    
    return codecs


def get_sharding_codec(inner_chunks: tuple, codecs: list):
    """Create sharding codec that wraps inner chunks."""
    from zarr.codecs import ShardingCodec
    
    return ShardingCodec(
        chunk_shape=inner_chunks,
        codecs=codecs,
    )


def main() -> None:
    args = parse_args()
    
    # Check zarr version first
    check_zarr_version()
    
    import zarr
    from zarr import Array, Group
    import xarray as xr
    
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
    print(f"Converting to Zarr v3 with Sharding")
    print(f"{'='*60}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Shard size (time): {args.shard_time}")
    print(f"Inner chunk (time): {args.chunk_time}")
    print(f"Compressor: {args.compressor}")
    print()
    
    # Open input dataset with xarray (v2 format)
    print("Opening input dataset...")
    ds = xr.open_zarr(str(args.input), consolidated=True)
    
    print(f"\nInput dataset info:")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"  Dimensions: {dict(ds.dims)}")
    if "time" in ds.dims:
        print(f"  Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        print(f"  Timesteps: {len(ds.time)}")
    
    # Create output store (zarr v3)
    print(f"\nCreating Zarr v3 store...")
    store = zarr.open_group(
        str(args.output),
        mode="w",
        zarr_format=3,
    )
    
    # Copy attributes
    for key, value in ds.attrs.items():
        try:
            store.attrs[key] = value
        except Exception:
            store.attrs[key] = str(value)
    
    store.attrs["zarr_format"] = 3
    store.attrs["converted_from"] = str(args.input)
    store.attrs["conversion_time"] = datetime.now().isoformat()
    store.attrs["shard_time"] = args.shard_time
    store.attrs["chunk_time"] = args.chunk_time
    
    # Get compressor codecs
    inner_codecs = get_codec_pipeline(args.compressor)
    
    # Process each variable
    for var_name in ds.data_vars:
        print(f"\nProcessing variable: {var_name}")
        da = ds[var_name]
        
        # Compute chunk and shard shapes
        inner_chunks = []
        shard_shape = []
        
        for dim in da.dims:
            dim_size = da.sizes[dim]
            
            if dim == "time":
                # Time dim: inner chunk = 1, shard = 64 (or configured)
                inner_chunk = min(args.chunk_time, dim_size)
                shard = min(args.shard_time, dim_size)
            else:
                # Other dims: keep full extent in both inner chunk and shard
                inner_chunk = dim_size
                shard = dim_size
            
            inner_chunks.append(inner_chunk)
            shard_shape.append(shard)
        
        inner_chunks = tuple(inner_chunks)
        shard_shape = tuple(shard_shape)
        
        print(f"  Shape: {da.shape}")
        print(f"  Inner chunks: {inner_chunks}")
        print(f"  Shard shape: {shard_shape}")
        
        # Calculate sizes
        inner_size_mb = np.prod(inner_chunks) * 4 / (1024 * 1024)
        shard_size_mb = np.prod(shard_shape) * 4 / (1024 * 1024)
        print(f"  Inner chunk size: {inner_size_mb:.2f} MB")
        print(f"  Shard size: {shard_size_mb:.2f} MB")
        
        # Create sharding codec
        sharding_codec = get_sharding_codec(inner_chunks, inner_codecs)
        
        # Create array with sharding
        arr = store.create_array(
            var_name,
            shape=da.shape,
            chunks=shard_shape,  # Shard shape = outer chunks
            dtype=np.float32,
            codecs=[sharding_codec],
            fill_value=np.nan,
        )
        
        # Set dimension metadata for xarray compatibility
        arr.attrs["_ARRAY_DIMENSIONS"] = list(da.dims)
        
        # Copy data in chunks to avoid memory issues
        print(f"  Writing data...")
        if "time" in da.dims:
            time_idx = da.dims.index("time")
            n_times = da.sizes["time"]
            batch_size = args.shard_time
            
            for start in range(0, n_times, batch_size):
                end = min(start + batch_size, n_times)
                print(f"    Timesteps {start}-{end}...")
                
                # Build slice for this batch
                slices = [slice(None)] * len(da.dims)
                slices[time_idx] = slice(start, end)
                
                # Load and write
                data = da.isel(time=slice(start, end)).values.astype(np.float32)
                arr[tuple(slices)] = data
        else:
            # No time dimension - write all at once
            arr[:] = da.values.astype(np.float32)
        
        print(f"  ✓ Done: {var_name}")
    
    # Copy coordinates
    print(f"\nCopying coordinates...")
    for coord_name in ds.coords:
        print(f"  {coord_name}")
        coord = ds[coord_name]
        data = coord.values
        
        # Determine dtype
        if coord_name == "time":
            dtype = "datetime64[ns]"
        elif np.issubdtype(data.dtype, np.integer):
            dtype = "int32"
        else:
            dtype = "float32"
        
        # Create coordinate array (no sharding for small coords)
        arr = store.create_array(
            coord_name,
            shape=data.shape,
            chunks=data.shape,  # Single chunk for coordinates
            dtype=dtype,
            codecs=inner_codecs,
        )
        arr[:] = data.astype(dtype)
        arr.attrs["_ARRAY_DIMENSIONS"] = [coord_name] if len(data.shape) == 1 else list(coord.dims)
    
    print(f"\n{'='*60}")
    print(f"✓ Conversion complete!")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"Format: Zarr v3 with sharding")
    print(f"Shards contain {args.shard_time} timesteps each")
    print(f"Inner chunks of {args.chunk_time} timestep(s) for random access")
    
    print(f"""
IMPORTANT: Zarr v3 with xarray
==============================
xarray support for zarr v3 is experimental. To load:

    import xarray as xr
    
    # Method 1: Direct zarr v3 (if xarray supports it)
    ds = xr.open_zarr("{args.output}", zarr_format=3)
    
    # Method 2: Using zarr directly
    import zarr
    store = zarr.open_group("{args.output}", mode="r")
    data = store["surface_latents"][0]  # Access by index
    
If xarray doesn't work with v3, use the v2 optimized version instead.
""")


if __name__ == "__main__":
    main()
