#!/usr/bin/env python3
"""Inspect Zarr datasets and print metadata (variables, coordinates, shapes, chunks)."""

import argparse
import xarray as xr
import zarr


def format_size(size_bytes: int) -> str:
    """Format bytes into human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def inspect_dataset(path: str, zarr_format: int = 2) -> None:
    """Inspect a Zarr dataset and print its metadata."""
    print(f"\n{'='*60}")
    print(f"Dataset: {path}")
    print(f"{'='*60}")
    
    # Open with xarray
    try:
        if zarr_format == 3:
            ds = xr.open_zarr(path, zarr_format=3)
        else:
            ds = xr.open_zarr(path)
    except Exception as e:
        print(f"Error opening dataset: {e}")
        return
    
    # Coordinates
    print(f"\n--- Coordinates ({len(ds.coords)}) ---")
    for name, coord in ds.coords.items():
        chunks = coord.encoding.get('chunks', 'N/A')
        print(f"  {name}:")
        print(f"    dtype: {coord.dtype}")
        print(f"    shape: {coord.shape}")
        print(f"    chunks: {chunks}")
    
    # Data variables
    print(f"\n--- Data Variables ({len(ds.data_vars)}) ---")
    for name, var in ds.data_vars.items():
        chunks = var.encoding.get('chunks', 'N/A')
        # Calculate approx size
        size_bytes = var.size * var.dtype.itemsize
        print(f"  {name}:")
        print(f"    dims: {var.dims}")
        print(f"    dtype: {var.dtype}")
        print(f"    shape: {var.shape}")
        print(f"    chunks: {chunks}")
        print(f"    size: {format_size(size_bytes)}")
    
    # Attributes
    print(f"\n--- Attributes ({len(ds.attrs)}) ---")
    for key, val in ds.attrs.items():
        val_str = str(val)
        if len(val_str) > 60:
            val_str = val_str[:57] + "..."
        print(f"  {key}: {val_str}")
    
    # Total dataset summary
    total_size = sum(var.size * var.dtype.itemsize for var in ds.data_vars.values())
    print(f"\n--- Summary ---")
    print(f"  Total data variables: {len(ds.data_vars)}")
    print(f"  Total coordinates: {len(ds.coords)}")
    print(f"  Estimated uncompressed size: {format_size(total_size)}")
    
    ds.close()


def main():
    parser = argparse.ArgumentParser(description="Inspect Zarr datasets")
    parser.add_argument("--data-dir", default="/projects/prjs1858",
                        help="Base directory for datasets")
    parser.add_argument("--zarr-format", type=int, default=2, choices=[2, 3],
                        help="Zarr format version (default: 2)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to inspect (default: all)")
    args = parser.parse_args()
    
    # Default datasets
    if args.zarr_format == 3:
        default_datasets = [
            f"{args.data_dir}/hres_europe_2018_2020_v3.zarr",
            f"{args.data_dir}/latents_europe_2018_2020_v3.zarr",
            f"{args.data_dir}/static_hres_europe_v3.zarr",
            f"{args.data_dir}/static_hres_europe_v3_rechunked.zarr",
        ]
    else:
        default_datasets = [
            f"{args.data_dir}/hres_europe_2018_2020.zarr",
            f"{args.data_dir}/latents_europe_2018_2020.zarr",
            f"{args.data_dir}/static_hres_europe.zarr",
        ]
    
    datasets = args.datasets if args.datasets else default_datasets
    
    print(f"Inspecting Zarr v{args.zarr_format} datasets...")
    
    for path in datasets:
        inspect_dataset(path, args.zarr_format)
    
    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
