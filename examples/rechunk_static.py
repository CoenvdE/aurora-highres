#!/usr/bin/env python3
"""Rechunk the static Zarr v3 dataset to use full spatial extent as a single chunk.

The static dataset is small (7.4 MB) so we can load it all at once.
This improves read performance by eliminating chunk boundary overhead.
"""

import argparse
import shutil
from pathlib import Path

import xarray as xr
from zarr.codecs import BloscCodec


def main():
    parser = argparse.ArgumentParser(description="Rechunk static dataset")
    parser.add_argument("--data-dir", default="/projects/prjs1858")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    input_path = Path(args.data_dir) / "static_hres_europe_v3.zarr"
    output_path = Path(args.data_dir) / "static_hres_europe_v3_rechunked.zarr"
    
    if output_path.exists():
        if args.overwrite:
            print(f"Removing existing: {output_path}")
            shutil.rmtree(output_path)
        else:
            print(f"Output exists: {output_path}")
            print("Use --overwrite to replace")
            return
    
    print(f"Opening: {input_path}")
    ds = xr.open_zarr(str(input_path), zarr_format=3)
    
    print(f"\nOriginal chunks:")
    for var in ds.data_vars:
        print(f"  {var}: {ds[var].encoding.get('chunks')}")
    
    # Load into memory (only 7 MB, so this is fine)
    # This avoids Dask chunk alignment issues when rechunking
    print("\nLoading into memory (small dataset)...")
    ds = ds.load()
    
    # Get full spatial dimensions
    lat_size = ds.sizes["latitude"]
    lon_size = ds.sizes["longitude"]
    
    print(f"\nSpatial dimensions: ({lat_size}, {lon_size})")
    print(f"Setting chunk size to full extent: ({lat_size}, {lon_size})")
    
    # Build encoding with full-extent chunks
    compressor = BloscCodec(cname="lz4", clevel=5)
    encoding = {}
    
    for var in ds.data_vars:
        encoding[var] = {
            "chunks": (lat_size, lon_size),  # Full extent = single chunk per variable
            "compressor": compressor,
        }
    
    for coord in ds.coords:
        encoding[coord] = {"compressor": compressor}
    
    print(f"\nWriting rechunked dataset to: {output_path}")
    ds.to_zarr(str(output_path), zarr_format=3, mode="w", encoding=encoding)
    
    # Verify
    print("\nVerifying...")
    ds_new = xr.open_zarr(str(output_path), zarr_format=3)
    print(f"New chunks:")
    for var in ds_new.data_vars:
        print(f"  {var}: {ds_new[var].encoding.get('chunks')}")
    
    # Optionally replace original
    print(f"\n✓ Done!")
    print(f"\nTo replace the original:")
    print(f"  rm -rf {input_path}")
    print(f"  mv {output_path} {input_path}")


if __name__ == "__main__":
    main()
