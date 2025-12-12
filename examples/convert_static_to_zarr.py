#!/usr/bin/env python3
"""Download Aurora 0.1° HRES static variables from HuggingFace and convert to Zarr.

This script:
1. Downloads aurora-0.1-static.pickle from HuggingFace (microsoft/aurora)
2. Converts to an xarray Dataset with proper coordinates
3. Saves as a Zarr store for efficient access

Static variables (0.1° resolution, 1801x3600 grid):
    - z: Surface geopotential (m²/s²)
    - slt: Soil type (categorical, 0-7)
    - lsm: Land-sea mask (0-1)

Usage:
    python examples/convert_static_to_zarr.py
    python examples/convert_static_to_zarr.py --output examples/downloads/static_hres.zarr
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from huggingface_hub import hf_hub_download

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


# HRES 0.1° static pickle from HuggingFace
STATIC_FILENAME = "aurora-0.1-static.pickle"


def download_static_pickle(cache_dir: Path | None = None) -> Path:
    """Download Aurora 0.1° static variables pickle from HuggingFace."""
    cache_str = str(cache_dir) if cache_dir else None
    
    print(f"Downloading {STATIC_FILENAME} from microsoft/aurora...")
    pickle_path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename=STATIC_FILENAME,
        cache_dir=cache_str,
    )
    print(f"Downloaded to: {pickle_path}")
    return Path(pickle_path)


def load_static_pickle(pickle_path: Path) -> dict[str, np.ndarray]:
    """Load static variables from pickle file."""
    print(f"Loading static variables from {pickle_path}...")
    with open(pickle_path, "rb") as f:
        static_vars = pickle.load(f)
    
    print(f"Found variables: {list(static_vars.keys())}")
    for name, arr in static_vars.items():
        print(f"  {name}: shape={arr.shape}, dtype={arr.dtype}")
    
    return static_vars


def create_static_dataset(static_vars: dict[str, np.ndarray]) -> xr.Dataset:
    """Create xarray Dataset from static variables with 0.1° HRES coordinates."""
    # Get shape from first variable
    first_var = next(iter(static_vars.values()))
    nlat, nlon = first_var.shape
    
    # 0.1° HRES grid: expected 1801 x 3600
    latitudes = np.linspace(90, -90, nlat)
    longitudes = np.linspace(0, 360, nlon, endpoint=False)
    
    # Variable attributes
    attrs = {
        "z": {"long_name": "Surface geopotential", "units": "m^2/s^2"},
        "slt": {"long_name": "Soil type", "units": "1"},
        "lsm": {"long_name": "Land-sea mask", "units": "1"},
    }
    
    # Create dataset
    data_vars = {
        name: xr.DataArray(
            arr.astype(np.float32),
            dims=["latitude", "longitude"],
            attrs=attrs.get(name, {"long_name": name}),
        )
        for name, arr in static_vars.items()
    }
    
    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "latitude": ("latitude", latitudes.astype(np.float64)),
            "longitude": ("longitude", longitudes.astype(np.float64)),
        },
        attrs={
            "source": "microsoft/aurora (HuggingFace)",
            "resolution": "0.1 degrees (HRES)",
            "description": "Aurora static surface variables",
        },
    )
    
    print(f"\nCreated Dataset: {dict(ds.dims)}")
    print(f"  Lat: [{latitudes.min():.1f}, {latitudes.max():.1f}]")
    print(f"  Lon: [{longitudes.min():.1f}, {longitudes.max():.1f}]")
    
    return ds


def save_to_zarr(ds: xr.Dataset, output_path: Path, overwrite: bool = False) -> None:
    """Save dataset to Zarr format."""
    output_path = Path(output_path)
    
    if output_path.exists():
        if overwrite:
            import shutil
            print(f"Removing existing: {output_path}")
            shutil.rmtree(output_path)
        else:
            raise FileExistsError(f"{output_path} exists. Use --overwrite.")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Chunk for efficient access
    ds_chunked = ds.chunk({"latitude": 180, "longitude": 360})
    
    print(f"\nSaving to Zarr: {output_path}")
    ds_chunked.to_zarr(output_path, mode="w", consolidated=True)
    
    total_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"Done! Size: {total_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Convert Aurora 0.1° static variables to Zarr")
    parser.add_argument("--output", "-o", type=Path, default=Path("/projects/prjs1858/static_hres.zarr"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Aurora 0.1° HRES Static Variables -> Zarr")
    print("=" * 60)
    
    pickle_path = download_static_pickle(args.cache_dir)
    static_vars = load_static_pickle(pickle_path)
    ds = create_static_dataset(static_vars)
    save_to_zarr(ds, args.output, args.overwrite)
    
    print(f"\n✓ Complete! Output: {args.output}")


if __name__ == "__main__":
    main()
