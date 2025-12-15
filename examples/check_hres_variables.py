#!/usr/bin/env python3
"""
Quick check script to inspect what variables are in the HRES zarr file.

Usage:
    python scripts/check_hres_variables.py
    python scripts/check_hres_variables.py --zarr-path /path/to/hres.zarr
"""

import argparse
import xarray as xr
from pathlib import Path


def inspect_zarr(zarr_path: str):
    """Open and inspect a zarr file."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {zarr_path}")
    print('='*60)
    
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception as e:
        print(f"❌ Error opening zarr file: {e}")
        return
    
    # Coordinates
    print("\n📍 Coordinates:")
    for coord_name, coord in ds.coords.items():
        values = coord.values
        print(f"  {coord_name}:")
        print(f"    - shape: {coord.shape}")
        print(f"    - dtype: {coord.dtype}")
        if coord.shape == ():
            print(f"    - value: {values}")
        elif len(values) > 0:
            # Check if values are numeric for formatting
            try:
                vmin = values.min()
                vmax = values.max()
                # Try float formatting
                if isinstance(vmin, (int, float)):
                    print(f"    - range: [{vmin:.4f}, {vmax:.4f}]")
                else:
                    print(f"    - range: [{vmin}, {vmax}]")
            except (TypeError, AttributeError):
                print(f"    - range: [{values[0]}, {values[-1]}]")
            
            if len(values) <= 5:
                print(f"    - values: {values}")
    
    # Dimensions
    print("\n📐 Dimensions:")
    for dim_name, size in ds.dims.items():
        print(f"  {dim_name}: {size}")
    
    # Data Variables
    print("\n📊 Data Variables:")
    if len(ds.data_vars) == 0:
        print("  (none)")
    else:
        for var_name, var in ds.data_vars.items():
            print(f"  {var_name}:")
            print(f"    - dims: {var.dims}")
            print(f"    - shape: {var.shape}")
            print(f"    - dtype: {var.dtype}")
            # Try to get a sample value
            try:
                sample = var.isel(time=0, latitude=0, longitude=0).values
                print(f"    - sample value: {sample}")
            except:
                pass
    
    # Attributes
    print("\n📝 Attributes:")
    if len(ds.attrs) == 0:
        print("  (none)")
    else:
        for attr_name, attr_value in ds.attrs.items():
            print(f"  {attr_name}: {attr_value}")
    
    print(f"\n{'='*60}")
    print(f"✅ Summary:")
    print(f"  - {len(ds.coords)} coordinates")
    print(f"  - {len(ds.dims)} dimensions")
    print(f"  - {len(ds.data_vars)} data variables: {list(ds.data_vars.keys())}")
    print('='*60)


def main():
    parser = argparse.ArgumentParser(
        description="Quick check of HRES zarr variables"
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="/projects/prjs1858/hres_europe_2018_2020.zarr",
        help="Path to HRES zarr file (default: /projects/prjs1858/hres_europe_2018_2020.zarr)"
    )
    args = parser.parse_args()
    
    if not Path(args.zarr_path).exists():
        print(f"❌ Zarr path does not exist: {args.zarr_path}")
        print("\nTrying alternative paths...")
        
        # Try alternative paths
        alternatives = [
            "/projects/prjs1858/hres_europe_2018.zarr",
            "/projects/prjs1858/hres_europe_2019.zarr",
            "/projects/prjs1858/hres_europe_2020.zarr",
        ]
        
        for alt_path in alternatives:
            if Path(alt_path).exists():
                print(f"✅ Found: {alt_path}")
                inspect_zarr(alt_path)
                return
        
        print("\n❌ No HRES zarr files found.")
        return
    
    inspect_zarr(args.zarr_path)


if __name__ == "__main__":
    main()
