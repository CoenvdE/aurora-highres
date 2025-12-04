#!/usr/bin/env python3
"""Compare static files from different sources to verify they contain the same data.

This script compares:
1. Manually downloaded ERA5 static.nc (from CDS API)
2. HuggingFace pickle converted to NetCDF (aurora-0.25-static.pickle)
"""

from pathlib import Path
import xarray as xr
import numpy as np
from huggingface_hub import hf_hub_download
import pickle


def load_era5_static(path: Path) -> xr.Dataset:
    """Load manually downloaded ERA5 static file."""
    return xr.open_dataset(path)


def load_hf_pickle_static(cache_dir: Path) -> dict:
    """Download and load HuggingFace pickle static data."""
    pickle_path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.25-static.pickle",
        cache_dir=str(cache_dir),
    )
    
    with open(pickle_path, "rb") as f:
        static_vars = pickle.load(f)
    
    return static_vars


def compare_variables(era5_ds: xr.Dataset, hf_vars: dict) -> None:
    """Compare variable names and contents."""
    print("\n" + "="*60)
    print("VARIABLE COMPARISON")
    print("="*60)
    
    era5_vars = set(era5_ds.data_vars)
    hf_var_names = set(hf_vars.keys())
    
    print(f"\nERA5 variables: {sorted(era5_vars)}")
    print(f"HuggingFace variables: {sorted(hf_var_names)}")
    
    print(f"\n✓ Variables in ERA5 only: {sorted(era5_vars - hf_var_names)}")
    print(f"✓ Variables in HuggingFace only: {sorted(hf_var_names - era5_vars)}")
    print(f"✓ Common variables: {sorted(era5_vars & hf_var_names)}")


def compare_data_values(era5_ds: xr.Dataset, hf_vars: dict) -> None:
    """Compare the actual data values for common variables."""
    print("\n" + "="*60)
    print("DATA VALUE COMPARISON")
    print("="*60)
    
    common_vars = set(era5_ds.data_vars) & set(hf_vars.keys())
    
    for var_name in sorted(common_vars):
        print(f"\n📊 Variable: {var_name}")
        era5_data = era5_ds[var_name].values
        hf_data = hf_vars[var_name]
        
        print(f"  ERA5 shape: {era5_data.shape}")
        print(f"  HuggingFace shape: {hf_data.shape}")
        
        if era5_data.shape != hf_data.shape:
            print(f"  ✗ SHAPES DIFFER!")
            continue
        
        # Check if values are identical
        if np.array_equal(era5_data, hf_data):
            print(f"  ✓ Values are IDENTICAL")
        else:
            # Check if they're close (accounting for floating point precision)
            if np.allclose(era5_data, hf_data, rtol=1e-5, atol=1e-8, equal_nan=True):
                print(f"  ✓ Values are NEARLY IDENTICAL (within floating point tolerance)")
                print(f"    Max absolute difference: {np.nanmax(np.abs(era5_data - hf_data))}")
            else:
                print(f"  ✗ Values DIFFER significantly")
                print(f"    Max absolute difference: {np.nanmax(np.abs(era5_data - hf_data))}")
                print(f"    Mean absolute difference: {np.nanmean(np.abs(era5_data - hf_data))}")
                
                # Show statistics
                print(f"\n  ERA5 stats:")
                print(f"    Min: {np.nanmin(era5_data)}, Max: {np.nanmax(era5_data)}")
                print(f"    Mean: {np.nanmean(era5_data)}, Std: {np.nanstd(era5_data)}")
                print(f"\n  HuggingFace stats:")
                print(f"    Min: {np.nanmin(hf_data)}, Max: {np.nanmax(hf_data)}")
                print(f"    Mean: {np.nanmean(hf_data)}, Std: {np.nanstd(hf_data)}")


def compare_coordinates(era5_ds: xr.Dataset) -> None:
    """Check ERA5 coordinate system."""
    print("\n" + "="*60)
    print("COORDINATE SYSTEM")
    print("="*60)
    
    print(f"\nERA5 coordinates:")
    for coord_name in era5_ds.coords:
        coord = era5_ds.coords[coord_name]
        print(f"  {coord_name}: shape={coord.shape}, range=[{coord.min().values}, {coord.max().values}]")
    
    # Expected HuggingFace coordinates
    print(f"\nExpected HuggingFace coordinates (from pickle conversion):")
    print(f"  latitude: shape=(721,), range=[90, -90]")
    print(f"  longitude: shape=(1440,), range=[0, 360)")


def main():
    # Paths
    era5_static_path = Path("examples/downloads/era5/static.nc")
    hf_cache_dir = Path("examples/downloads/hf_cache")
    
    print("="*60)
    print("STATIC FILE COMPARISON TOOL")
    print("="*60)
    
    # Check if ERA5 file exists
    if not era5_static_path.exists():
        print(f"\n⚠️  ERA5 static file not found at {era5_static_path}")
        print("   Please run examples/init_exploring/download_era_sample.py first")
        return
    
    print(f"\n✓ Loading ERA5 static file from {era5_static_path}")
    era5_ds = load_era5_static(era5_static_path)
    
    print(f"✓ Downloading HuggingFace pickle...")
    hf_vars = load_hf_pickle_static(hf_cache_dir)
    
    # Perform comparisons
    compare_variables(era5_ds, hf_vars)
    compare_coordinates(era5_ds)
    compare_data_values(era5_ds, hf_vars)
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("""
The HuggingFace pickle file contains PREPROCESSED static data that Aurora
was trained on. This may or may not be identical to raw ERA5 data because:

1. Microsoft may have applied their own preprocessing
2. They may have used a different reference date/time
3. They may have normalized or scaled certain variables
4. They may have derived variables differently

For Aurora inference, you should use the HuggingFace static data
(aurora-0.25-static.pickle) because that's what the model expects.

The manually downloaded ERA5 static.nc is useful for:
- Understanding the raw data
- Custom preprocessing pipelines
- Validation and debugging
    """)


if __name__ == "__main__":
    main()

# Usage:
# python examples/compare_static_files.py
