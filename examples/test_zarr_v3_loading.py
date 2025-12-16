#!/usr/bin/env python3
"""Quick test to verify Zarr v3 datasets load correctly.

Usage:
    python examples/test_zarr_v3_loading.py
"""

import sys

# Test both xarray and direct zarr access
def test_xarray_loading(path: str, is_v3: bool = True):
    """Test loading with xarray."""
    import xarray as xr
    
    try:
        if is_v3:
            ds = xr.open_zarr(path, consolidated=False, zarr_format=3)
        else:
            ds = xr.open_zarr(path, consolidated=True)
        
        print(f"  ✓ xarray loaded successfully")
        print(f"    Variables: {list(ds.data_vars)}")
        print(f"    Dimensions: {dict(ds.dims)}")
        if "time" in ds.dims:
            print(f"    Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
        return True, ds
    except Exception as e:
        print(f"  ✗ xarray failed: {e}")
        return False, None


def test_zarr_loading(path: str):
    """Test loading with direct zarr access."""
    import zarr
    
    try:
        store = zarr.open_group(path, mode="r")
        print(f"  ✓ zarr loaded successfully")
        print(f"    Arrays: {list(store.array_keys())}")
        
        # Print shape for each array
        for name in store.array_keys():
            arr = store[name]
            print(f"    {name}: shape={arr.shape}, dtype={arr.dtype}")
        return True, store
    except Exception as e:
        print(f"  ✗ zarr failed: {e}")
        return False, None


def test_sample_access(path: str, var_name: str):
    """Test accessing a single sample (the real test for sharding)."""
    import zarr
    import time
    
    try:
        store = zarr.open_group(path, mode="r")
        arr = store[var_name]
        
        # Time single sample access
        start = time.time()
        sample = arr[0]  # Get first timestep
        elapsed = time.time() - start
        
        print(f"  ✓ Sample access: {var_name}[0] loaded in {elapsed*1000:.1f}ms")
        print(f"    Sample shape: {sample.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Sample access failed: {e}")
        return False


def main():
    DATA_DIR = "/projects/prjs1858"
    
    datasets = [
        ("HRES v3", f"{DATA_DIR}/hres_europe_2018_2020_v3.zarr", "2t"),
        ("Latents v3", f"{DATA_DIR}/latents_europe_2018_2020_v3.zarr", "surface_latents"),
        ("Static v3", f"{DATA_DIR}/static_hres_europe_v3.zarr", "z"),
    ]
    
    print("=" * 60)
    print("Zarr v3 Dataset Loading Test")
    print("=" * 60)
    
    all_passed = True
    
    for name, path, test_var in datasets:
        print(f"\n{name}: {path}")
        print("-" * 50)
        
        # Test xarray
        xr_ok, _ = test_xarray_loading(path, is_v3=True)
        
        # Test direct zarr
        zarr_ok, _ = test_zarr_loading(path)
        
        # Test sample access
        access_ok = test_sample_access(path, test_var)
        
        if not (xr_ok or zarr_ok):
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All datasets loaded successfully!")
        print("  You can now use zarr_format: 3 in your training config.")
    else:
        print("⚠ Some tests failed. Check output above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
