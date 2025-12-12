#!/usr/bin/env python3
"""Extensive validation script for the latents Zarr dataset structure and contents.

Checks:
- Coordinate integrity (types, ranges, NaN values)
- Data array shapes and dtypes
- Data quality (NaN distribution, value ranges, statistics)
- Metadata completeness
- Zarr structure and chunking
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import zarr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def print_section(title: str, char: str = "="):
    """Print a formatted section header."""
    print(f"\n{char*60}")
    print(f" {title}")
    print(f"{char*60}")


def validate_coordinate(name: str, coord_values: np.ndarray, expected_dtype: str = None) -> dict:
    """Validate a coordinate array and return diagnostics."""
    result = {
        "name": name,
        "shape": coord_values.shape,
        "dtype": str(coord_values.dtype),
        "n_values": len(coord_values),
        "n_nan": int(np.sum(np.isnan(coord_values.astype(float)))),
        "n_unique": len(np.unique(coord_values[~np.isnan(coord_values.astype(float))])),
        "min": float(np.nanmin(coord_values)) if len(coord_values) > 0 else None,
        "max": float(np.nanmax(coord_values)) if len(coord_values) > 0 else None,
        "is_monotonic": None,
        "has_duplicates": False,
        "issues": [],
    }
    
    # Check for NaN values
    if result["n_nan"] > 0:
        result["issues"].append(f"Contains {result['n_nan']} NaN values")
    
    # Check dtype
    if expected_dtype and not np.issubdtype(coord_values.dtype, np.dtype(expected_dtype)):
        result["issues"].append(f"Expected dtype {expected_dtype}, got {coord_values.dtype}")
    
    # Check monotonicity (for lat/lon/time)
    if len(coord_values) > 1 and result["n_nan"] == 0:
        diffs = np.diff(coord_values.astype(float))
        is_increasing = np.all(diffs > 0)
        is_decreasing = np.all(diffs < 0)
        result["is_monotonic"] = is_increasing or is_decreasing
        
        # Check for duplicates
        if result["n_unique"] != result["n_values"]:
            result["has_duplicates"] = True
            result["issues"].append("Contains duplicate values")
    
    return result


def validate_data_array(name: str, data: np.ndarray, sample_size: int = 10) -> dict:
    """Validate a data array and return diagnostics."""
    result = {
        "name": name,
        "shape": data.shape,
        "dtype": str(data.dtype),
        "n_elements": int(np.prod(data.shape)),
        "n_nan": int(np.sum(np.isnan(data))),
        "nan_percentage": 0.0,
        "has_inf": bool(np.isinf(data).any()),
        "min": None,
        "max": None,
        "mean": None,
        "std": None,
        "issues": [],
    }
    
    result["nan_percentage"] = 100 * result["n_nan"] / result["n_elements"]
    
    # Compute statistics on non-NaN values
    valid_data = data[~np.isnan(data)]
    if len(valid_data) > 0:
        result["min"] = float(np.min(valid_data))
        result["max"] = float(np.max(valid_data))
        result["mean"] = float(np.mean(valid_data))
        result["std"] = float(np.std(valid_data))
    
    # Check for issues
    if result["has_inf"]:
        result["issues"].append("Contains infinite values")
    
    if result["nan_percentage"] == 100:
        result["issues"].append("All values are NaN")
    elif result["nan_percentage"] > 50:
        result["issues"].append(f"High NaN percentage: {result['nan_percentage']:.1f}%")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Extensive latents Zarr dataset validation")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("examples/latents_europe_2018_2020.zarr"),
        help="Path to the latents Zarr dataset",
    )
    parser.add_argument(
        "--n-timesteps-check",
        type=int,
        default=10,
        help="Number of timesteps to check for data validation",
    )
    args = parser.parse_args()
    
    print(f"Opening {args.zarr_path}...")
    
    # Open with xarray first to get high-level view
    ds = xr.open_zarr(str(args.zarr_path), consolidated=True)
    
    # Also open raw zarr for low-level inspection
    store = zarr.open(str(args.zarr_path), mode="r")
    
    issues_found = []
    
    # =========================================================================
    # SECTION 1: Raw Dataset Info
    # =========================================================================
    print_section("RAW XARRAY DATASET INFO")
    print("\n📋 ds.coords:")
    print(ds.coords)
    print("\n📋 Full Dataset:")
    print(ds)
    
    # =========================================================================
    # SECTION 2: Zarr Structure Validation
    # =========================================================================
    print_section("ZARR STRUCTURE VALIDATION")
    
    print("\n📁 ZARR ARRAYS IN STORE:")
    for key in store.array_keys():
        arr = store[key]
        print(f"  {key}:")
        print(f"    shape: {arr.shape}")
        print(f"    dtype: {arr.dtype}")
        print(f"    chunks: {arr.chunks}")
        if hasattr(arr, 'attrs') and '_ARRAY_DIMENSIONS' in arr.attrs:
            print(f"    dimensions: {arr.attrs['_ARRAY_DIMENSIONS']}")
    
    # =========================================================================
    # SECTION 3: Coordinate Validation
    # =========================================================================
    print_section("COORDINATE VALIDATION")
    
    # Check channel coordinate specifically (the issue mentioned by user)
    print("\n🔍 CHANNEL COORDINATE DEBUG:")
    channel_raw = store["channel"][:]
    print(f"  Raw zarr channel array: {channel_raw[:10]}... (first 10)")
    print(f"  Raw dtype: {channel_raw.dtype}")
    print(f"  Contains 0? {0 in channel_raw}")
    print(f"  First value: {channel_raw[0]} (type: {type(channel_raw[0]).__name__})")
    
    # Check if xarray is converting 0 to NaN somehow
    channel_xr = ds.channel.values
    print(f"  xarray channel values: {channel_xr[:10]}... (first 10)")
    print(f"  xarray dtype: {channel_xr.dtype}")
    print(f"  First xarray value: {channel_xr[0]}")
    print(f"  Is first value NaN? {np.isnan(channel_xr[0])}")
    
    if channel_raw[0] == 0 and np.isnan(channel_xr[0]):
        print("  ⚠️ ISSUE: 0 in raw zarr is being converted to NaN by xarray!")
        print("     This might be due to dtype mismatch or fill_value settings")
        issues_found.append("Channel 0 → NaN conversion issue")
    
    # Validate all coordinates
    coords_to_check = [
        ("lat", "float"),
        ("lon", "float"),
        ("level", "float"),  # Could be int but stored as float
        ("channel", "int"),
        ("time", None),  # datetime
    ]
    
    print("\n📍 COORDINATE DETAILS:")
    for coord_name, expected_dtype in coords_to_check:
        if coord_name not in ds.coords:
            print(f"  ❌ {coord_name}: MISSING")
            issues_found.append(f"Missing coordinate: {coord_name}")
            continue
            
        coord_values = ds[coord_name].values
        result = validate_coordinate(coord_name, coord_values)
        
        status = "✅" if len(result["issues"]) == 0 else "⚠️"
        print(f"\n  {status} {coord_name}:")
        print(f"      Shape: {result['shape']}, dtype: {result['dtype']}")
        print(f"      Values: {result['n_values']}, Unique: {result['n_unique']}, NaN: {result['n_nan']}")
        
        if result["min"] is not None:
            if coord_name == "time":
                print(f"      Range: [{ds.time.values[0]} → {ds.time.values[-1]}]")
            else:
                print(f"      Range: [{result['min']:.4f} → {result['max']:.4f}]")
        
        if result["is_monotonic"] is not None:
            print(f"      Monotonic: {result['is_monotonic']}")
        
        for issue in result["issues"]:
            print(f"      ⚠️ Issue: {issue}")
            issues_found.append(f"{coord_name}: {issue}")
    
    # =========================================================================
    # SECTION 4: Processed Timesteps Analysis
    # =========================================================================
    print_section("PROCESSED TIMESTEPS ANALYSIS")
    
    processed_mask = ds.processed.values.astype(bool)
    n_processed = int(processed_mask.sum())
    n_total = len(ds.time)
    
    print(f"\n📊 PROCESSING STATUS:")
    print(f"  Total timesteps: {n_total}")
    print(f"  Processed: {n_processed} ({100*n_processed/n_total:.1f}%)")
    print(f"  Remaining: {n_total - n_processed}")
    
    if n_processed > 0:
        processed_indices = np.where(processed_mask)[0]
        unprocessed_indices = np.where(~processed_mask)[0]
        
        print(f"\n  First processed idx: {processed_indices[0]} → {ds.time.values[processed_indices[0]]}")
        print(f"  Last processed idx: {processed_indices[-1]} → {ds.time.values[processed_indices[-1]]}")
        
        # Check for gaps in processing
        if len(processed_indices) > 1:
            diffs = np.diff(processed_indices)
            gaps = np.where(diffs > 1)[0]
            if len(gaps) > 0:
                print(f"\n  ⚠️ Found {len(gaps)} gaps in processed timesteps:")
                for gap_idx in gaps[:5]:  # Show first 5 gaps
                    start = processed_indices[gap_idx]
                    end = processed_indices[gap_idx + 1]
                    print(f"      Gap at indices {start}-{end} ({end-start-1} missing)")
                issues_found.append(f"Found {len(gaps)} gaps in processed timesteps")
    
    # =========================================================================
    # SECTION 5: Data Array Validation
    # =========================================================================
    print_section("DATA ARRAY VALIDATION")
    
    if n_processed > 0:
        # Get sample indices for validation
        n_to_check = min(n_processed, args.n_timesteps_check)
        processed_indices = np.where(processed_mask)[0]
        sample_indices = processed_indices[:n_to_check]
        
        print(f"\n🔬 Checking {n_to_check} processed timesteps (indices: {list(sample_indices[:5])}...)")
        
        # Surface latents validation
        print("\n📊 SURFACE LATENTS:")
        surf_samples = []
        for idx in sample_indices:
            data = ds.surface_latents.isel(time=int(idx)).values
            surf_samples.append(data)
        surface_data = np.stack(surf_samples)
        
        result = validate_data_array("surface_latents", surface_data)
        print(f"  Sample shape: {result['shape']} (first dim = timesteps checked)")
        print(f"  Dtype: {result['dtype']}")
        print(f"  NaN count: {result['n_nan']} ({result['nan_percentage']:.4f}%)")
        print(f"  Range: [{result['min']:.4f}, {result['max']:.4f}]")
        print(f"  Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")
        print(f"  Has Inf: {result['has_inf']}")
        
        for issue in result["issues"]:
            print(f"  ⚠️ {issue}")
            issues_found.append(f"surface_latents: {issue}")
        
        # Check first channel specifically
        print(f"\n  Per-channel stats (first 5 channels):")
        for c in range(min(5, surface_data.shape[-1])):
            channel_data = surface_data[..., c]
            n_nan = np.sum(np.isnan(channel_data))
            mean_val = np.nanmean(channel_data)
            std_val = np.nanstd(channel_data)
            print(f"    Channel {c}: mean={mean_val:.4f}, std={std_val:.4f}, NaN={n_nan}")
        
        # Pressure latents validation
        print("\n📊 PRESSURE LATENTS:")
        pres_samples = []
        for idx in sample_indices:
            data = ds.pressure_latents.isel(time=int(idx)).values
            pres_samples.append(data)
        pressure_data = np.stack(pres_samples)
        
        result = validate_data_array("pressure_latents", pressure_data)
        print(f"  Sample shape: {result['shape']} (first dim = timesteps checked)")
        print(f"  Dtype: {result['dtype']}")
        print(f"  NaN count: {result['n_nan']} ({result['nan_percentage']:.4f}%)")
        print(f"  Range: [{result['min']:.4f}, {result['max']:.4f}]")
        print(f"  Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")
        print(f"  Has Inf: {result['has_inf']}")
        
        for issue in result["issues"]:
            print(f"  ⚠️ {issue}")
            issues_found.append(f"pressure_latents: {issue}")
        
        # Per-level statistics
        print(f"\n  Per-level statistics:")
        for i, level in enumerate(ds.level.values):
            level_data = pressure_data[:, i, :, :, :]
            n_nan = np.sum(np.isnan(level_data))
            pct_nan = 100 * n_nan / level_data.size
            mean_val = np.nanmean(level_data)
            std_val = np.nanstd(level_data)
            min_val = np.nanmin(level_data)
            max_val = np.nanmax(level_data)
            
            status = "✅" if pct_nan == 0 else "⚠️"
            print(f"    {status} Level {int(level):4d} hPa: "
                  f"mean={mean_val:+.4f}, std={std_val:.4f}, "
                  f"range=[{min_val:.2f}, {max_val:.2f}], "
                  f"NaN={pct_nan:.2f}%")
        
        # Check channel consistency between surface and pressure
        print(f"\n  Per-channel stats for pressure (first 5 channels, averaged over levels):")
        for c in range(min(5, pressure_data.shape[-1])):
            channel_data = pressure_data[..., c]
            n_nan = np.sum(np.isnan(channel_data))
            mean_val = np.nanmean(channel_data)
            std_val = np.nanstd(channel_data)
            print(f"    Channel {c}: mean={mean_val:.4f}, std={std_val:.4f}, NaN={n_nan}")
        
    else:
        print("\n⚠️ No processed timesteps found - skipping data validation")
        issues_found.append("No processed timesteps")
    
    # =========================================================================
    # SECTION 6: Check Unprocessed Timesteps
    # =========================================================================
    print_section("UNPROCESSED TIMESTEPS CHECK")
    
    unprocessed_indices = np.where(~processed_mask)[0]
    if len(unprocessed_indices) > 0:
        print(f"\n🔍 Checking {min(3, len(unprocessed_indices))} unprocessed timesteps...")
        
        for idx in unprocessed_indices[:3]:
            surf_data = ds.surface_latents.isel(time=int(idx)).values
            pres_data = ds.pressure_latents.isel(time=int(idx)).values
            
            surf_all_nan = np.all(np.isnan(surf_data))
            pres_all_nan = np.all(np.isnan(pres_data))
            surf_all_zero = np.all(surf_data == 0)
            pres_all_zero = np.all(pres_data == 0)
            
            print(f"\n  Timestep {idx} ({ds.time.values[idx]}):")
            print(f"    Surface: all_NaN={surf_all_nan}, all_zero={surf_all_zero}")
            print(f"    Pressure: all_NaN={pres_all_nan}, all_zero={pres_all_zero}")
            
            if not (surf_all_nan or surf_all_zero):
                print(f"    ⚠️ WARNING: Unprocessed surface data is not empty!")
            if not (pres_all_nan or pres_all_zero):
                print(f"    ⚠️ WARNING: Unprocessed pressure data is not empty!")
    else:
        print("\n✅ All timesteps are processed!")
    
    # =========================================================================
    # SECTION 7: Bounds Validation
    # =========================================================================
    print_section("BOUNDS VALIDATION")
    
    print("\n📐 PATCH BOUNDS:")
    lat_bounds = ds.lat_bounds.values
    lon_bounds = ds.lon_bounds.values
    
    print(f"  lat_bounds shape: {lat_bounds.shape}")
    print(f"  lon_bounds shape: {lon_bounds.shape}")
    
    # Check bounds are sensible
    lat_centers = ds.lat.values
    lon_centers = ds.lon.values
    
    print(f"\n  Lat center-to-bounds check (first 3):")
    for i in range(min(3, len(lat_centers))):
        center = lat_centers[i]
        bounds = lat_bounds[i]
        span = bounds[1] - bounds[0]
        mid = (bounds[0] + bounds[1]) / 2
        print(f"    lat[{i}]={center:.2f}, bounds=[{bounds[0]:.2f}, {bounds[1]:.2f}], "
              f"span={span:.2f}, midpoint={mid:.2f}")
    
    print(f"\n  Lon center-to-bounds check (first 3):")
    for i in range(min(3, len(lon_centers))):
        center = lon_centers[i]
        bounds = lon_bounds[i]
        span = bounds[1] - bounds[0]
        mid = (bounds[0] + bounds[1]) / 2
        print(f"    lon[{i}]={center:.2f}, bounds=[{bounds[0]:.2f}, {bounds[1]:.2f}], "
              f"span={span:.2f}, midpoint={mid:.2f}")
    
    # =========================================================================
    # SECTION 8: Attribute Validation
    # =========================================================================
    print_section("ATTRIBUTE VALIDATION")
    
    expected_attrs = [
        "atmos_levels",
        "start_time", "end_time",
        "n_timesteps",
        "region_lat_min", "region_lat_max",
        "region_lon_min", "region_lon_max",
        "patch_row_start", "patch_row_end",
        "patch_col_start", "patch_col_end",
        "surface_shape", "pressure_shape",
    ]
    
    print("\n📝 DATASET ATTRIBUTES:")
    for attr in expected_attrs:
        if attr in ds.attrs:
            val = ds.attrs[attr]
            print(f"  ✅ {attr}: {val}")
        else:
            print(f"  ❌ {attr}: MISSING")
            issues_found.append(f"Missing attribute: {attr}")
    
    # Any extra attributes
    extra_attrs = set(ds.attrs.keys()) - set(expected_attrs)
    if extra_attrs:
        print(f"\n  Additional attributes:")
        for attr in sorted(extra_attrs):
            print(f"     {attr}: {ds.attrs[attr]}")
    
    # =========================================================================
    # SECTION 9: Example Access Patterns
    # =========================================================================
    print_section("EXAMPLE ACCESS PATTERNS")
    
    if n_processed > 0:
        print("\n🔍 ACCESS EXAMPLES:")
        
        # Single timestep
        t0_surface = ds.surface_latents.isel(time=0)
        t0_pressure = ds.pressure_latents.isel(time=0)
        print(f"  ds.surface_latents.isel(time=0).shape: {t0_surface.shape}")
        print(f"  ds.pressure_latents.isel(time=0).shape: {t0_pressure.shape}")
        
        # Single point
        lat_idx, lon_idx = len(ds.lat)//2, len(ds.lon)//2
        point_surface = ds.surface_latents.isel(time=0, lat=lat_idx, lon=lon_idx)
        print(f"  Single patch (lat={lat_idx}, lon={lon_idx}): shape={point_surface.shape}")
        
        # Time series for a point
        ts_data = ds.surface_latents.isel(lat=lat_idx, lon=lon_idx, channel=0)
        print(f"  Time series for one point, one channel: shape={ts_data.shape}")
        
        # Single level from pressure
        level_data = ds.pressure_latents.isel(time=0, level=6)  # 500 hPa approx
        print(f"  Single level (level=6, ~500 hPa): shape={level_data.shape}")
    
    # =========================================================================
    # SECTION 10: Summary
    # =========================================================================
    print_section("VALIDATION SUMMARY")
    
    if len(issues_found) == 0:
        print("\n✅ ALL CHECKS PASSED!")
        print("   No issues found in the dataset.")
    else:
        print(f"\n⚠️ FOUND {len(issues_found)} ISSUE(S):")
        for i, issue in enumerate(issues_found, 1):
            print(f"   {i}. {issue}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60 + "\n")
    
    ds.close()


if __name__ == "__main__":
    main()
