#!/usr/bin/env python3
"""Validate the latents Zarr dataset structure and contents."""

import argparse
from pathlib import Path
import numpy as np
import xarray as xr

def main():
    parser = argparse.ArgumentParser(description="Validate latents Zarr dataset")
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("examples/latents_europe_2018_2020.zarr"),
        help="Path to the latents Zarr dataset",
    )
    args = parser.parse_args()
    
    print(f"Opening {args.zarr_path}...")
    ds = xr.open_zarr(str(args.zarr_path))
    
    print("\n" + "="*60)
    print("ZARR STRUCTURE VALIDATION")
    print("="*60)
    
    # 1. Check coordinates
    print("\n📍 COORDINATES:")
    print(f"  lat: {ds.lat.values.shape} - range [{ds.lat.values.min():.2f}, {ds.lat.values.max():.2f}]")
    print(f"  lon: {ds.lon.values.shape} - range [{ds.lon.values.min():.2f}, {ds.lon.values.max():.2f}]")
    print(f"  level: {ds.level.values}")
    print(f"  time: {len(ds.time)} timesteps")
    print(f"    First: {ds.time.values[0]}")
    processed_mask = ds.processed.values.astype(bool)
    print(f"    Last processed: {ds.time.values[processed_mask].max() if processed_mask.any() else 'None'}")
    
    # 2. Check data arrays
    print("\n📊 DATA ARRAYS:")
    print(f"  surface_latents: {ds.surface_latents.shape}")
    print(f"  pressure_latents: {ds.pressure_latents.shape}")
    print(f"  processed: {ds.processed.values.sum()} / {len(ds.processed)} timesteps")
    
    # 3. Check bounds
    print("\n📐 PATCH BOUNDS:")
    print(f"  lat_bounds: {ds.lat_bounds.shape}")
    print(f"  lon_bounds: {ds.lon_bounds.shape}")
    
    # 4. Check attributes
    print("\n📝 ATTRIBUTES:")
    for key in sorted(ds.attrs.keys()):
        print(f"  {key}: {ds.attrs[key]}")
    
    # 5. Validate data for processed timesteps
    print("\n✅ DATA VALIDATION:")
    n_processed = processed_mask.sum()
    
    if n_processed > 0:
        # Check surface latents
        surface_data = ds.surface_latents.isel(time=slice(0, n_processed)).values
        surf_has_nan = np.isnan(surface_data).any()
        surf_range = (np.nanmin(surface_data), np.nanmax(surface_data))
        surf_mean = np.nanmean(surface_data)
        surf_std = np.nanstd(surface_data)
        
        print(f"  Surface latents (first {n_processed} timesteps):")
        print(f"    Has NaN: {surf_has_nan}")
        print(f"    Range: [{surf_range[0]:.4f}, {surf_range[1]:.4f}]")
        print(f"    Mean: {surf_mean:.4f}, Std: {surf_std:.4f}")
        
        # Check pressure latents
        pressure_data = ds.pressure_latents.isel(time=slice(0, n_processed)).values
        pres_has_nan = np.isnan(pressure_data).any()
        pres_range = (np.nanmin(pressure_data), np.nanmax(pressure_data))
        pres_mean = np.nanmean(pressure_data)
        pres_std = np.nanstd(pressure_data)
        
        print(f"  Pressure latents (first {n_processed} timesteps):")
        print(f"    Has NaN: {pres_has_nan}")
        print(f"    Range: [{pres_range[0]:.4f}, {pres_range[1]:.4f}]")
        print(f"    Mean: {pres_mean:.4f}, Std: {pres_std:.4f}")
        
        # Check per-level statistics
        print(f"\n  Per-level pressure latent stats:")
        for i, level in enumerate(ds.level.values):
            level_data = pressure_data[:, i, :, :, :]
            print(f"    Level {level:4d} hPa: mean={np.nanmean(level_data):.4f}, std={np.nanstd(level_data):.4f}")
    
    # 6. Example access patterns
    print("\n🔍 EXAMPLE ACCESS:")
    if n_processed > 0:
        # Get first timestep
        t0_surface = ds.surface_latents.isel(time=0).values
        t0_pressure = ds.pressure_latents.isel(time=0).values
        print(f"  ds.surface_latents.isel(time=0).shape: {t0_surface.shape}")
        print(f"  ds.pressure_latents.isel(time=0).shape: {t0_pressure.shape}")
        
        # Get a specific lat/lon
        lat_idx, lon_idx = len(ds.lat)//2, len(ds.lon)//2
        point_surface = ds.surface_latents.isel(time=0, lat=lat_idx, lon=lon_idx).values
        print(f"  Single patch surface latent: shape={point_surface.shape}")
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE ✓")
    print("="*60)
    
    ds.close()

if __name__ == "__main__":
    main()
