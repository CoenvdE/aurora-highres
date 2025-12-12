#!/usr/bin/env python3
"""Verify that static and HRES zarr files are on the same grid and plot static variables.

This script:
1. Loads static_hres_europe.zarr and hres_europe_2018_2020.zarr
2. Confirms they have identical grids (lat/lon coordinates)
3. Plots all static variables (z, lsm, slt) with proper geographic extent

Usage:
    python examples/verify_static_hres_alignment.py
    python examples/verify_static_hres_alignment.py --output-dir plots/static
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def verify_grid_alignment(ds_static: xr.Dataset, ds_hres: xr.Dataset) -> bool:
    """Verify that static and HRES datasets have identical lat/lon grids."""
    # Get coordinate names
    static_lat = "latitude" if "latitude" in ds_static.coords else "lat"
    static_lon = "longitude" if "longitude" in ds_static.coords else "lon"
    hres_lat = "latitude" if "latitude" in ds_hres.coords else "lat"
    hres_lon = "longitude" if "longitude" in ds_hres.coords else "lon"
    
    static_lat_vals = ds_static[static_lat].values
    static_lon_vals = ds_static[static_lon].values
    hres_lat_vals = ds_hres[hres_lat].values
    hres_lon_vals = ds_hres[hres_lon].values
    
    print("=== Grid Comparison ===")
    print(f"\nStatic grid:")
    print(f"  lat: shape={static_lat_vals.shape}, range=[{static_lat_vals.min():.4f}, {static_lat_vals.max():.4f}]")
    print(f"  lon: shape={static_lon_vals.shape}, range=[{static_lon_vals.min():.4f}, {static_lon_vals.max():.4f}]")
    
    print(f"\nHRES grid:")
    print(f"  lat: shape={hres_lat_vals.shape}, range=[{hres_lat_vals.min():.4f}, {hres_lat_vals.max():.4f}]")
    print(f"  lon: shape={hres_lon_vals.shape}, range=[{hres_lon_vals.min():.4f}, {hres_lon_vals.max():.4f}]")
    
    # Check shape match
    shape_match = (static_lat_vals.shape == hres_lat_vals.shape and 
                   static_lon_vals.shape == hres_lon_vals.shape)
    
    # Check value match
    lat_match = np.allclose(static_lat_vals, hres_lat_vals, rtol=1e-5)
    lon_match = np.allclose(static_lon_vals, hres_lon_vals, rtol=1e-5)
    
    print(f"\n=== Alignment Check ===")
    print(f"  Shape match: {shape_match}")
    print(f"  Latitude values match: {lat_match}")
    print(f"  Longitude values match: {lon_match}")
    
    if shape_match and lat_match and lon_match:
        print("\n✓ GRIDS ARE ALIGNED - Static and HRES data can be used together!")
        return True
    else:
        print("\n✗ GRIDS DO NOT MATCH - Re-run slice_static_to_region.py with --hres-zarr")
        if not shape_match:
            print(f"  Static: ({len(static_lat_vals)}, {len(static_lon_vals)})")
            print(f"  HRES:   ({len(hres_lat_vals)}, {len(hres_lon_vals)})")
        if not lat_match:
            print(f"  Lat diff max: {np.abs(static_lat_vals - hres_lat_vals).max():.6f}")
        if not lon_match:
            print(f"  Lon diff max: {np.abs(static_lon_vals - hres_lon_vals).max():.6f}")
        return False


def plot_static_variables(
    ds_static: xr.Dataset, 
    output_dir: Path,
    extent: tuple[float, float, float, float],
) -> None:
    """Plot all static variables with geographic extent."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    lat_name = "latitude" if "latitude" in ds_static.coords else "lat"
    lon_name = "longitude" if "longitude" in ds_static.coords else "lon"
    
    # Variable info: name -> (title, colormap, label)
    var_info = {
        "z": ("Surface Geopotential", "terrain", "m²/s²"),
        "lsm": ("Land-Sea Mask", "Blues_r", "0=sea, 1=land"),
        "slt": ("Soil Type", "tab10", "categorical"),
    }
    
    for var_name in ds_static.data_vars:
        if var_name not in var_info:
            continue
            
        title, cmap, label = var_info[var_name]
        data = ds_static[var_name].values
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot with correct extent
        im = ax.imshow(
            data,
            extent=extent,
            origin='upper' if ds_static[lat_name].values[0] > ds_static[lat_name].values[-1] else 'lower',
            cmap=cmap,
            aspect='auto',
        )
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"{title} ({var_name})")
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(label)
        
        # Add grid lines
        ax.grid(True, alpha=0.3)
        
        output_path = output_dir / f"static_{var_name}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}")
        
        # Print stats
        print(f"    {var_name}: min={data.min():.2f}, max={data.max():.2f}, "
              f"mean={data.mean():.2f}, NaN%={100*np.isnan(data).mean():.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Verify static and HRES grid alignment and plot static variables"
    )
    parser.add_argument(
        "--static-zarr",
        type=Path,
        default=Path("/projects/prjs1858/static_hres_europe.zarr"),
        help="Static HRES zarr (sliced to region)"
    )
    parser.add_argument(
        "--hres-zarr",
        type=Path,
        default=Path("/projects/prjs1858/hres_europe_2018_2020.zarr"),
        help="HRES zarr to compare grid with"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_images/static_verification"),
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Verify Static and HRES Grid Alignment")
    print("=" * 60)
    
    # Load datasets
    print(f"\nLoading static: {args.static_zarr}")
    ds_static = xr.open_zarr(args.static_zarr, consolidated=True)
    
    print(f"Loading HRES: {args.hres_zarr}")
    ds_hres = xr.open_zarr(args.hres_zarr, consolidated=True)
    
    # Print dataset info
    print(f"\nStatic variables: {list(ds_static.data_vars)}")
    print(f"HRES variables: {list(ds_hres.data_vars)}")
    
    # Verify alignment
    aligned = verify_grid_alignment(ds_static, ds_hres)
    
    # Get extent for plotting
    lat_name = "latitude" if "latitude" in ds_static.coords else "lat"
    lon_name = "longitude" if "longitude" in ds_static.coords else "lon"
    lat_vals = ds_static[lat_name].values
    lon_vals = ds_static[lon_name].values
    extent = (lon_vals.min(), lon_vals.max(), lat_vals.min(), lat_vals.max())
    
    # Plot static variables
    print(f"\n=== Plotting Static Variables ===")
    plot_static_variables(ds_static, args.output_dir, extent)
    
    print(f"\n{'='*60}")
    if aligned:
        print("✓ Verification complete! Grids are aligned.")
    else:
        print("✗ Verification failed! Grids are not aligned.")
    print(f"  Plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
