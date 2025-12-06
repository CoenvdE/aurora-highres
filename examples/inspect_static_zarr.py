#!/usr/bin/env python3
"""Inspect Aurora static variables stored in Zarr format.

This script provides detailed information about the static variables:
- Dataset structure and dimensions
- Variable statistics (min, max, mean, std)
- Data type and memory usage
- Optional visualization

Usage:
    python examples/inspect_static_zarr.py examples/downloads/static_hres.zarr
    python examples/inspect_static_zarr.py examples/downloads/static_hres.zarr --plot
    python examples/inspect_static_zarr.py examples/downloads/static_hres.zarr --plot --save-dir plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


def inspect_dataset(zarr_path: Path) -> xr.Dataset:
    """Open and inspect the Zarr dataset.
    
    Args:
        zarr_path: Path to the Zarr store
        
    Returns:
        Opened xarray Dataset
    """
    print("=" * 70)
    print("AURORA STATIC VARIABLES INSPECTOR")
    print("=" * 70)
    print(f"\nZarr path: {zarr_path}")
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")
    
    ds = xr.open_zarr(zarr_path, consolidated=True)
    
    # Basic info
    print("\n" + "-" * 70)
    print("DATASET STRUCTURE")
    print("-" * 70)
    print(f"\nDimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars)}")
    print(f"Coordinates: {list(ds.coords)}")
    
    # Dataset attributes
    if ds.attrs:
        print(f"\nGlobal attributes:")
        for key, val in ds.attrs.items():
            print(f"  {key}: {val}")
    
    return ds


def inspect_coordinates(ds: xr.Dataset) -> None:
    """Inspect coordinate arrays."""
    print("\n" + "-" * 70)
    print("COORDINATES")
    print("-" * 70)
    
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        values = coord.values
        
        print(f"\n{coord_name}:")
        print(f"  Shape: {coord.shape}")
        print(f"  Dtype: {coord.dtype}")
        print(f"  Range: [{values.min():.6f}, {values.max():.6f}]")
        
        if len(values) > 1:
            # Check spacing
            diffs = np.diff(values)
            print(f"  Spacing: mean={np.mean(diffs):.6f}, std={np.std(diffs):.6e}")
            print(f"  Monotonic: {np.all(diffs > 0) or np.all(diffs < 0)}")
        
        # Show first and last few values
        if len(values) > 6:
            print(f"  First 3: {values[:3]}")
            print(f"  Last 3: {values[-3:]}")
        else:
            print(f"  Values: {values}")


def inspect_variables(ds: xr.Dataset) -> None:
    """Inspect data variables with statistics."""
    print("\n" + "-" * 70)
    print("DATA VARIABLES")
    print("-" * 70)
    
    total_memory = 0
    
    for var_name in ds.data_vars:
        var = ds[var_name]
        values = var.values
        
        print(f"\n{var_name}:")
        print(f"  Dimensions: {var.dims}")
        print(f"  Shape: {var.shape}")
        print(f"  Dtype: {var.dtype}")
        
        # Memory usage
        memory_mb = values.nbytes / 1024 / 1024
        total_memory += memory_mb
        print(f"  Memory: {memory_mb:.2f} MB")
        
        # Statistics
        valid_values = values[~np.isnan(values)]
        print(f"  Statistics:")
        print(f"    Min: {np.min(valid_values):.6f}")
        print(f"    Max: {np.max(valid_values):.6f}")
        print(f"    Mean: {np.mean(valid_values):.6f}")
        print(f"    Std: {np.std(valid_values):.6f}")
        print(f"    NaN count: {np.sum(np.isnan(values)):,}")
        
        # Unique values (for categorical variables like soil type)
        if var_name in ("slt",):
            unique = np.unique(valid_values)
            if len(unique) <= 20:
                print(f"    Unique values: {unique}")
        
        # Variable attributes
        if var.attrs:
            print(f"  Attributes:")
            for key, val in var.attrs.items():
                print(f"    {key}: {val}")
    
    print(f"\nTotal memory usage: {total_memory:.2f} MB")


def inspect_chunks(ds: xr.Dataset) -> None:
    """Inspect chunking information."""
    print("\n" + "-" * 70)
    print("CHUNKING")
    print("-" * 70)
    
    for var_name in ds.data_vars:
        var = ds[var_name]
        if var.chunks:
            print(f"\n{var_name}:")
            for dim, chunks in zip(var.dims, var.chunks):
                print(f"  {dim}: {chunks[:3]}... (n_chunks={len(chunks)})")
        else:
            print(f"\n{var_name}: Not chunked")


def plot_variables(ds: xr.Dataset, save_dir: Path | None = None) -> None:
    """Create visualization of static variables."""
    try:
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        print("\nWarning: matplotlib and/or cartopy not available for plotting")
        return
    
    print("\n" + "-" * 70)
    print("GENERATING PLOTS")
    print("-" * 70)
    
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get coordinates
    lats = ds.coords["latitude"].values
    lons = ds.coords["longitude"].values
    
    # Convert lon from 0-360 to -180-180 for plotting
    lon_plot = np.where(lons > 180, lons - 360, lons)
    sort_idx = np.argsort(lon_plot)
    lon_plot = lon_plot[sort_idx]
    
    for var_name in ds.data_vars:
        print(f"  Plotting {var_name}...")
        
        values = ds[var_name].values[:, sort_idx]  # Reorder longitudes
        
        fig, ax = plt.subplots(
            figsize=(14, 8),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        
        # Choose colormap based on variable
        if var_name == "z":
            cmap = "terrain"
            vmin, vmax = None, None
        elif var_name == "lsm":
            cmap = "Blues_r"
            vmin, vmax = 0, 1
        elif var_name == "slt":
            cmap = "tab10"
            vmin, vmax = 0, 7
        else:
            cmap = "viridis"
            vmin, vmax = None, None
        
        im = ax.pcolormesh(
            lon_plot, lats, values,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_global()
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        if var_name in ds.data_vars and "units" in ds[var_name].attrs:
            cbar.set_label(ds[var_name].attrs["units"])
        
        # Title
        long_name = ds[var_name].attrs.get("long_name", var_name)
        ax.set_title(f"{long_name} ({var_name})", fontsize=14)
        
        plt.tight_layout()
        
        if save_dir:
            fig_path = save_dir / f"static_{var_name}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"    Saved: {fig_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    print("  Done plotting!")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Aurora static variables in Zarr format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "zarr_path",
        type=Path,
        help="Path to the static variables Zarr store",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory to save plots (if not specified, plots are shown interactively)",
    )
    
    args = parser.parse_args()
    
    # Inspect dataset
    ds = inspect_dataset(args.zarr_path)
    inspect_coordinates(ds)
    inspect_variables(ds)
    inspect_chunks(ds)
    
    # Optional plotting
    if args.plot:
        plot_variables(ds, args.save_dir)
    
    ds.close()
    
    print("\n" + "=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
