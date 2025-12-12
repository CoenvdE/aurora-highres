#!/usr/bin/env python3
"""Convert downloaded HRES GRIB files for a SINGLE YEAR to Zarr with region cropping.

This is a memory-efficient version that processes one year at a time.
Use merge_hres_zarr.py to combine yearly files after processing.

Usage:
    python examples/convert_hres_to_zarr_year.py --year 2018
    python examples/convert_hres_to_zarr_year.py --year 2019
    python examples/convert_hres_to_zarr_year.py --year 2020
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Region bounds (must match Aurora latents!)
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -30.0
DEFAULT_LON_MAX = 50.0



def iterate_timesteps(year: int) -> Iterator[datetime]:
    """Generate all 6-hourly timesteps for a single year."""
    current = datetime(year, 1, 1, 0, 0)
    end = datetime(year, 12, 31, 18, 0)
    delta = timedelta(hours=6)
    while current <= end:
        yield current
        current += delta


def find_grib_files(
    hres_dir: Path,
    date: datetime,
    var_name: str,
    is_atmos: bool,
    hour: int | None = None,
) -> Path | None:
    """Find GRIB file for a specific variable and date."""
    day_dir = hres_dir / f"{date:%Y/%m/%d}"
    
    if is_atmos:
        pattern = f"atmos_{var_name}_{date:%Y-%m-%d}_{hour:02d}.grib"
    else:
        pattern = f"surf_{var_name}_{date:%Y-%m-%d}.grib"
    
    file_path = day_dir / pattern
    return file_path if file_path.exists() else None


def load_grib_surface(file_path: Path, var_name: str, hour: int) -> xr.DataArray | None:
    """Load a surface variable from GRIB at a specific hour."""
    try:
        ds = xr.open_dataset(str(file_path), engine="cfgrib")
        grib_names = {
            "2t": ["t2m", "2t"],
            "msl": ["msl"],
            "z": ["z", "orog"],
        }
        da = None
        for name in grib_names.get(var_name, [var_name]):
            if name in ds.data_vars:
                da = ds[name]
                break
        if da is None:
            print(f"  Warning: Variable {var_name} not found in {file_path}")
            return None
        
        # Select the correct timestep based on hour
        hour_to_idx = {0: 0, 6: 1, 12: 2, 18: 3}
        idx = hour_to_idx.get(hour, 0)
        
        if "step" in da.dims:
            da = da.isel(step=idx)
        elif "time" in da.dims and da.sizes.get("time", 1) > 1:
            da = da.isel(time=idx)
        
        return da
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def load_grib_atmos(file_path: Path, var_name: str) -> xr.DataArray | None:
    """Load an atmospheric variable from GRIB (has pressure levels)."""
    try:
        ds = xr.open_dataset(
            str(file_path),
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
        grib_names = {
            "t": ["t"],
            "q": ["q"],
            "z": ["z", "gh"],
        }
        for name in grib_names.get(var_name, [var_name]):
            if name in ds.data_vars:
                return ds[name]
        print(f"  Warning: Variable {var_name} not found in {file_path}")
        return None
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def crop_to_region(
    da: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.DataArray:
    """Crop data array to the specified region."""
    lat_coord = "latitude" if "latitude" in da.coords else "lat"
    lon_coord = "longitude" if "longitude" in da.coords else "lon"
    
    lons = da[lon_coord].values
    source_is_0_360 = lons.min() >= 0 and lons.max() > 180
    
    if source_is_0_360 and lon_min < 0:
        lon_min_adj = lon_min + 360
        lon_max_adj = lon_max if lon_max >= 0 else lon_max + 360
    else:
        lon_min_adj, lon_max_adj = lon_min, lon_max
    
    lat_decreasing = da[lat_coord].values[0] > da[lat_coord].values[-1]
    
    if lat_decreasing:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    
    if lon_min_adj > lon_max_adj:
        da1 = da.sel({lat_coord: lat_slice, lon_coord: slice(lon_min_adj, 360)})
        da2 = da.sel({lat_coord: lat_slice, lon_coord: slice(0, lon_max_adj)})
        da_cropped = xr.concat([da1, da2], dim=lon_coord)
    else:
        da_cropped = da.sel({
            lat_coord: lat_slice,
            lon_coord: slice(lon_min_adj, lon_max_adj),
        })
    
    new_lons = da_cropped[lon_coord].values.copy()
    new_lons = np.where(new_lons > 180, new_lons - 360, new_lons)
    
    sort_idx = np.argsort(new_lons)
    da_cropped = da_cropped.isel({lon_coord: sort_idx})
    da_cropped = da_cropped.assign_coords({lon_coord: new_lons[sort_idx]})
    
    return da_cropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HRES GRIB files for a single year to Zarr."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process (e.g., 2018)",
    )
    parser.add_argument(
        "--hres-dir",
        type=Path,
        default=Path("/projects/prjs1858/hres"),
        help="Directory containing downloaded GRIB files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/projects/prjs1858"),
        help="Output directory for Zarr files",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_LAT_MIN,
        help="Minimum latitude",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_LAT_MAX,
        help="Maximum latitude",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=DEFAULT_LON_MIN,
        help="Minimum longitude",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=DEFAULT_LON_MAX,
        help="Maximum longitude",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=None,
        help="Process only first N timesteps (for testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.hres_dir.exists():
        raise FileNotFoundError(f"HRES directory not found: {args.hres_dir}")
    
    output_zarr = args.output_dir / f"hres_europe_{args.year}.zarr"
    
    # Surface variables only
    surf_vars = ["2t", "msl"]
    
    # Get all timesteps for this year
    all_timesteps = list(iterate_timesteps(args.year))
    if args.max_timesteps:
        all_timesteps = all_timesteps[:args.max_timesteps]
    
    n_timesteps = len(all_timesteps)
    print(f"Processing year {args.year}: {n_timesteps} timesteps...")
    print(f"Region: lat=[{args.lat_min}, {args.lat_max}], lon=[{args.lon_min}, {args.lon_max}]")
    print(f"Output: {output_zarr}")
    
    # Track progress
    valid_count = 0
    zarr_initialized = False
    
    for i, ts in enumerate(all_timesteps):
        if i % 100 == 0:
            print(f"  Processing timestep {i}/{n_timesteps}: {ts}")
        
        date = ts.date()
        hour = ts.hour
        
        missing = False
        
        # Load surface variables
        surf_arrays = {}
        for var in surf_vars:
            grib_path = find_grib_files(args.hres_dir, datetime(date.year, date.month, date.day), var, is_atmos=False)
            if grib_path is None:
                missing = True
                break
            da = load_grib_surface(grib_path, var, hour)
            if da is None:
                missing = True
                break
            da_cropped = crop_to_region(da, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
            surf_arrays[var] = da_cropped
        
        if missing:
            print(f"  ⊘ Skipping {ts} (missing surface data)")
            continue
        
        
        # Create single-timestep dataset
        time_coord = np.array([ts], dtype="datetime64[ns]")
        data_vars = {}
        
        for var in surf_vars:
            da = surf_arrays[var].expand_dims("time").assign_coords(time=time_coord)
            data_vars[var] = da
        
        
        ds_timestep = xr.Dataset(data_vars)
        
        # Initialize Zarr on first valid timestep, then append
        if not zarr_initialized:
            # Add metadata on first write
            ds_timestep.attrs["region_lat_min"] = args.lat_min
            ds_timestep.attrs["region_lat_max"] = args.lat_max
            ds_timestep.attrs["region_lon_min"] = args.lon_min
            ds_timestep.attrs["region_lon_max"] = args.lon_max
            ds_timestep.attrs["year"] = args.year
            ds_timestep.attrs["created"] = datetime.now().isoformat()
            ds_timestep.attrs["source"] = "ECMWF HRES"
            ds_timestep.attrs["description"] = f"HRES analysis data for {args.year}"
            
            # Set up encoding for efficient access
            encoding = {}
            for var in ds_timestep.data_vars:
                encoding[var] = {
                    "dtype": "float32",
                    "chunks": tuple(
                        1 if dim == "time" else ds_timestep[var].sizes[dim]
                        for dim in ds_timestep[var].dims
                    ),
                }
            
            print(f"  Initializing Zarr store: {output_zarr}")
            ds_timestep.to_zarr(str(output_zarr), mode="w", encoding=encoding)
            zarr_initialized = True
        else:
            # Append subsequent timesteps
            ds_timestep.to_zarr(str(output_zarr), append_dim="time")
        
        valid_count += 1
        
        # Explicitly free memory
        del surf_arrays, ds_timestep, data_vars
    
    if valid_count == 0:
        print("No valid timesteps found!")
        return
    
    print(f"\n{'='*60}")
    print(f"✓ Year {args.year} conversion complete!")
    print(f"  Output: {output_zarr}")
    print(f"  Timesteps: {valid_count}")
    print(f"  Variables: {surf_vars}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
