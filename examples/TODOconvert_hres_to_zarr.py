#!/usr/bin/env python3
"""Convert downloaded HRES GRIB files to a single Zarr dataset with region cropping.

This script:
1. Reads GRIB files downloaded by download_hres_year.py
2. Crops to the specified region (Europe by default)
3. Aligns timestamps with Aurora latents
4. Saves as a single Zarr dataset optimized for DataLoader

Output format:
    hres_europe.zarr/
    ├── time                # (n_timesteps,) datetime64
    ├── 2t                  # (time, lat, lon) - 2m temperature
    ├── msl                 # (time, lat, lon) - mean sea level pressure
    ├── z_surface           # (time, lat, lon) - surface geopotential
    ├── t                   # (time, level, lat, lon) - temperature
    ├── q                   # (time, level, lat, lon) - specific humidity
    └── z                   # (time, level, lat, lon) - geopotential
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import xarray as xr

# Region bounds (must match Aurora latents!)
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -30.0
DEFAULT_LON_MAX = 50.0


def iterate_timesteps(start_year: int, end_year: int) -> Iterator[datetime]:
    """Generate all 6-hourly timesteps for the year range."""
    current = datetime(start_year, 1, 1, 0, 0)
    end = datetime(end_year, 12, 31, 18, 0)
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


def load_grib_surface(file_path: Path, var_name: str) -> xr.DataArray | None:
    """Load a surface variable from GRIB."""
    try:
        ds = xr.open_dataset(str(file_path), engine="cfgrib")
        # GRIB variable names may differ
        grib_names = {
            "2t": ["t2m", "2t"],
            "msl": ["msl"],
            "z": ["z", "orog"],
        }
        for name in grib_names.get(var_name, [var_name]):
            if name in ds.data_vars:
                return ds[name]
        print(f"  Warning: Variable {var_name} not found in {file_path}")
        return None
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        return None


def load_grib_atmos(file_path: Path, var_name: str) -> xr.DataArray | None:
    """Load an atmospheric variable from GRIB (has pressure levels)."""
    try:
        # For atmospheric data, we need to specify filter_by_keys
        ds = xr.open_dataset(
            str(file_path),
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )
        grib_names = {
            "t": ["t"],
            "q": ["q"],
            "z": ["z", "gh"],  # geopotential or geopotential height
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
    # Handle different coordinate names
    lat_coord = "latitude" if "latitude" in da.coords else "lat"
    lon_coord = "longitude" if "longitude" in da.coords else "lon"
    
    # Handle longitude wrapping (-30 to 50 needs conversion from 0-360 or vice versa)
    lons = da[lon_coord].values
    if lon_min < 0 and lons.min() >= 0:
        # Convert negative lons to 0-360
        lon_min_adj = lon_min + 360
        lon_max_adj = lon_max if lon_max >= 0 else lon_max + 360
    else:
        lon_min_adj, lon_max_adj = lon_min, lon_max
    
    # Check if latitude is decreasing (90 to -90)
    lat_decreasing = da[lat_coord].values[0] > da[lat_coord].values[-1]
    
    if lat_decreasing:
        lat_slice = slice(lat_max, lat_min)  # Reversed for decreasing
    else:
        lat_slice = slice(lat_min, lat_max)
    
    # Handle wrapping around 0 degrees longitude
    if lon_min_adj > lon_max_adj:
        # Need to concatenate two slices
        da1 = da.sel({lon_coord: slice(lon_min_adj, 360)})
        da2 = da.sel({lon_coord: slice(0, lon_max_adj)})
        da_cropped = xr.concat([da1, da2], dim=lon_coord)
    else:
        da_cropped = da.sel({
            lat_coord: lat_slice,
            lon_coord: slice(lon_min_adj, lon_max_adj),
        })
    
    return da_cropped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HRES GRIB files to Zarr with region cropping."
    )
    parser.add_argument(
        "--hres-dir",
        type=Path,
        default=Path("examples/downloads/hres"),
        help="Directory containing downloaded GRIB files",
    )
    parser.add_argument(
        "--output-zarr",
        type=Path,
        default=Path("examples/hres_europe_2018_2020.zarr"),
        help="Output Zarr path",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="Start year",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="End year",
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
    
    # Surface and atmospheric variables
    surf_vars = ["2t", "msl", "z"]
    atmos_vars = ["q", "t", "z"]
    
    # Get all timesteps
    all_timesteps = list(iterate_timesteps(args.start_year, args.end_year))
    if args.max_timesteps:
        all_timesteps = all_timesteps[:args.max_timesteps]
    
    n_timesteps = len(all_timesteps)
    print(f"Processing {n_timesteps} timesteps...")
    print(f"Region: lat=[{args.lat_min}, {args.lat_max}], lon=[{args.lon_min}, {args.lon_max}]")
    
    # Collect data per variable
    surface_data = {var: [] for var in surf_vars}
    atmos_data = {var: [] for var in atmos_vars}
    valid_timesteps = []
    
    for i, ts in enumerate(all_timesteps):
        if i % 100 == 0:
            print(f"  Processing timestep {i}/{n_timesteps}: {ts}")
        
        date = ts.date()
        hour = ts.hour
        
        # Check if we have all required files
        missing = False
        
        # Load surface variables (daily files)
        surf_arrays = {}
        for var in surf_vars:
            grib_path = find_grib_files(args.hres_dir, datetime(date.year, date.month, date.day), var, is_atmos=False)
            if grib_path is None:
                missing = True
                break
            da = load_grib_surface(grib_path, var)
            if da is None:
                missing = True
                break
            # Crop to region
            da_cropped = crop_to_region(da, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
            surf_arrays[var] = da_cropped
        
        if missing:
            print(f"  ⊘ Skipping {ts} (missing surface data)")
            continue
        
        # Load atmospheric variables (hourly files)
        atmos_arrays = {}
        for var in atmos_vars:
            grib_path = find_grib_files(args.hres_dir, datetime(date.year, date.month, date.day), var, is_atmos=True, hour=hour)
            if grib_path is None:
                missing = True
                break
            da = load_grib_atmos(grib_path, var)
            if da is None:
                missing = True
                break
            da_cropped = crop_to_region(da, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
            atmos_arrays[var] = da_cropped
        
        if missing:
            print(f"  ⊘ Skipping {ts} (missing atmospheric data)")
            continue
        
        # Add to lists
        for var in surf_vars:
            surface_data[var].append(surf_arrays[var])
        for var in atmos_vars:
            atmos_data[var].append(atmos_arrays[var])
        valid_timesteps.append(ts)
    
    if not valid_timesteps:
        print("No valid timesteps found!")
        return
    
    print(f"\n✓ Loaded {len(valid_timesteps)} timesteps successfully")
    
    # Create xarray dataset
    print("Creating xarray Dataset...")
    
    # Time coordinate
    time_coord = np.array(valid_timesteps, dtype="datetime64[ns]")
    
    data_vars = {}
    
    # Stack surface variables
    for var in surf_vars:
        stacked = xr.concat(surface_data[var], dim="time")
        stacked = stacked.assign_coords(time=time_coord)
        # Rename surface z to avoid conflict
        var_name = "z_surface" if var == "z" else var
        data_vars[var_name] = stacked
    
    # Stack atmospheric variables
    for var in atmos_vars:
        stacked = xr.concat(atmos_data[var], dim="time")
        stacked = stacked.assign_coords(time=time_coord)
        data_vars[var] = stacked
    
    ds = xr.Dataset(data_vars)
    
    # Add metadata
    ds.attrs["region_lat_min"] = args.lat_min
    ds.attrs["region_lat_max"] = args.lat_max
    ds.attrs["region_lon_min"] = args.lon_min
    ds.attrs["region_lon_max"] = args.lon_max
    ds.attrs["created"] = datetime.now().isoformat()
    ds.attrs["source"] = "ECMWF HRES"
    
    print(f"Dataset: {ds}")
    
    # Set up chunking for efficient access
    chunks = {}
    for var in ds.data_vars:
        dims = ds[var].dims
        if "level" in dims or "isobaricInhPa" in dims:
            # Atmospheric: chunk by time, keep levels together
            chunks[var] = {"time": 1}
        else:
            # Surface: chunk by time
            chunks[var] = {"time": 1}
    
    # Save to Zarr
    print(f"Saving to {args.output_zarr}...")
    
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            "dtype": "float32",
            "chunks": tuple(
                1 if dim == "time" else ds[var].sizes[dim]
                for dim in ds[var].dims
            ),
        }
    
    ds.to_zarr(str(args.output_zarr), mode="w", encoding=encoding)
    
    print(f"\n{'='*60}")
    print("✓ Conversion complete!")
    print(f"  Output: {args.output_zarr}")
    print(f"  Timesteps: {len(valid_timesteps)}")
    print(f"  Variables: {list(ds.data_vars)}")
    print(f"{'='*60}")
    
    # Print usage example
    print(f"""
Example usage:

    import xarray as xr
    ds = xr.open_zarr("{args.output_zarr}")
    
    # Get first timestep
    t2m = ds["2t"].isel(time=0).values
    temp = ds.t.isel(time=0).values  # (level, lat, lon)
    
    # DataLoader usage (matches latents indexing)
    for idx in range(len(ds.time)):
        t2m = ds["2t"].isel(time=idx).values
        temp = ds.t.isel(time=idx).values
""")


if __name__ == "__main__":
    main()

# Usage:
#
# After downloading GRIB files with download_hres_year.py:
#   python examples/convert_hres_to_zarr.py \
#       --hres-dir examples/downloads/hres \
#       --output-zarr examples/hres_europe_2018_2020.zarr \
#       --start-year 2018 --end-year 2020
#
# Test with first 10 timesteps:
#   python examples/convert_hres_to_zarr.py \
#       --max-timesteps 10
