"""Flexible batch loading for Aurora that can predict any 6-hourly timestep."""

import numpy as np
import pandas as pd
import torch
import xarray as xr
from datetime import datetime, timedelta
from aurora import Batch, Metadata

VAR_MAP = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


def load_batch_for_timestep(
    target_time: datetime,
    zarr_path: str,
    static_path: str,
    dataset: xr.Dataset | None = None,
    static_dataset: xr.Dataset | None = None,
) -> Batch:
    """Construct a Batch to predict a specific target timestep.

    Aurora requires two consecutive 6-hourly timesteps as input (t-12h, t-6h)
    to predict the target time (t).

    Args:
        target_time: The datetime to predict (must be on 6-hour boundary: 00, 06, 12, 18)
        zarr_path: Path to ERA5 Zarr archive
        static_path: Path to static variables
        dataset: Optional pre-opened Zarr dataset
        static_dataset: Optional pre-opened static dataset

    Returns:
        Batch ready for Aurora inference that will predict target_time

    Example:
        # To predict 2020-01-01 18:00, loads:
        # - Input 1: 2020-01-01 06:00
        # - Input 2: 2020-01-01 12:00
        # → Prediction: 2020-01-01 18:00
    """
    # Validate target time is on 6-hour boundary
    if target_time.hour not in [0, 6, 12, 18]:
        raise ValueError(
            f"target_time must be on 6-hour boundary (00, 06, 12, 18), "
            f"got hour={target_time.hour}"
        )

    # Calculate the two input timesteps (12 hours and 6 hours before target)
    input_t0 = target_time - timedelta(hours=12)  # First input (t-12h)
    input_t1 = target_time - timedelta(hours=6)   # Second input (t-6h)

    ds = dataset or xr.open_zarr(zarr_path, consolidated=True)

    # Load the two input timesteps
    time_t0_str = input_t0.strftime("%Y-%m-%dT%H:%M:%S")
    time_t1_str = input_t1.strftime("%Y-%m-%dT%H:%M:%S")
    
    try:
        frame_t0 = ds.sel(time=time_t0_str, method="nearest").load()
        frame_t1 = ds.sel(time=time_t1_str, method="nearest").load()
    except Exception as e:
        raise ValueError(
            f"Failed to load inputs for target {target_time}: "
            f"tried to load {time_t0_str} and {time_t1_str}. Error: {e}"
        )

    # Load static dataset
    if static_dataset is not None:
        static = static_dataset
        should_close_static = False
    else:
        static = _open_static_dataset(static_path)
        should_close_static = True

    def _ensure_2d(data_array: xr.DataArray) -> torch.Tensor:
        values = data_array.values
        if values.ndim == 2:
            return torch.from_numpy(values)
        if values.ndim >= 3:
            squeezed = values.squeeze()
            if squeezed.ndim == 2:
                return torch.from_numpy(squeezed)
        raise ValueError(
            f"Static variable {data_array.name!r} must reduce to 2 dimensions, "
            f"but has shape {values.shape}."
        )

    def _stack_surf_var(name: str) -> torch.Tensor:
        """Stack two timesteps for a surface variable: (1, 2, lat, lon)."""
        arr_t0 = frame_t0[VAR_MAP[name]].values
        arr_t1 = frame_t1[VAR_MAP[name]].values
        return torch.from_numpy(np.stack([arr_t0, arr_t1], axis=0)[None])

    def _stack_atmos_var(name: str) -> torch.Tensor:
        """Stack two timesteps for an atmos variable: (1, 2, level, lat, lon)."""
        arr_t0 = frame_t0[VAR_MAP[name]].values
        arr_t1 = frame_t1[VAR_MAP[name]].values
        return torch.from_numpy(np.stack([arr_t0, arr_t1], axis=0)[None])

    try:
        return Batch(
            surf_vars={
                "2t": _stack_surf_var("2t"),
                "10u": _stack_surf_var("10u"),
                "10v": _stack_surf_var("10v"),
                "msl": _stack_surf_var("msl"),
            },
            static_vars={
                "z": _ensure_2d(static["z"]),
                "slt": _ensure_2d(static["slt"]),
                "lsm": _ensure_2d(static["lsm"]),
            },
            atmos_vars={
                "t": _stack_atmos_var("t"),
                "u": _stack_atmos_var("u"),
                "v": _stack_atmos_var("v"),
                "q": _stack_atmos_var("q"),
                "z": _stack_atmos_var("z"),
            },
            metadata=Metadata(
                lat=torch.from_numpy(frame_t1.latitude.values),
                lon=torch.from_numpy(frame_t1.longitude.values),
                # Time is the second input timestep (t-6h)
                # Prediction will be +6h → target_time
                time=(pd.to_datetime(frame_t1.time.values).to_pydatetime(),),
                atmos_levels=tuple(int(lvl) for lvl in frame_t1.level.values),
            ),
        )
    finally:
        if should_close_static:
            static.close()


def iterate_timesteps(start_date: datetime, end_date: datetime):
    """Generate all 6-hourly timesteps between start and end dates (inclusive).
    
    Args:
        start_date: First date to process (will start at 00:00)
        end_date: Last date to process (will end at 18:00)
        
    Yields:
        datetime objects for each 6-hourly timestep
        
    Example:
        >>> list(iterate_timesteps(datetime(2020, 1, 1), datetime(2020, 1, 2)))
        [datetime(2020, 1, 1, 0, 0),
         datetime(2020, 1, 1, 6, 0),
         datetime(2020, 1, 1, 12, 0),
         datetime(2020, 1, 1, 18, 0),
         datetime(2020, 1, 2, 0, 0),
         datetime(2020, 1, 2, 6, 0),
         datetime(2020, 1, 2, 12, 0),
         datetime(2020, 1, 2, 18, 0)]
    """
    current = datetime(start_date.year, start_date.month, start_date.day, 0, 0)
    end = datetime(end_date.year, end_date.month, end_date.day, 18, 0)
    delta = timedelta(hours=6)
    
    while current <= end:
        yield current
        current += delta


def _open_static_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except OSError:
        return xr.open_dataset(path, engine="scipy")


if __name__ == "__main__":
    # Example: Process all timesteps for a single day
    from datetime import datetime
    
    zarr_path = "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
    static_path = "examples/downloads/era5/static.nc"
    
    # Generate all timesteps for 2020-01-01
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 1)
    
    print(f"Processing all timesteps from {start} to {end}:")
    for timestep in iterate_timesteps(start, end):
        print(f"\nTarget: {timestep}")
        input_t0 = timestep - timedelta(hours=12)
        input_t1 = timestep - timedelta(hours=6)
        print(f"  Inputs: {input_t0} and {input_t1}")
        print(f"  → Predicts: {timestep}")
