
import numpy as np
import pandas as pd
import torch
import xarray as xr
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


def load_batch_from_zarr(
    zarr_path: str,
    static_path: str,
    date_str: str,
    dataset: xr.Dataset | None = None,
    static_dataset: xr.Dataset | None = None,
) -> Batch:
    """Construct a Batch from ERA5 stored in Zarr/NetCDF.

    Aurora requires two consecutive timesteps as input (t-1, t-0) to predict
    6 hours ahead. This function loads 00:00 and 06:00 for the given date,
    so the prediction will be for 12:00.
    """

    ds = dataset or xr.open_zarr(zarr_path, consolidated=True)

    # Load two input timesteps: 00:00 and 06:00 -> prediction at 12:00
    time_t0 = f"{date_str}T00:00:00"
    time_t1 = f"{date_str}T06:00:00"
    try:
        frame_t0 = ds.sel(time=time_t0, method="nearest").load()
        frame_t1 = ds.sel(time=time_t1, method="nearest").load()
    except Exception as e:
        raise ValueError(f"Failed to load {date_str}: {e}")

    if static_dataset is not None:
        static = static_dataset
        should_close_static = False
    else:
        static = _open_static_dataset(static_path)
        should_close_static = True
    should_close_static = static_dataset is None

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

    # Construct Batch with two input timesteps (00:00, 06:00)
    # Metadata.time is set to the second timestep (06:00) per Aurora convention
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
                # Time is the second input timestep (06:00); prediction will be +6h (12:00)
                time=(pd.to_datetime(frame_t1.time.values).to_pydatetime(),),
                atmos_levels=tuple(int(lvl) for lvl in frame_t1.level.values),
            ),
        )
    finally:
        if should_close_static:
            static.close()


def _open_static_dataset(path: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except OSError:
        return xr.open_dataset(path, engine="scipy")


if __name__ == "__main__":
    # Example usage:
    zarr_path = "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
    static_path = "path/to/static_data.nc"
    date_str = "2023-01-15"

    batch = load_batch_from_zarr(zarr_path, static_path, date_str)
    print(batch)
# Example usage:
