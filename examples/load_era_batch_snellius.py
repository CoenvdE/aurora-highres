
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
    """Construct a Batch from ERA5 stored in Zarr/NetCDF."""

    ds = dataset or xr.open_zarr(zarr_path, consolidated=True)

    target_time = f"{date_str}T12:00:00"
    try:
        frame = ds.sel(time=target_time, method="nearest").load()
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

    # 4. Construct Batch
    try:
        return Batch(
            surf_vars={
                "2t": torch.from_numpy(frame[VAR_MAP["2t"]].values[None, None]),
                "10u": torch.from_numpy(frame[VAR_MAP["10u"]].values[None, None]),
                "10v": torch.from_numpy(frame[VAR_MAP["10v"]].values[None, None]),
                "msl": torch.from_numpy(frame[VAR_MAP["msl"]].values[None, None]),
            },
            static_vars={
                "z": _ensure_2d(static["z"]),
                "slt": _ensure_2d(static["slt"]),
                "lsm": _ensure_2d(static["lsm"]),
            },
            atmos_vars={
                "t": torch.from_numpy(frame[VAR_MAP["t"]].values[None, None]),
                "u": torch.from_numpy(frame[VAR_MAP["u"]].values[None, None]),
                "v": torch.from_numpy(frame[VAR_MAP["v"]].values[None, None]),
                "q": torch.from_numpy(frame[VAR_MAP["q"]].values[None, None]),
                "z": torch.from_numpy(frame[VAR_MAP["z"]].values[None, None]),
            },
            metadata=Metadata(
                lat=torch.from_numpy(frame.latitude.values),
                lon=torch.from_numpy(frame.longitude.values),
                time=(pd.to_datetime(frame.time.values).to_pydatetime(),),
                atmos_levels=tuple(int(lvl) for lvl in frame.level.values),
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
