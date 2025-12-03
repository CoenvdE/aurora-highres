
from pathlib import Path

import pandas as pd
import torch
import xarray as xr
from aurora import Batch, Metadata

def load_era_batch_example(device: torch.device, download_path: str = "examples/downloads/era5") -> Batch:
    download_path = Path(download_path)
    static_vars_ds = xr.open_dataset(download_path / "static.nc", engine="h5netcdf")
    surf_vars_ds = xr.open_dataset(download_path / "2020-01-01-surface-level.nc", engine="h5netcdf")
    atmos_vars_ds = xr.open_dataset(download_path / "2020-01-01-atmospheric.nc", engine="h5netcdf")

    batch = Batch(
        surf_vars={
            # First select the first two time points: 00:00 and 06:00. Afterwards, `[None]`
            # inserts a batch dimension of size one.
            "2t": torch.from_numpy(surf_vars_ds["t2m"].values[:2][None]),
            "10u": torch.from_numpy(surf_vars_ds["u10"].values[:2][None]),
            "10v": torch.from_numpy(surf_vars_ds["v10"].values[:2][None]),
            "msl": torch.from_numpy(surf_vars_ds["msl"].values[:2][None]),
        },
        static_vars={
            # The static variables are constant, so we just get them for the first time.
            "z": torch.from_numpy(static_vars_ds["z"].values[0]),
            "slt": torch.from_numpy(static_vars_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_vars_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_vars_ds["t"].values[:2][None]),
            "u": torch.from_numpy(atmos_vars_ds["u"].values[:2][None]),
            "v": torch.from_numpy(atmos_vars_ds["v"].values[:2][None]),
            "q": torch.from_numpy(atmos_vars_ds["q"].values[:2][None]),
            "z": torch.from_numpy(atmos_vars_ds["z"].values[:2][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_vars_ds.latitude.values),
            lon=torch.from_numpy(surf_vars_ds.longitude.values),
            # Converting to `datetime64[s]` ensures that the output of `tolist()` gives
            # `datetime.datetime`s. Note that this needs to be a tuple of length one:
            # one value for every batch element. Select element 1, corresponding to time
            # 06:00.
            time=(surf_vars_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_vars_ds.pressure_level.values),
        ),
    )
    batch = batch.to(device)
    return batch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = load_era_batch_example(device)
    print(batch)

# usage:
# python examples/load_era_batch_example.py