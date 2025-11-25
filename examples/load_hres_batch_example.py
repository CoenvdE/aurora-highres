import numpy as np
import torch

from aurora import Batch, Metadata
import xarray as xr
from pathlib import Path
from datetime import datetime

def load_hres_batch_example(device: torch.device, download_path: str = "examples/downloads/hres") -> Batch:
    download_path = Path(download_path)
    # Load these pressure levels.
    levels = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
    date = datetime(2023, 1, 1)
    downloads = {
        download_path / date.strftime(f"surf_2t_%Y-%m-%d.grib"),
        download_path / date.strftime(f"surf_10u_%Y-%m-%d.grib"),
        download_path / date.strftime(f"surf_10v_%Y-%m-%d.grib"),
        download_path / date.strftime(f"surf_msl_%Y-%m-%d.grib"),
        download_path / date.strftime(f"atmos_t_%Y-%m-%d_00.grib"),
        download_path / date.strftime(f"atmos_t_%Y-%m-%d_06.grib"),
        download_path / date.strftime(f"atmos_u_%Y-%m-%d_00.grib"),
        download_path / date.strftime(f"atmos_u_%Y-%m-%d_06.grib"),
        download_path / date.strftime(f"atmos_v_%Y-%m-%d_00.grib"),
        download_path / date.strftime(f"atmos_v_%Y-%m-%d_06.grib"),
        download_path / date.strftime(f"atmos_q_%Y-%m-%d_00.grib"),
        download_path / date.strftime(f"atmos_q_%Y-%m-%d_06.grib"),
        download_path / date.strftime(f"atmos_z_%Y-%m-%d_00.grib"),
        download_path / date.strftime(f"atmos_z_%Y-%m-%d_06.grib"),
    }       


    def load_surf(v: str, v_in_file: str) -> torch.Tensor:
        """Load the downloaded surface-level or static variable `v` for hours 00 and 06."""
        ds = xr.open_dataset(download_path / date.strftime(f"surf_{v}_%Y-%m-%d.grib"), engine="cfgrib")
        data = ds[v_in_file].values[:2]  # Use hours 00 and 06.
        data = data[None]  # Insert a batch dimension.
        return torch.from_numpy(data)


    def load_atmos(v: str) -> torch.Tensor:
        """Load the downloaded atmospheric variable `v` for hours 00 and 06."""
        ds_00 = xr.open_dataset(
            download_path / date.strftime(f"atmos_{v}_%Y-%m-%d_00.grib"), engine="cfgrib"
        )
        ds_06 = xr.open_dataset(
            download_path / date.strftime(f"atmos_{v}_%Y-%m-%d_06.grib"), engine="cfgrib"
        )
        # Select the right pressure levels.
        ds_00 = ds_00[v].sel(isobaricInhPa=list(levels))
        ds_06 = ds_06[v].sel(isobaricInhPa=list(levels))
        data = np.stack((ds_00.values, ds_06.values), axis=0)
        data = data[None]  # Insert a batch dimension.
        return torch.from_numpy(data)


    # Extract the latitude and longitude from an arbitrary downloaded file.
    ds = xr.open_dataset(next(iter(downloads)), engine="cfgrib")

    batch = Batch(
        surf_vars={
            "2t": load_surf("2t", "t2m"),
            "10u": load_surf("10u", "u10"),
            "10v": load_surf("10v", "v10"),
            "msl": load_surf("msl", "msl"),
        },
        # Set the static variables _after_ regridding. The static variables
        # downloaded from HuggingFace are already at the right resolution.
        static_vars={},
        atmos_vars={
            "t": load_atmos("t"),
            "u": load_atmos("u"),
            "v": load_atmos("v"),
            "q": load_atmos("q"),
            "z": load_atmos("z"),
        },
        metadata=Metadata(
            lat=torch.from_numpy(ds.latitude.values),
            lon=torch.from_numpy(ds.longitude.values),
            time=(date.replace(hour=6),),
            atmos_levels=levels,
        ),
    )
    batch = batch.to(device)
    return batch

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = load_hres_batch_example(device)
    print(batch)

# usage:
# python examples/load_hres_batch_example.py