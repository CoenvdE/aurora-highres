from pathlib import Path
import requests
import xarray as xr
import cdsapi

# Data will be downloaded here.
download_path = Path("examples/downloads/hres")
from datetime import datetime
from pathlib import Path


# Day to download. This will download all times for that day.
date = datetime(2023, 1, 1)

# Each variable has a number associated with it. This is the number that will be used in
# the RDA request.
var_nums = {
    "2t": "167",  # 2m temperature
    "10u": "165",  # 10m u-component of wind
    "10v": "166",  # 10m v-component of wind
    "msl": "151",  # Mean sea level pressure
    "t": "130",  # Temperature
    "u": "131",  # u-component of wind (atmospheric)
    "v": "132",  # v-component of wind (atmospheric)
    "q": "133",  # Specific humidity (atmospheric)
    "z": "129",  # Geopotential
    "slt": "043",  # Soil type
    "lsm": "172",  # Land-sea mask
}

# Construct the URLs to download the data from.
downloads: dict[Path, str] = {}
for v in ["2t", "10u", "10v", "msl", "z", "slt", "lsm"]:
    downloads[download_path / date.strftime(f"surf_{v}_%Y-%m-%d.grib")] = (
        f"https://data.rda.ucar.edu/d113001/"
        f"ec.oper.an.sfc/{date.year}{date.month:02d}/ec.oper.an.sfc.128_{var_nums[v]}_{v}."
        f"regn1280sc.{date.year}{date.month:02d}{date.day:02d}.grb"
    )
for v in ["z", "t", "u", "v", "q"]:
    for hour in [0, 6, 12, 18]:
        prefix = "uv" if v in {"u", "v"} else "sc"
        downloads[download_path / date.strftime(f"atmos_{v}_%Y-%m-%d_{hour:02d}.grib")] = (
            f"https://data.rda.ucar.edu/d113001/"
            f"ec.oper.an.pl/{date.year}{date.month:02d}/ec.oper.an.pl.128_{var_nums[v]}_{v}."
            f"regn1280{prefix}.{date.year}{date.month:02d}{date.day:02d}{hour:02d}.grb"
        )

# Perform the downloads.
for target, source in downloads.items():
    if not target.exists():
        print(f"Downloading {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(source)
        response.raise_for_status()
        with open(target, "wb") as f:
            f.write(response.content)
print("Downloads finished!")

from huggingface_hub import hf_hub_download

# Download the static variables from HuggingFace.
static_path = hf_hub_download(
    repo_id="microsoft/aurora",
    filename="aurora-0.1-static.pickle",
    cache_dir=download_path,
)
print("Static variables downloaded!")

# Download the surface-level variables.
# usage:
# python examples/init_exploring/download_hres_sample.py