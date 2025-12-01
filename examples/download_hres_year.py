"""Download a full year of ECMWF HRES analysis data for Aurora examples.

This script extends `download_hres_sample.py` by looping over all days in a date
range (defaults to the full year 2021). Files are written under
`examples/downloads/hres/YYYY/MM/DD` so they can be consumed by
`load_hres_batch_example.py` or custom loaders.

Usage:
    python examples/download_hres_year.py --start 2021-01-01 --end 2021-12-31 \
        --out examples/downloads/hres
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

import requests
from huggingface_hub import hf_hub_download

# Each variable has a numeric identifier in the RDA archive.
VAR_NUMS: Dict[str, str] = {
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

SURF_VARS = ["2t", "10u", "10v", "msl", "z", "slt", "lsm"]
ATMOS_VARS = ["z", "t", "u", "v", "q"]
ATMOS_HOURS = [0, 6, 12, 18]


def daterange(start: date, end: date) -> Iterable[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def build_surface_url(day: date, var: str) -> str:
    return (
        "https://data.rda.ucar.edu/d113001/"
        f"ec.oper.an.sfc/{day:%Y%m}/ec.oper.an.sfc.128_{VAR_NUMS[var]}_{var}."
        f"regn1280sc.{day:%Y%m%d}.grb"
    )


def build_atmos_url(day: date, var: str, hour: int) -> str:
    prefix = "uv" if var in {"u", "v"} else "sc"
    return (
        "https://data.rda.ucar.edu/d113001/"
        f"ec.oper.an.pl/{day:%Y%m}/ec.oper.an.pl.128_{VAR_NUMS[var]}_{var}."
        f"regn1280{prefix}.{day:%Y%m%d}{hour:02d}.grb"
    )


def download_to(url: str, target: Path, chunk_size: int = 4 * 1024 * 1024) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                handle.write(chunk)


def ensure_static_variables(root: Path) -> None:
    static_dir = root / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id="microsoft/aurora",
        filename="aurora-0.1-static.pickle",
        cache_dir=str(static_dir),
    )


def collect_downloads(day: date, base_dir: Path) -> Dict[Path, str]:
    day_dir = base_dir / f"{day:%Y/%m/%d}"
    downloads: Dict[Path, str] = {}
    for var in SURF_VARS:
        target = day_dir / day.strftime(f"surf_{var}_%Y-%m-%d.grib")
        downloads[target] = build_surface_url(day, var)
    for var in ATMOS_VARS:
        for hour in ATMOS_HOURS:
            target = day_dir / \
                day.strftime(f"atmos_{var}_%Y-%m-%d_{hour:02d}.grib")
            downloads[target] = build_atmos_url(day, var, hour)
    return downloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=datetime.fromisoformat,
        default=datetime(2021, 1, 1),
        help="First day to download (inclusive). Format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        default=datetime(2021, 12, 31),
        help="Last day to download (inclusive). Format YYYY-MM-DD",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/downloads/hres"),
        help="Output directory root",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload files even if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_day = args.start.date()
    end_day = args.end.date()
    if start_day > end_day:
        raise ValueError("--start must be before or equal to --end")

    ensure_static_variables(args.out)

    for day in daterange(start_day, end_day):
        downloads = collect_downloads(day, args.out)
        print(f"{day:%Y-%m-%d}: {len(downloads)} files")
        for target, url in downloads.items():
            if target.exists() and not args.refresh:
                continue
            print(f"  -> {target.relative_to(args.out)}")
            download_to(url, target)


if __name__ == "__main__":
    main()
