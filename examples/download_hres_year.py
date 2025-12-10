"""Download multiple years of ECMWF HRES analysis data for Aurora examples.

This script extends `download_hres_sample.py` by looping over all days in a date
range (defaults to the full year 2021). Files are written under
`examples/downloads/hres/YYYY/MM/DD` so they can be consumed by
`load_hres_batch_example.py` or custom loaders.

Features:
- Automatic retry on network errors (up to 3 attempts with exponential backoff)
- Validates existing files are not corrupted before skipping
- Removes partial downloads on failure
- Resume capability: re-run to continue where it left off

Usage:
    python examples/download_hres_year.py --start 2018-01-01 --end 2021-12-31 \
        --out examples/downloads/hres
"""
from __future__ import annotations

import argparse
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable

import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, Timeout
from huggingface_hub import hf_hub_download

# Minimum expected file size in bytes (GRIB files should be at least 1KB)
MIN_FILE_SIZE = 1024
# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_BASE = 5  # seconds, will use exponential backoff

# Each variable has a numeric identifier in the RDA archive.
VAR_NUMS: Dict[str, str] = {
    "2t": "167",  # 2m temperature
    # "10u": "165",  # 10m u-component of wind
    # "10v": "166",  # 10m v-component of wind
    "msl": "151",  # Mean sea level pressure
    "t": "130",  # Temperature (atmospheric)
    # "u": "131",  # u-component of wind (atmospheric)
    # "v": "132",  # v-component of wind (atmospheric)
    # "q": "133",  # Specific humidity (atmospheric)
    # "z": "129",  # Geopotential (atmospheric only)
    # "slt": "043",  # Soil type
    # "lsm": "172",  # Land-sea mask
}

SURF_VARS = [
    "2t",  # 2m temperature (t2m)
    "msl",  # Mean sea level pressure
    # "z",  # Surface geopotential - REMOVED
]
ATMOS_VARS = [
    # "q",  # Specific humidity - REMOVED
    "t",  # Temperature on pressure levels
    # "z",  # Geopotential on pressure levels
]
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


def is_valid_grib_file(path: Path) -> bool:
    """Check if a file exists and appears to be a valid GRIB file."""
    if not path.exists():
        return False
    # Check minimum size
    if path.stat().st_size < MIN_FILE_SIZE:
        return False
    # Check GRIB magic bytes (GRIB files start with 'GRIB')
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            return magic == b"GRIB"
    except Exception:
        return False


def download_to(url: str, target: Path, chunk_size: int = 4 * 1024 * 1024) -> bool:
    """Download a file with retry logic and integrity checking.
    
    Returns True if download succeeded, False otherwise.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file already exists and is valid
    if target.exists():
        if is_valid_grib_file(target):
            return True
        else:
            # Remove corrupted/incomplete file
            print(f"    ⚠ Removing corrupted file: {target.name}")
            target.unlink()
    
    # Temporary file for atomic writes
    temp_target = target.with_suffix(target.suffix + ".tmp")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            with temp_target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size):
                    if chunk:
                        handle.write(chunk)
            
            # Verify the download is valid
            if is_valid_grib_file(temp_target):
                temp_target.rename(target)
                return True
            else:
                print(f"    ⚠ Downloaded file invalid (attempt {attempt}/{MAX_RETRIES})")
                temp_target.unlink(missing_ok=True)
                
        except (ChunkedEncodingError, ConnectionError, Timeout, requests.exceptions.RequestException) as e:
            print(f"    ⚠ Download error (attempt {attempt}/{MAX_RETRIES}): {type(e).__name__}")
            temp_target.unlink(missing_ok=True)
            
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY_BASE * (2 ** (attempt - 1))  # Exponential backoff
                print(f"    ⏳ Retrying in {delay}s...")
                time.sleep(delay)
    
    print(f"    ✗ Failed after {MAX_RETRIES} attempts: {target.name}")
    return False


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
        default=datetime(2018, 1, 1),
        help="First day to download (inclusive). Format YYYY-MM-DD",
    )
    parser.add_argument(
        "--end",
        type=datetime.fromisoformat,
        default=datetime(2020, 12, 31),
        help="Last day to download (inclusive). Format YYYY-MM-DD",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/projects/prjs1858/hres"),
        help="Output directory root",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload files even if they already exist",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing files, don't download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_day = args.start.date()
    end_day = args.end.date()
    if start_day > end_day:
        raise ValueError("--start must be before or equal to --end")

    if not args.validate_only:
        ensure_static_variables(args.out)

    total_success = 0
    total_failed = 0
    total_skipped = 0

    for day in daterange(start_day, end_day):
        downloads = collect_downloads(day, args.out)
        day_success = 0
        day_failed = 0
        day_skipped = 0
        
        print(f"{day:%Y-%m-%d}: {len(downloads)} files")
        
        for target, url in downloads.items():
            if args.validate_only:
                if is_valid_grib_file(target):
                    day_skipped += 1
                else:
                    print(f"  ✗ Missing/invalid: {target.relative_to(args.out)}")
                    day_failed += 1
                continue
                
            if target.exists() and not args.refresh:
                if is_valid_grib_file(target):
                    day_skipped += 1
                    continue
                # File exists but is invalid - will be re-downloaded
            
            print(f"  -> {target.relative_to(args.out)}")
            if download_to(url, target):
                day_success += 1
            else:
                day_failed += 1
        
        total_success += day_success
        total_failed += day_failed
        total_skipped += day_skipped
        
        if day_failed > 0:
            print(f"  Summary: {day_success} downloaded, {day_skipped} skipped, {day_failed} FAILED")
    
    print(f"\n{'='*60}")
    print(f"Total: {total_success} downloaded, {total_skipped} already present, {total_failed} failed")
    if total_failed > 0:
        print(f"⚠ {total_failed} files failed - re-run the script to retry")


if __name__ == "__main__":
    main()
