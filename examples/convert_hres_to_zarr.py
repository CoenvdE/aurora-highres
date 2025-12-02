"""Convert ECMWF HRES GRIB downloads into a chunked Zarr dataset.

This script complements ``examples/download_hres_year.py`` by walking the
resulting ``surf_*`` and ``atmos_*`` files under a download directory,
normalising the dimensions, and writing a consolidated Xarray Dataset to a Zarr
store suitable for model training.

Example usage::

    python examples/convert_hres_to_zarr.py \
        --input examples/downloads/hres \
        --output data/hres_2018_2020.zarr \
        --start 2018-01-01 --end 2020-12-31

The script keeps operations lazy via Dask (as exposed by Xarray) so the write
step streams from GRIB into Zarr without loading the full dataset in memory.
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import xarray as xr

SURFACE_VAR_MAP: Dict[str, str] = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "msl": "msl",
}

ATMOS_VAR_MAP: Dict[str, str] = {
    "t": "t",
    "u": "u",
    "v": "v",
    "q": "q",
    "z": "z",
}

DEFAULT_ATMOS_LEVELS: Sequence[int] = (
    1000,
    925,
    850,
    700,
    600,
    500,
    400,
    300,
    250,
    200,
    150,
    100,
    50,
)

UNWANTED_COORDS = {
    "surface",
    "heightAboveGround",
    "depthBelowLand",
    "depthBelowLandLayer",
    "sigma",
    "step",
    "valid_time",
    "number",
    "expver",
    "forecastReferenceTime",
    "forecastReferenceTimeSince",
    "methodNumber",
}


def parse_date_arg(value: str) -> date:
    try:
        return datetime.fromisoformat(value).date()
    except ValueError as exc:  # pragma: no cover - defensive parsing guard
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected YYYY-MM-DD."
        ) from exc


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def collect_surface_files(
    root: Path, var: str, start: Optional[date], end: Optional[date]
) -> List[Path]:
    pattern = f"surf_{var}_*.grib"
    candidates = []
    for path in sorted(root.rglob(pattern)):
        if path.suffix != ".grib":
            continue
        try:
            parts = path.stem.split("_")
            file_date = datetime.strptime(parts[-1], "%Y-%m-%d").date()
        except (ValueError, IndexError):
            logging.debug(
                "Skipping unrecognised surface file name %s", path.name)
            continue
        if start and file_date < start:
            continue
        if end and file_date > end:
            continue
        candidates.append((file_date, path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def collect_atmos_files(
    root: Path,
    var: str,
    hours: Sequence[int],
    start: Optional[date],
    end: Optional[date],
) -> List[Path]:
    pattern = f"atmos_{var}_*.grib"
    allow_hours = set(hours)
    candidates = []
    for path in sorted(root.rglob(pattern)):
        if path.suffix != ".grib":
            continue
        try:
            parts = path.stem.split("_")
            file_dt = datetime.strptime(
                f"{parts[-2]}_{parts[-1]}", "%Y-%m-%d_%H")
        except (ValueError, IndexError):
            logging.debug(
                "Skipping unrecognised atmospheric file name %s", path.name)
            continue
        if allow_hours and file_dt.hour not in allow_hours:
            continue
        if start and file_dt.date() < start:
            continue
        if end and file_dt.date() > end:
            continue
        candidates.append((file_dt, path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def unfold_step_dimension(
    array: xr.DataArray, fallback_time: datetime
) -> xr.DataArray:
    if "step" not in array.dims:
        return array
    item = array
    has_time_dim = "time" in item.dims
    if has_time_dim:
        item = item.rename({"time": "reference_time"})
        base = item["reference_time"]
    else:
        base = None
    step = item["step"]
    if "valid_time" in item.coords:
        valid = item["valid_time"].values
    elif base is not None:
        base_b, step_b = xr.broadcast(base, step)
        valid = (base_b + step_b).values
    else:
        base_value = np.datetime64(fallback_time)
        valid = base_value + step.values
    if has_time_dim:
        item = item.stack(time=("reference_time", "step"))
    else:
        item = item.rename({"step": "time"})
    flat_valid = np.asarray(valid).reshape(-1)
    if not np.issubdtype(flat_valid.dtype, np.datetime64):
        flat_valid = np.asarray(flat_valid, dtype="timedelta64[ns]")
        flat_valid = np.datetime64(fallback_time) + flat_valid
    flat_valid = flat_valid.astype("datetime64[ns]")
    item = item.assign_coords(time=flat_valid)
    for coord in ("valid_time", "reference_time", "step"):
        if coord in item.coords:
            item = item.drop_vars(coord)
    return item


def squeeze_ecmwf_dims(
    array: xr.DataArray, keep_dims: Sequence[str]
) -> xr.DataArray:
    item = array
    for dim in ("expver", "number"):
        if dim in item.dims:
            item = item.isel({dim: 0}, drop=True)
    for dim in list(item.dims):
        if dim in keep_dims:
            continue
        if item.sizes.get(dim, 0) == 1:
            item = item.isel({dim: 0}, drop=True)
    return item


def drop_unwanted_coords(array: xr.DataArray) -> xr.DataArray:
    item = array
    for coord in list(item.coords):
        if coord in item.dims:
            continue
        if coord in UNWANTED_COORDS or coord.lower().startswith("grib"):
            item = item.drop_vars(coord, errors="ignore")
    item.attrs = {k: v for k, v in item.attrs.items()
                  if not k.startswith("GRIB_")}
    return item


def ensure_time_dimension(
    array: xr.DataArray, fallback_time: datetime
) -> xr.DataArray:
    item = array
    if "time" not in item.dims:
        time_value = np.array([np.datetime64(fallback_time)])
        item = item.expand_dims(time=time_value)
    else:
        time_coord = item["time"].values
        time_coord = np.asarray(time_coord)
        if time_coord.ndim == 0:
            time_coord = np.array([time_coord])
        item = item.assign_coords(time=time_coord.astype("datetime64[ns]"))
    return item


def prepare_surface_array(
    ds: xr.Dataset,
    cf_name: str,
    target_name: str,
    fallback_time: datetime,
    dtype: str,
) -> xr.DataArray:
    data = unfold_step_dimension(ds[cf_name], fallback_time)
    data = ensure_time_dimension(data, fallback_time)
    data = squeeze_ecmwf_dims(data, keep_dims=(
        "time", "latitude", "longitude"))
    data = drop_unwanted_coords(data)
    data = data.transpose("time", "latitude", "longitude")
    data.name = target_name
    if dtype:
        data = data.astype(dtype)
    return data


def prepare_atmos_array(
    ds: xr.Dataset,
    cf_name: str,
    target_name: str,
    fallback_time: datetime,
    levels: Sequence[int],
    dtype: str,
) -> Optional[xr.DataArray]:
    data = unfold_step_dimension(ds[cf_name], fallback_time)
    data = ensure_time_dimension(data, fallback_time)
    data = squeeze_ecmwf_dims(
        data, keep_dims=("time", "isobaricInhPa", "latitude", "longitude")
    )
    if "isobaricInhPa" not in data.dims:
        logging.warning(
            "Skipping %s: missing isobaricInhPa dimension", target_name)
        return None
    requested_levels = list(levels)
    available_levels = list(np.asarray(
        data["isobaricInhPa"].values, dtype=int))
    missing = [lvl for lvl in requested_levels if lvl not in available_levels]
    if missing:
        logging.warning(
            "Variable %s is missing levels: %s", target_name, ", ".join(
                map(str, missing))
        )
    present_levels = [
        lvl for lvl in requested_levels if lvl in available_levels]
    if not present_levels:
        logging.warning(
            "Skipping %s: none of the requested levels are available", target_name)
        return None
    data = data.sel(isobaricInhPa=present_levels)
    data = drop_unwanted_coords(data)
    data = data.transpose("time", "isobaricInhPa", "latitude", "longitude")
    data.name = target_name
    if dtype:
        data = data.astype(dtype)
    return data


def load_surface_series(
    root: Path,
    var: str,
    cf_var: str,
    start: Optional[date],
    end: Optional[date],
    dtype: str,
) -> Optional[xr.DataArray]:
    files = collect_surface_files(root, var, start, end)
    if not files:
        logging.info("Surface variable %s: no files found", var)
        return None
    arrays: List[xr.DataArray] = []
    for path in files:
        file_date = datetime.strptime(
            path.stem.split("_")[-1], "%Y-%m-%d").date()
        fallback = datetime.combine(file_date, datetime_time.min)
        logging.debug("Opening surface file %s", path)
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            chunks={},
            cache=False,
        )
        arrays.append(prepare_surface_array(ds, cf_var, var, fallback, dtype))
    combined = xr.concat(arrays, dim="time",
                         coords="minimal", compat="override")
    combined = combined.sortby("time")
    return combined


def load_atmos_series(
    root: Path,
    var: str,
    cf_var: str,
    hours: Sequence[int],
    start: Optional[date],
    end: Optional[date],
    levels: Sequence[int],
    dtype: str,
) -> Optional[xr.DataArray]:
    files = collect_atmos_files(root, var, hours, start, end)
    if not files:
        logging.info("Atmospheric variable %s: no files found", var)
        return None
    arrays: List[xr.DataArray] = []
    for path in files:
        parts = path.stem.split("_")
        file_dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y-%m-%d_%H")
        logging.debug("Opening atmospheric file %s", path)
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs={"indexpath": ""},
            chunks={},
            cache=False,
        )
        array = prepare_atmos_array(ds, cf_var, var, file_dt, levels, dtype)
        if array is not None:
            arrays.append(array)
    if not arrays:
        logging.warning("Atmospheric variable %s had no usable arrays", var)
        return None
    combined = xr.concat(arrays, dim="time",
                         coords="minimal", compat="override")
    combined = combined.sortby("time")
    return combined


def build_dataset(
    root: Path,
    start: Optional[date],
    end: Optional[date],
    hours: Sequence[int],
    levels: Sequence[int],
    dtype: str,
) -> xr.Dataset:
    data_vars: Dict[str, xr.DataArray] = {}

    for var, cf_var in SURFACE_VAR_MAP.items():
        series = load_surface_series(root, var, cf_var, start, end, dtype)
        if series is not None:
            data_vars[var] = series

    for var, cf_var in ATMOS_VAR_MAP.items():
        series = load_atmos_series(
            root, var, cf_var, hours, start, end, levels, dtype)
        if series is not None:
            data_vars[var] = series

    if not data_vars:
        raise RuntimeError("No variables found to assemble a dataset")

    dataset = xr.merge(data_vars.values(), compat="override", join="outer")
    if "time" in dataset.coords:
        dataset = dataset.sortby("time")
    dataset.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
    dataset.attrs["source"] = "ECMWF HRES analysis (download_hres_year.py)"
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("examples/downloads/hres"),
        help="Directory that contains surf_* and atmos_* GRIB files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hres.zarr"),
        help="Target Zarr store path",
    )
    parser.add_argument(
        "--start",
        type=parse_date_arg,
        default=None,
        help="Optional first day to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=parse_date_arg,
        default=None,
        help="Optional last day to include (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        nargs="*",
        default=(0, 6, 12, 18),
        help="Filter atmospheric files to these synoptic hours",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=DEFAULT_ATMOS_LEVELS,
        help="Pressure levels (hPa) to keep for atmospheric variables",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Floating dtype to cast data to before writing",
    )
    parser.add_argument(
        "--chunk-time",
        type=int,
        default=64,
        help="Chunk size along the time dimension",
    )
    parser.add_argument(
        "--chunk-lat",
        type=int,
        default=256,
        help="Chunk size along the latitude dimension",
    )
    parser.add_argument(
        "--chunk-lon",
        type=int,
        default=256,
        help="Chunk size along the longitude dimension",
    )
    parser.add_argument(
        "--chunk-level",
        type=int,
        default=None,
        help="Chunk size along the isobaricInhPa dimension",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output Zarr store if it already exists",
    )
    parser.add_argument(
        "--no-consolidate",
        action="store_true",
        help="Disable consolidated Zarr metadata (enabled by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    input_dir = args.input.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    logging.info("Collecting data from %s", input_dir)
    dataset = build_dataset(
        input_dir,
        args.start,
        args.end,
        args.hours,
        args.levels,
        args.dtype,
    )

    chunk_map = {
        dim: size
        for dim, size in (
            ("time", args.chunk_time),
            ("latitude", args.chunk_lat),
            ("longitude", args.chunk_lon),
            ("isobaricInhPa", args.chunk_level),
        )
        if size
    }
    chunk_map = {dim: size for dim,
                 size in chunk_map.items() if dim in dataset.dims}
    if chunk_map:
        dataset = dataset.chunk(chunk_map)
        logging.info("Applied chunks: %s", chunk_map)

    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "w-"
    consolidated = not args.no_consolidate

    time_dim = dataset.dims.get("time")
    if time_dim:
        time_values = dataset.indexes["time"]
        logging.info(
            "Dataset spans %s -> %s with %d time steps",
            time_values[0],
            time_values[-1],
            len(time_values),
        )
    logging.info("Variables: %s", ", ".join(sorted(dataset.data_vars)))
    logging.info("Writing Zarr dataset to %s", out_path)

    dataset.to_zarr(out_path, mode=mode, consolidated=consolidated)
    logging.info("Finished writing %s", out_path)


if __name__ == "__main__":
    main()
