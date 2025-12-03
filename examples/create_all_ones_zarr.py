#!/usr/bin/env python3
"""Create a Zarr dataset with Aurora-like latent structure filled with ones.

This is a synthetic helper to prototype a good Zarr layout for latents
before wiring it into the real forward code.

The layout mimics what comes out of ``run_year_forward_with_latents.py``:

- dimensions: time, patch, level, latent
- coordinates: time, patch, level, latent, lat(patch), lon(patch)
- variables:
    * pressure_latents(time, patch, level, latent)
    * surface_latents(time, patch, latent)

All values are set to 1.0 so we can cheaply test IO, chunking, etc.

Usage
-----
    python -m examples.create_all_ones_zarr \
        --source-h5 examples/run_year_forward_latents/latents/2018/pressure_surface_latents.h5 \
        --zarr-out examples/run_year_forward_latents/latents/all_ones_latents.zarr

The script will:
- Read the patch grid + one latent sample from the HDF5 file
- Infer shapes and coordinate sizes
- Create a Zarr store with the same shape but filled with ones for *all*
  timesteps and years in the requested range.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd
import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an all-ones Aurora latent Zarr dataset.")
    parser.add_argument(
        "--source-h5",
        type=Path,
        required=True,
        help=(
            "Path to an existing pressure_surface_latents.h5 file used only "
            "to infer patch grid and latent shapes."
        ),
    )
    parser.add_argument(
        "--zarr-out",
        type=Path,
        required=True,
        help="Path to the Zarr root that will be created.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First year (inclusive) to cover in the synthetic time axis.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2018,
        help="Last year (inclusive) to cover in the synthetic time axis.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        nargs="+",
        default=[12],
        help=(
            "Hours of day to include for each date (e.g. 0 6 12 18). "
            "All combinations of date x hour in the year range are used."
        ),
    )
    parser.add_argument(
        "--chunks-patch",
        type=int,
        default=1024,
        help="Chunk size along patch dimension.",
    )
    parser.add_argument(
        "--chunks-latent",
        type=int,
        default=64,
        help="Chunk size along latent dimension.",
    )
    return parser.parse_args()


def nice_time_grid(start_year: int, end_year: int, hours: Iterable[int]) -> pd.DatetimeIndex:
    """Create a dense time grid for all years and requested hours.

    We assume 1-day steps, and include each hour in ``hours`` for each date.
    """

    # Daily dates across the span
    start = pd.Timestamp(start_year, 1, 1, 0)
    end = pd.Timestamp(end_year, 12, 31, 23)

    dates = pd.date_range(start=start.normalize(),
                          end=end.normalize(), freq="D")
    hour_list = sorted(set(int(h) for h in hours))

    times: list[pd.Timestamp] = []
    for d in dates:
        for h in hour_list:
            times.append(d + pd.Timedelta(hours=h))

    return pd.DatetimeIndex(times).sort_values()


def infer_from_h5(source_h5: Path):
    if not source_h5.exists():
        raise SystemExit(f"Source HDF5 not found: {source_h5}")

    with h5py.File(source_h5, "r") as handle:
        if "patch_grid" not in handle:
            raise SystemExit("HDF5 file is missing 'patch_grid' group.")

        patch_grid = handle["patch_grid"]
        centres = patch_grid["centres"][()]  # (patch, 2)
        bounds = patch_grid["bounds"][()]    # (patch, 4)
        root_area = patch_grid["root_area"][()]

        patch_count = centres.shape[0]
        lat = centres[:, 0]
        lon = centres[:, 1]

        pressure_group = handle.get("pressure_latents")
        surface_group = handle.get("surface_latents")
        if pressure_group is None or surface_group is None:
            raise SystemExit(
                "HDF5 file must contain 'pressure_latents' and 'surface_latents' groups.")

        dataset_names = list(pressure_group.keys())
        if not dataset_names:
            raise SystemExit("No datasets found in 'pressure_latents' group.")

        # Use first pressure and surface sample to infer latent dims
        # (1, patch, level, latent)
        p_sample = pressure_group[dataset_names[0]][()]
        if p_sample.shape[1] != patch_count:
            raise SystemExit(
                f"Inconsistent patch dimension: patch_grid has {patch_count}, "
                f"pressure sample has {p_sample.shape[1]}"
            )

        _, _, n_level, n_latent = p_sample.shape

        s_sample = surface_group[dataset_names[0]][()]  # (1, patch, 1, latent)
        if s_sample.shape[1] != patch_count:
            raise SystemExit(
                f"Inconsistent patch dimension between patch_grid and surface sample: "
                f"{patch_count} vs {s_sample.shape[1]}"
            )

    return {
        "patch_count": patch_count,
        "lat": lat,
        "lon": lon,
        "bounds": bounds,
        "root_area": root_area,
        "n_level": n_level,
        "n_latent": n_latent,
    }


def create_all_ones_zarr(
    source_h5: Path,
    zarr_out: Path,
    start_year: int,
    end_year: int,
    hours: Iterable[int],
    chunks_patch: int,
    chunks_latent: int,
) -> None:
    meta = infer_from_h5(source_h5)

    patch_count = meta["patch_count"]
    lat = meta["lat"]
    lon = meta["lon"]
    bounds = meta["bounds"]
    root_area = meta["root_area"]
    n_level = meta["n_level"]
    n_latent = meta["n_latent"]

    times = nice_time_grid(start_year, end_year, hours)
    n_time = times.size

    print(f"patch_count={patch_count}, n_level={n_level}, n_latent={n_latent}")
    print(f"n_time={n_time}, spanning {times[0]} .. {times[-1]}")

    # Coordinates
    coords = {
        "time": ("time", times),
        "patch": ("patch", np.arange(patch_count, dtype=np.int32)),
        "level": ("level", np.arange(n_level, dtype=np.int32)),
        "latent": ("latent", np.arange(n_latent, dtype=np.int32)),
        "lat": ("patch", lat.astype(np.float32)),
        "lon": ("patch", lon.astype(np.float32)),
        "root_area": ("patch", root_area.astype(np.float32)),
        "bounds": (("patch", "corner"), bounds.reshape(patch_count, -1).astype(np.float32)),
    }

    # Data variables: fill lazily with ones using dask-like broadcasting via xarray
    # but here we construct dense arrays chunk by chunk via to_zarr(region=...).

    # Initialise empty dataset on disk with proper metadata and encoding
    empty = xr.Dataset(
        data_vars=dict(
            pressure_latents=(
                ("time", "patch", "level", "latent"),
                np.empty((0, patch_count, n_level, n_latent),
                         dtype=np.float32),
            ),
            surface_latents=(
                ("time", "patch", "latent"),
                np.empty((0, patch_count, n_latent), dtype=np.float32),
            ),
        ),
        coords=coords,
    )

    # Write only metadata (no time slices yet)
    if zarr_out.exists():
        # Safety: avoid clobbering silently
        raise SystemExit(
            f"Refusing to overwrite existing Zarr store: {zarr_out}")

    empty.isel(time=slice(0, 0)).to_zarr(
        zarr_out,
        mode="w",
        encoding={
            "pressure_latents": {
                "chunks": (1, chunks_patch, n_level, chunks_latent),
            },
            "surface_latents": {
                "chunks": (1, chunks_patch, chunks_latent),
            },
        },
    )

    # Now fill each time index with ones, using region writes to avoid
    # holding the full (time, ...) array in memory.
    one_pressure = np.ones(
        (1, patch_count, n_level, n_latent), dtype=np.float32)
    one_surface = np.ones((1, patch_count, n_latent), dtype=np.float32)

    for t_idx in range(n_time):
        t = times[t_idx]
        if t_idx % 100 == 0 or t_idx == n_time - 1:
            print(f"Writing time index {t_idx+1}/{n_time} -> {t}")

        ds_step = xr.Dataset(
            data_vars=dict(
                pressure_latents=(
                    ("time", "patch", "level", "latent"),
                    one_pressure,
                ),
                surface_latents=(
                    ("time", "patch", "latent"),
                    one_surface,
                ),
            ),
            coords={
                "time": ("time", [t]),
                "patch": coords["patch"],
                "level": coords["level"],
                "latent": coords["latent"],
            },
        )

        ds_step.to_zarr(
            zarr_out,
            mode="r+",
            region={"time": slice(t_idx, t_idx + 1)},
        )

    print(f"Finished writing all-ones Zarr store to {zarr_out}")


def main() -> None:
    args = parse_args()
    create_all_ones_zarr(
        source_h5=args.source_h5.expanduser(),
        zarr_out=args.zarr_out.expanduser(),
        start_year=args.start_year,
        end_year=args.end_year,
        hours=args.hours,
        chunks_patch=args.chunks_patch,
        chunks_latent=args.chunks_latent,
    )


if __name__ == "__main__":
    main()
