#!/usr/bin/env python3
"""Process one or more years of ERA5 samples and store Aurora latents efficiently."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import torch
import xarray as xr

from examples.extract_latents import register_latent_hooks
from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.init_exploring.utils import (
    compute_patch_grid,
    ensure_static_dataset,
    load_model,
)

DEFAULT_ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Aurora forward passes for each day in a year while reusing "
            "a single Zarr Dataset handle and capturing latents."
        )
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=DEFAULT_ZARR_PATH,
        help="Path to the ERA5 Zarr archive",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First year (inclusive) to process",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="Last year (inclusive) to process",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/run_year_forward_latents"),
        help="Root directory for downloads, tensors, and latents",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose latents dataset already exists",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=2,
        help="Stop after processing at most this many time steps",
    )
    return parser.parse_args()


def iterate_days(year: int):
    current = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    delta = timedelta(days=1)
    while current <= end:
        yield current
        current += delta


def latent_exists(latents_h5: Path, label: str) -> bool:
    if not latents_h5.exists():
        return False

    with h5py.File(latents_h5, "r") as handle:
        latents_group = handle.get("pressure_latents")
        if latents_group is None:
            return False
        return label in latents_group


def write_patch_grid_once(latents_h5: Path, patch_grid: dict[str, torch.Tensor | int | tuple[int, int]]) -> None:
    with h5py.File(latents_h5, "a") as handle:
        if "patch_grid" in handle:
            return

        patch_group = handle.create_group("patch_grid")
        centres = patch_grid["centres"].detach().cpu().numpy()
        bounds = patch_grid["bounds"].detach().cpu().numpy()
        root_area = patch_grid["root_area"].detach().cpu().numpy()

        patch_group.create_dataset("centres", data=centres)
        patch_group.create_dataset("bounds", data=bounds)
        patch_group.create_dataset("root_area", data=root_area)
        patch_group.attrs["patch_shape"] = tuple(
            int(x) for x in patch_grid["patch_shape"])
        patch_group.attrs["patch_size"] = int(patch_grid["patch_size"])
        patch_group.attrs["patch_count"] = int(patch_grid["patch_count"])


def save_patch_grid_file(destination: Path, patch_grid: dict[str, torch.Tensor | int | tuple[int, int]]) -> None:
    if destination.exists():
        return

    serialisable: dict[str, torch.Tensor | int | tuple[int, int]] = {}
    for key, value in patch_grid.items():
        if isinstance(value, torch.Tensor):
            serialisable[key] = value.detach().cpu()
        else:
            serialisable[key] = value

    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialisable, destination)


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    work_dir = args.work_dir.expanduser()
    latents_dir = work_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    static_path = ensure_static_dataset(work_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    model.eval()

    patch_grid: dict[str, torch.Tensor | int | tuple[int, int]] | None = None
    patch_grid_cache_path = latents_dir / "patch_grid.pt"

    captures: dict[str, torch.Tensor] = {}
    handles, decoder_cleanup = register_latent_hooks(model, captures)

    try:
        print("Opening ERA5 Zarr dataset once...")
        dataset = xr.open_zarr(args.zarr_path, consolidated=True)

        print("Opening static dataset once...")
        try:
            static_dataset = xr.open_dataset(static_path, engine="netcdf4")
        except OSError:
            print("Falling back to scipy engine for static dataset...")
            static_dataset = xr.open_dataset(static_path, engine="scipy")

        processed_samples = 0
        for year in range(args.start_year, args.end_year + 1):
            print(f"Processing year {year}...")
            year_latents_dir = latents_dir / str(year)
            year_latents_dir.mkdir(parents=True, exist_ok=True)

            latents_h5 = year_latents_dir / "pressure_surface_latents.h5"

            for day in iterate_days(year):
                date_str = day.strftime("%Y-%m-%d")
                expected_time = datetime(day.year, day.month, day.day, 12)
                expected_label = expected_time.strftime("%Y-%m-%dT%H-%M-%S")
                if args.skip_existing and latent_exists(latents_h5, expected_label):
                    print(f"  Skipping {date_str} (latents already exist)")
                    continue

                print(f"  Processing {date_str}...")
                try:
                    batch = load_batch_from_zarr(
                        zarr_path=args.zarr_path,
                        static_path=str(static_path),
                        date_str=date_str,
                        dataset=dataset,
                        static_dataset=static_dataset,
                    )
                except Exception as exc:
                    print(f"    Failed to load batch: {exc}")
                    continue

                if patch_grid is None:
                    patch_grid = compute_patch_grid(
                        batch.metadata.lat,
                        batch.metadata.lon,
                        model.patch_size,
                    )
                    save_patch_grid_file(patch_grid_cache_path, patch_grid)
                write_patch_grid_once(latents_h5, patch_grid)

                batch_device = batch.to(device)
                with torch.no_grad():
                    prediction = model(batch_device).to("cpu")

                actual_time = prediction.metadata.time[0]
                deagg_latents = captures.get(
                    "decoder.deaggregated_atmospheric_latents")
                surface_latents = captures.get("decoder.surface_latents")
                if deagg_latents is None or surface_latents is None:
                    captures.clear()
                    raise RuntimeError(
                        "Decoder latent captures are missing; ensure hooks are registered."
                    )
                timestamp_label = actual_time.strftime("%Y-%m-%dT%H-%M-%S")

                print(
                    "Writing decoder latents for",
                    timestamp_label,
                    "to",
                    latents_h5,
                )

                with h5py.File(latents_h5, "a") as handle:
                    levels_group = handle.require_group("pressure_latents")
                    if timestamp_label in levels_group:
                        del levels_group[timestamp_label]

                    pressure_dataset = levels_group.create_dataset(
                        timestamp_label,
                        data=deagg_latents.numpy(),
                        compression="gzip",
                    )
                    pressure_dataset.attrs["timestamp"] = actual_time.isoformat()
                    pressure_dataset.attrs["shape"] = deagg_latents.shape

                    surface_group = handle.require_group("surface_latents")
                    if timestamp_label in surface_group:
                        del surface_group[timestamp_label]
                    surface_dataset = surface_group.create_dataset(
                        timestamp_label,
                        data=surface_latents.numpy(),
                        compression="gzip",
                    )
                    surface_dataset.attrs["timestamp"] = actual_time.isoformat(
                    )
                    surface_dataset.attrs["shape"] = surface_latents.shape

                captures.clear()
                processed_samples += 1
                if args.max_samples is not None and processed_samples >= args.max_samples:
                    print("Reached max sample limit, stopping early.")
                    break
            if args.max_samples is not None and processed_samples >= args.max_samples:
                break
    finally:
        for handle in handles:
            handle.remove()
        decoder_cleanup()
        if "static_dataset" in locals():
            static_close = getattr(static_dataset, "close", None)
            if callable(static_close):
                static_close()
        if "dataset" in locals():
            dataset_close = getattr(dataset, "close", None)
            if callable(dataset_close):
                dataset_close()

    print("Finished processing requested years.")


if __name__ == "__main__":
    main()

# usage: python examples/run_year_forward_with_latents.py --zarr-path <path_to_zarr> --start-year 2018 --end-year 2018 --work-dir ./output_dir --max-samples 5