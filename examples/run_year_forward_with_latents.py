#!/usr/bin/env python3
"""Process one or more years of ERA5 samples and store Aurora latents efficiently."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import torch
import xarray as xr

from examples.extract_latents_big import register_latent_hooks
from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.utils import (
    compute_patch_grid,
    ensure_static_dataset,
    format_latents_filename,
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
        default=2020,
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
        "--cpu",
        action="store_true",
        help="Force running on CPU even if CUDA is available",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples whose latents file already exists",
    )
    return parser.parse_args()


def iterate_days(year: int):
    current = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    delta = timedelta(days=1)
    while current <= end:
        yield current
        current += delta


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year:
        raise SystemExit("--start-year must be <= --end-year")

    work_dir = args.work_dir.expanduser()
    inputs_dir = work_dir / "inputs"
    outputs_dir = work_dir / "outputs"
    latents_dir = work_dir / "latents"
    cache_dir = work_dir / "cache"
    for directory in (inputs_dir, outputs_dir, latents_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    static_path = ensure_static_dataset(cache_dir)

    print("Opening ERA5 Zarr dataset once...")
    dataset = xr.open_zarr(args.zarr_path, consolidated=True)

    print("Opening static dataset once...")
    static_dataset = xr.open_dataset(static_path, engine="netcdf4")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    model = load_model(device)
    model.eval()

    captures: dict[str, torch.Tensor] = {}
    handles, decoder_cleanup = register_latent_hooks(model, captures)

    try:
        for year in range(args.start_year, args.end_year + 1):
            print(f"Processing year {year}...")
            year_inputs_dir = inputs_dir / str(year)
            year_outputs_dir = outputs_dir / str(year)
            year_latents_dir = latents_dir / str(year)
            for directory in (year_inputs_dir, year_outputs_dir, year_latents_dir):
                directory.mkdir(parents=True, exist_ok=True)

            for day in iterate_days(year):
                date_str = day.strftime("%Y-%m-%d")
                expected_time = datetime(day.year, day.month, day.day, 12)
                latents_file = year_latents_dir / \
                    format_latents_filename(expected_time)
                if args.skip_existing and latents_file.exists():
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

                batch_device = batch.to(device)
                with torch.no_grad():
                    prediction = model(batch_device).to("cpu")

                actual_time = prediction.metadata.time[0]
                latents_file = year_latents_dir / \
                    format_latents_filename(actual_time)

                patch_grid = compute_patch_grid(
                    batch.metadata.lat,
                    batch.metadata.lon,
                    model.patch_size,
                )

                input_file = year_inputs_dir / f"era5_batch_{date_str}.pt"
                output_file = year_outputs_dir / \
                    f"aurora_prediction_{date_str}.pt"

                batch_cpu = batch.to("cpu")

                print(f"    Saving input batch to {input_file}")
                torch.save({"batch": batch_cpu}, input_file)

                print(f"    Saving prediction to {output_file}")
                torch.save({"prediction": prediction}, output_file)

                print(f"    Saving latents to {latents_file}")
                torch.save(
                    {
                        "captures": dict(captures),
                        "patch_grid": patch_grid,
                        "prediction": prediction,
                    },
                    latents_file,
                )

                captures.clear()
    finally:
        for handle in handles:
            handle.remove()
        decoder_cleanup()
        static_dataset.close()
        dataset.close()

    print("Finished processing requested years.")


if __name__ == "__main__":
    main()
