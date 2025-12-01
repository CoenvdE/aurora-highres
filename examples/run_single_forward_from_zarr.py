#!/usr/bin/env python3
"""Run a single Aurora forward pass from an ERA5 Zarr chunk.

This script downloads the static fields (if needed), loads one ERA5 batch
for a requested day from the Zarr archive, executes a forward pass with a
pretrained Aurora checkpoint, and stores the input batch and prediction in
separate folders for easy inspection.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import torch

from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.utils import ensure_static_dataset, load_model

DEFAULT_ZARR_PATH = \
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-batch Aurora inference from ERA5 Zarr")
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=DEFAULT_ZARR_PATH,
        help="Path to the ERA5 Zarr archive",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2020-01-01",
        help="Date (YYYY-MM-DD) to extract at 12Z",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/run_single_forward"),
        help="Root directory for downloads and saved tensors",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force running on CPU even if CUDA is available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            f"--date must be YYYY-MM-DD, received {args.date!r}") from exc

    work_dir = args.work_dir.expanduser()
    inputs_dir = work_dir / "inputs"
    outputs_dir = work_dir / "outputs"
    cache_dir = work_dir / "cache"
    for directory in (inputs_dir, outputs_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    static_path = ensure_static_dataset(cache_dir)

    print("Loading batch from ERA5 Zarr...")
    batch = load_batch_from_zarr(
        zarr_path=args.zarr_path,
        static_path=str(static_path),
        date_str=args.date,
    )

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    batch_device = batch.to(device)
    model = load_model(device)

    print("Running forward pass...")
    with torch.no_grad():
        prediction = model(batch_device).to("cpu")

    batch_cpu = batch.to("cpu")

    input_file = inputs_dir / f"era5_batch_{args.date}.pt"
    output_file = outputs_dir / f"aurora_prediction_{args.date}.pt"

    print(f"Saving input batch to {input_file}")
    torch.save({"batch": batch_cpu}, input_file)

    print(f"Saving prediction to {output_file}")
    torch.save({"prediction": prediction}, output_file)

    print("Done. Inspect the saved tensors with torch.load() or Batch.to_netcdf().")


if __name__ == "__main__":
    main()

# usage:
# python examples/run_single_forward_from_zarr.py --date 2020-01-01