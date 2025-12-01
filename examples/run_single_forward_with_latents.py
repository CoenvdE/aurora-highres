#!/usr/bin/env python3
"""Run one Aurora forward pass and persist the captured latents."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from examples.extract_latents_big import register_latent_hooks
from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.utils import (
    compute_patch_grid,
    ensure_static_dataset,
    format_latents_filename,
    load_model,
)

DEFAULT_ZARR_PATH = \
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-batch Aurora inference with latent capture",
    )
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
        default=Path("examples/run_single_forward_latents"),
        help="Root directory for downloads, tensors, and latents",
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
    latents_dir = work_dir / "latents"
    cache_dir = work_dir / "cache"
    for directory in (inputs_dir, outputs_dir, latents_dir, cache_dir):
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

    captures: dict[str, torch.Tensor] = {}
    handles, decoder_cleanup = register_latent_hooks(model, captures)

    print("Running forward pass and capturing latents...")
    try:
        with torch.no_grad():
            prediction = model(batch_device).to("cpu")
    finally:
        for handle in handles:
            handle.remove()
        decoder_cleanup()

    batch_cpu = batch.to("cpu")

    input_file = inputs_dir / f"era5_batch_{args.date}.pt"
    output_file = outputs_dir / f"aurora_prediction_{args.date}.pt"

    print(f"Saving input batch to {input_file}")
    torch.save({"batch": batch_cpu}, input_file)

    print(f"Saving prediction to {output_file}")
    torch.save({"prediction": prediction}, output_file)

    timestamp = prediction.metadata.time[0]
    latents_file = latents_dir / format_latents_filename(timestamp)
    patch_grid = compute_patch_grid(
        batch.metadata.lat,
        batch.metadata.lon,
        model.patch_size,
    )

    print(f"Saving latents snapshot to {latents_file}")
    torch.save(
        {
            "captures": captures,
            "patch_grid": patch_grid,
            "prediction": prediction,
        },
        latents_file,
    )

    print("Done. Latents and tensors are ready for inspection.")


if __name__ == "__main__":
    main()
