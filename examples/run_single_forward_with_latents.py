#!/usr/bin/env python3
"""Run one Aurora forward pass and persist the captured latents."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import torch

from examples.extract_latents import register_latent_hooks
from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.init_exploring.utils import (
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

    timestamp = prediction.metadata.time[0]
    patch_grid = compute_patch_grid(
        batch.metadata.lat,
        batch.metadata.lon,
        model.patch_size,
    )

    patch_grid_file = latents_dir / "patch_grid.pt"
    print(f"Saving patch grid to {patch_grid_file}")
    torch.save({
        "patch_grid": patch_grid,
        "metadata": {
            "lat_shape": tuple(batch.metadata.lat.shape),
            "lon_shape": tuple(batch.metadata.lon.shape),
            "patch_size": model.patch_size,
        },
    }, patch_grid_file)

    latents_key = "decoder.deaggregated_atmospheric_latents"
    if latents_key not in captures:
        raise KeyError(
            f"Latents dictionary does not contain {latents_key!r}."
            " Ensure the decoder hook is emitting deaggregated latents."
        )

    deagg_latents = captures[latents_key].detach().cpu()
    timestamp_label = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    latents_h5 = latents_dir / "deaggregated_latents.h5"
    print(
        "Writing deaggregated latents for",
        timestamp_label,
        "to",
        latents_h5,
    )

    with h5py.File(latents_h5, "a") as handle:
        latents_group = handle.require_group("latents")
        if timestamp_label in latents_group:
            del latents_group[timestamp_label]
        dataset = latents_group.create_dataset(
            timestamp_label,
            data=deagg_latents.numpy(),
            compression="gzip",
        )
        dataset.attrs["timestamp"] = timestamp.isoformat()
        dataset.attrs["shape"] = deagg_latents.shape

    print("Done. Latents and tensors are ready for inspection.")


if __name__ == "__main__":
    main()
