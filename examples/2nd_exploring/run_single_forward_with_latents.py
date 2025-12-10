#!/usr/bin/env python3
"""Run one Aurora forward pass and persist the captured latents."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py
import torch

from examples.init_exploring.extract_latents import register_latent_hooks
from examples.init_exploring.load_era_batch_snellius import load_batch_from_zarr
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            f"--date must be YYYY-MM-DD, received {args.date!r}") from exc

    work_dir = args.work_dir.expanduser()
    latents_dir = work_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)

    static_path = ensure_static_dataset(work_dir)

    print("Loading batch from ERA5 Zarr...")
    batch = load_batch_from_zarr(
        zarr_path=args.zarr_path,
        static_path=str(static_path),
        date_str=args.date,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "atmos_levels": tuple(batch.metadata.atmos_levels),
        },
    }, patch_grid_file)

    deagg_latents = captures.get("decoder.deaggregated_atmospheric_latents")
    surface_latents = captures.get("decoder.surface_latents")
    deagg_latents = deagg_latents.detach().cpu()
    surface_latents = surface_latents.detach().cpu()

    timestamp_label = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    latents_h5 = latents_dir / "pressure_surface_latents.h5"
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
        pressure_dataset.attrs["timestamp"] = timestamp.isoformat()
        pressure_dataset.attrs["shape"] = deagg_latents.shape

        surface_group = handle.require_group("surface_latents")
        if timestamp_label in surface_group:
            del surface_group[timestamp_label]

        surface_dataset = surface_group.create_dataset(
            timestamp_label,
            data=surface_latents.numpy(),
            compression="gzip",
        )
        surface_dataset.attrs["timestamp"] = timestamp.isoformat()
        surface_dataset.attrs["shape"] = surface_latents.shape

    captures.clear()

    print("Done. Pressure and surface latents are ready for inspection.")


if __name__ == "__main__":
    main()

# usage: python examples/run_single_forward_with_latents.py --zarr-path <path_to_zarr> --date 2020-01-01 --work-dir ./output_dir
