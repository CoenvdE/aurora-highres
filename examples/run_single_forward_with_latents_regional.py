#!/usr/bin/env python3
"""Run one Aurora forward pass and persist *regional* deaggregated latents.

This script mirrors ``run_single_forward_with_latents.py`` but additionally
restricts the captured deaggregated atmospheric latents to a geographic
window, using the same region-selection logic as
``examples/init_exploring/decode_deag_latents_region.py``. The intent is to
produce a compact latent dataset for downstream high-resolution decoding
experiments.
"""

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
from examples.init_exploring.decode_deag_latents_region import (
    _prepare_region_selection,
)


DEFAULT_ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/"
    "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-batch Aurora inference with *regional* latent capture"
        ),
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
        default=Path("examples/run_single_forward_latents_regional"),
        help="Root directory for downloads, tensors, and latents",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=30.0,
        help="Minimum latitude of the region of interest",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=70.0,
        help="Maximum latitude of the region of interest",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=-30.0,
        help="Minimum longitude of the region of interest",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=50.0,
        help="Maximum longitude of the region of interest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            f"--date must be YYYY-MM-DD, received {args.date!r}"
        ) from exc

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

    # Save the full patch grid once for reference.
    patch_grid_file = latents_dir / "patch_grid.pt"
    print(f"Saving patch grid to {patch_grid_file}")
    torch.save(
        {
            "patch_grid": patch_grid,
            "metadata": {
                "lat_shape": tuple(batch.metadata.lat.shape),
                "lon_shape": tuple(batch.metadata.lon.shape),
                "patch_size": model.patch_size,
            },
        },
        patch_grid_file,
    )

    latents_key = "decoder.deaggregated_atmospheric_latents"
    if latents_key not in captures:
        raise KeyError(
            f"Latents dictionary does not contain {latents_key!r}. "
            "Ensure the decoder hook is emitting deaggregated latents."
        )

    full_deagg_latents = captures[latents_key].detach().cpu()

    # Define the requested geographic region.
    requested_bounds = {
        "lat": (float(args.lat_min), float(args.lat_max)),
        "lon": (float(args.lon_min), float(args.lon_max)),
    }

    # Restrict latents to the region of interest using the same helper as the
    # interactive decode script. We only care about atmospheric latents here.
    mode = "atmos"
    region_latents, patch_rows, patch_cols, region_bounds, extent = (
        _prepare_region_selection(
            mode,
            requested_bounds,
            patch_grid,
            atmos_latents=full_deagg_latents,
        )
    )

    timestamp_label = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    latents_h5 = latents_dir / "deaggregated_latents_regional.h5"
    print(
        "Writing regional deaggregated latents for",
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
            data=region_latents.numpy(),
            compression="gzip",
        )
        dataset.attrs["timestamp"] = timestamp.isoformat()
        dataset.attrs["shape"] = tuple(region_latents.shape)
        dataset.attrs["patch_rows"] = int(patch_rows)
        dataset.attrs["patch_cols"] = int(patch_cols)
        dataset.attrs["patch_size"] = int(model.patch_size)
        dataset.attrs["patch_shape_full"] = tuple(
            int(v) for v in patch_grid["patch_shape"]
        )
        dataset.attrs["region_lat_min"] = float(region_bounds["lat"][0])
        dataset.attrs["region_lat_max"] = float(region_bounds["lat"][1])
        dataset.attrs["region_lon_min"] = float(region_bounds["lon"][0])
        dataset.attrs["region_lon_max"] = float(region_bounds["lon"][1])
        # ``extent`` is mainly for plotting; storing it is convenient but not
        # strictly required for training.
        dataset.attrs["extent"] = tuple(float(v) for v in extent)

    print("Done. Regional latents are ready for inspection/training.")


if __name__ == "__main__":
    main()

# usage:
# python examples/run_single_forward_with_latents_regional.py \
#   --zarr-path <path_to_zarr> --date 2020-01-01 \
#   --work-dir ./output_dir \
#   --lat-min 30 --lat-max 70 --lon-min -30 --lon-max 50
