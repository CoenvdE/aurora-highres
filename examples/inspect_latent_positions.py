#!/usr/bin/env python3
"""Inspect latent positions using saved patch_grid and latents.

This script:
- Loads `patch_grid.pt` and a latents HDF5 file produced by either
  `run_single_forward_with_latents.py` or
  `run_single_forward_with_latents_region.py`.
- Builds a per-patch position tensor from the stored patch centres
  (lat, lon for each patch).
- Prints:
    * shape of the position tensor
    * shape of the latent tensor
    * min/max over latitude and longitude positions.

Usage examples:

Global latents (from run_single_forward_with_latents.py):

    python examples/inspect_latent_positions.py \
        --work-dir examples/run_single_forward_latents \
        --group-name pressure_latents \
        --timestamp 2020-01-01T12-00-00

Regional latents (from run_single_forward_with_latents_region.py):

    python examples/inspect_latent_positions.py \
        --work-dir examples/run_single_forward_latents_region \
        --group-name pressure_latents_region \
        --timestamp 2020-01-01T12-00-00
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Aurora latents and their associated patch-centre positions."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help=(
            "Root directory containing a 'latents' subdirectory with "
            "patch_grid.pt and a latents HDF5 file."
        ),
    )
    parser.add_argument(
        "--latents-file-name",
        type=str,
        default="pressure_surface_latents_region.h5",
        help=(
            "Name of the HDF5 latents file under <work-dir>/latents/. "
            "For regional latents use 'pressure_surface_latents_region.h5'."
        ),
    )
    parser.add_argument(
        "--group-name",
        type=str,
        default="pressure_latents_region",
        help=(
            "HDF5 group containing the latents (e.g. 'pressure_latents', "
            "'surface_latents', 'pressure_latents_region', "
            "'surface_latents_region')."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        # required=True,
        default="2020-01-01T18-00-00",
        help=(
            "Timestamp label of the dataset inside the group, e.g. "
            "'2020-01-01T18-00-00'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    latents_dir = args.work_dir.expanduser() / "latents"
    patch_grid_path = latents_dir / "patch_grid.pt"
    latents_h5_path = latents_dir / args.latents_file_name

    if not patch_grid_path.is_file():
        raise FileNotFoundError(
            f"patch_grid.pt not found at {patch_grid_path}")
    if not latents_h5_path.is_file():
        raise FileNotFoundError(
            f"Latents HDF5 file not found at {latents_h5_path}")

    # Load patch grid metadata (contains centres and patch_shape).
    grid_obj = torch.load(patch_grid_path)
    patch_grid = grid_obj["patch_grid"]
    centres = patch_grid["centres"]  # (patch_count, 2) [lat, lon]
    patch_shape = patch_grid["patch_shape"]  # (lat_patches, lon_patches)

    # Load the latent tensor for the requested timestamp.
    with h5py.File(latents_h5_path, "r") as handle:
        if args.group_name not in handle:
            raise KeyError(
                f"Group {args.group_name!r} not found in {latents_h5_path.name}"
            )
        group = handle[args.group_name]
        if args.timestamp not in group:
            raise KeyError(
                f"Timestamp {args.timestamp!r} not found in group "
                f"{args.group_name!r} in {latents_h5_path.name}"
            )
        latents_np = group[args.timestamp][...]

    latents = torch.from_numpy(latents_np)

    # Latents are expected to be (batch, patches, levels, embed_dim).
    batch_size, patch_count, levels, embed_dim = latents.shape

    # Build per-patch positions by tiling centres over batch and levels.
    # centres: (patch_count, 2) -> (1, patch_count, 1, 2)
    centres_expanded = centres.view(1, patch_count, 1, 2)
    positions = centres_expanded.expand(batch_size, patch_count, levels, 2)

    print("Latents shape       :", tuple(latents.shape))
    print("Positions shape     :", tuple(positions.shape))

    # Compute min/max for lat / lon over all patches, levels, batch.
    lats = positions[..., 0]
    lons = positions[..., 1]

    print("Latitude min / max  :", float(lats.min()), float(lats.max()))
    print("Longitude min / max :", float(lons.min()), float(lons.max()))


if __name__ == "__main__":
    main()
