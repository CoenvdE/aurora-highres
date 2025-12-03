#!/usr/bin/env python3
"""Run one Aurora forward pass and persist latents for a specific region.

This script mirrors `run_single_forward_with_latents.py` but slices the
captured atmospheric and surface latents to a user-specified geographic
region before writing them to disk.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch

from examples.extract_latents import register_latent_hooks
from examples.init_exploring.load_era_batch_snellius import load_batch_from_zarr
from examples.init_exploring.utils import (
    compute_patch_grid,
    ensure_static_dataset,
    load_model,
)

DEFAULT_ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/"
    "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)

# Default region as in `decode_deag_latents_region.py` (roughly Europe)
DEFAULT_LAT_MIN = 30.0
DEFAULT_LAT_MAX = 70.0
DEFAULT_LON_MIN = -30.0
DEFAULT_LON_MAX = 50.0


def _select_region_latents(
    latents: torch.Tensor,
    patch_shape: tuple[int, int],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> tuple[torch.Tensor, int, int]:
    """Slice a rectangular subset of latents in patch space.

    Latents are expected to have shape (batch, patches, levels, embed_dim),
    where patches = lat_patches * lon_patches.
    """
    lat_patches, lon_patches = patch_shape
    batch_size, _, levels, embed_dim = latents.shape
    patch_rows = row_end - row_start + 1
    patch_cols = col_end - col_start + 1

    latents = latents.reshape(
        batch_size,
        lat_patches,
        lon_patches,
        levels,
        embed_dim,
    )
    latents = latents[
        :,
        row_start: row_end + 1,
        col_start: col_end + 1,
        :,
        :,
    ].contiguous()
    latents = latents.reshape(
        batch_size,
        patch_rows * patch_cols,
        levels,
        embed_dim,
    )
    return latents, patch_rows, patch_cols


def _prepare_region_selection(
    *,
    mode: str,
    requested_bounds: Dict[str, Tuple[float, float]],
    patch_grid: dict,
    surface_latents: torch.Tensor | None = None,
    atmos_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, Dict[str, Tuple[float, float]]]:
    """Slice latents and derive bounds for the requested region.

    This uses the same patch-grid based selection logic as
    `decode_deag_latents_region._prepare_region_selection`, but returns only
    the region latents, patch grid dimensions and effective region bounds.
    """

    centres = patch_grid["centres"]  # (patch_count, 2) [lat, lon]
    patch_shape = patch_grid["patch_shape"]  # (lat_patches, lon_patches)
    lat_min_req, lat_max_req = requested_bounds["lat"]
    lon_min_req, lon_max_req = requested_bounds["lon"]

    centres_np = centres.detach().cpu().numpy()
    lat_centres = centres_np[:, 0]
    lon_centres = centres_np[:, 1]

    lat_mask = (lat_centres >= lat_min_req) & (lat_centres <= lat_max_req)
    lon_mask = (lon_centres >= lon_min_req) & (lon_centres <= lon_max_req)
    mask = lat_mask & lon_mask
    if not np.any(mask):
        raise ValueError("Requested bounds do not overlap any patches.")

    indices = np.where(mask)[0]
    lat_indices, lon_indices = np.unravel_index(indices, patch_shape)
    row_start = int(lat_indices.min())
    row_end = int(lat_indices.max())
    col_start = int(lon_indices.min())
    col_end = int(lon_indices.max())

    if mode == "surface":
        if surface_latents is None:
            raise ValueError("Surface latents are required for surface mode.")
        region_latents, patch_rows, patch_cols = _select_region_latents(
            surface_latents,
            patch_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
    elif mode == "atmos":
        if atmos_latents is None:
            raise ValueError(
                "Atmospheric latents are required for atmos mode.")
        region_latents, patch_rows, patch_cols = _select_region_latents(
            atmos_latents,
            patch_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    lat_used = lat_centres[indices]
    lon_used = lon_centres[indices]
    region_bounds = {
        "lat": (float(lat_used.min()), float(lat_used.max())),
        "lon": (float(lon_used.min()), float(lon_used.max())),
    }

    return (
        region_latents,
        patch_rows,
        patch_cols,
        region_bounds,
        row_start,
        row_end,
        col_start,
        col_end,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Single-batch Aurora inference with latent capture for a given "
            "region"
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
        default=Path("examples/run_single_forward_latents_region"),
        help="Root directory for downloads, tensors, and latents",
    )
    parser.add_argument(
        "--lat-min",
        type=float,
        default=DEFAULT_LAT_MIN,
        help="Minimum latitude of region (degrees)",
    )
    parser.add_argument(
        "--lat-max",
        type=float,
        default=DEFAULT_LAT_MAX,
        help="Maximum latitude of region (degrees)",
    )
    parser.add_argument(
        "--lon-min",
        type=float,
        default=DEFAULT_LON_MIN,
        help="Minimum longitude of region (degrees)",
    )
    parser.add_argument(
        "--lon-max",
        type=float,
        default=DEFAULT_LON_MAX,
        help="Maximum longitude of region (degrees)",
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

    patch_grid_file = latents_dir / "patch_grid.pt"
    print(f"Saving patch grid to {patch_grid_file}")
    torch.save(
        {
            "patch_grid": patch_grid,
            "metadata": {
                "lat_shape": tuple(batch.metadata.lat.shape),
                "lon_shape": tuple(batch.metadata.lon.shape),
                "patch_size": model.patch_size,
                "atmos_levels": batch.metadata.atmos_levels.detach().cpu(),
            },
        },
        patch_grid_file,
    )

    deagg_latents = captures.get("decoder.deaggregated_atmospheric_latents")
    surface_latents = captures.get("decoder.surface_latents")
    if deagg_latents is None or surface_latents is None:
        raise RuntimeError(
            "Expected both atmospheric and surface latents to be captured."
        )

    deagg_latents = deagg_latents.detach().cpu()
    surface_latents = surface_latents.detach().cpu()

    requested_bounds = {
        "lat": (args.lat_min, args.lat_max),
        "lon": (args.lon_min, args.lon_max),
    }
    print(
        "Selecting regional latents for "
        f"lat={requested_bounds['lat']}, lon={requested_bounds['lon']} ..."
    )

    (
        surf_region_latents,
        surf_patch_rows,
        surf_patch_cols,
        region_bounds,
        row_start,
        row_end,
        col_start,
        col_end,
    ) = _prepare_region_selection(
        mode="surface",
        requested_bounds=requested_bounds,
        patch_grid=patch_grid,
        surface_latents=surface_latents,
        atmos_latents=None,
    )
    (
        atmos_region_latents,
        atmos_patch_rows,
        atmos_patch_cols,
        _,
        row_start_atm,
        row_end_atm,
        col_start_atm,
        col_end_atm,
    ) = _prepare_region_selection(
        mode="atmos",
        requested_bounds=requested_bounds,
        patch_grid=patch_grid,
        surface_latents=None,
        atmos_latents=deagg_latents,
    )

    # Sanity check: surface and atmos selections should refer to the same
    # rectangle in patch space.
    if not (
        row_start == row_start_atm
        and row_end == row_end_atm
        and col_start == col_start_atm
        and col_end == col_end_atm
    ):
        raise RuntimeError("Surface and atmospheric region indices differ.")

    timestamp_label = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    latents_h5 = latents_dir / "pressure_surface_latents_region.h5"
    print(
        "Writing REGIONAL decoder latents for",
        timestamp_label,
        "to",
        latents_h5,
    )

    with h5py.File(latents_h5, "a") as handle:
        # Integer index window in the global patch grid; these are enough to
        # reconstruct region bounds and centres from `patch_grid.pt`.
        handle.attrs["row_start"] = row_start
        handle.attrs["row_end"] = row_end
        handle.attrs["col_start"] = col_start
        handle.attrs["col_end"] = col_end

        levels_group = handle.require_group("pressure_latents_region")
        if timestamp_label in levels_group:
            del levels_group[timestamp_label]
        pressure_dataset = levels_group.create_dataset(
            timestamp_label,
            data=atmos_region_latents.numpy(),
            compression="gzip",
        )
        pressure_dataset.attrs["timestamp"] = timestamp.isoformat()
        pressure_dataset.attrs["shape"] = atmos_region_latents.shape

        surface_group = handle.require_group("surface_latents_region")
        if timestamp_label in surface_group:
            del surface_group[timestamp_label]
        surface_dataset = surface_group.create_dataset(
            timestamp_label,
            data=surf_region_latents.numpy(),
            compression="gzip",
        )
        surface_dataset.attrs["timestamp"] = timestamp.isoformat()
        surface_dataset.attrs["shape"] = surf_region_latents.shape

    captures.clear()

    print("Done. Regional pressure and surface latents are ready for inspection.")


if __name__ == "__main__":
    main()

# usage:
# python examples/run_single_forward_with_latents_region.py \
#   --zarr-path <path_to_zarr> \
#   --date 2020-01-01 \
#   --work-dir ./output_dir_region \
#   --lat-min 30 --lat-max 70 --lon-min -30 --lon-max 50
