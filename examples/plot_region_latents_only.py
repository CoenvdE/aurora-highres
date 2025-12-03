#!/usr/bin/env python3
"""Decode precomputed regional latents and plot them on a map.

This script:
- Loads regional surface + atmospheric latents saved by
  `run_single_forward_with_latents_region.py` (no new forward pass).
- Decodes one variable for a single timestamp.
- Produces:
    1) A global white map with land mask and the decoded region overlaid.
    2) A zoomed-in map of just the region.

Usage example:

    python examples/plot_region_latents_only.py \
        --work-dir examples/run_single_forward_latents_region \
        --timestamp 2020-01-01T12-00-00 \
        --mode surface \
        --var-name 2t
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch

from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.util import unpatchify
from aurora.normalisation import unnormalise_atmos_var, unnormalise_surf_var
from examples.init_exploring.utils import (
    load_model,
)
from examples.init_exploring.helpers_plot_region import (
    _bounds_to_extent,
    _compute_color_limits,
    _plot_world_and_region,
    _plot_region_only,
)


def _decode_subset(
    decoder: Perceiver3DDecoder,
    latents: torch.Tensor,
    var_names: Sequence[str],
    patch_rows: int,
    patch_cols: int,
    head_builder: Callable[[str], torch.Tensor],
    unnormalise,
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    patch_size = decoder.patch_size
    total_height = patch_rows * patch_size
    total_width = patch_cols * patch_size
    head_outputs = torch.stack(
        [head_builder(name) for name in var_names],
        dim=-1,
    )
    head_outputs = head_outputs.reshape(*head_outputs.shape[:3], -1)
    preds = unpatchify(
        head_outputs,
        len(var_names),
        total_height,
        total_width,
        patch_size,
    )
    if postprocess is not None:
        preds = postprocess(preds)
    return unnormalise(preds, var_names)


def _unnormalise_surface_preds(
    surf_preds: torch.Tensor,
    surf_vars: Sequence[str],
) -> torch.Tensor:
    return _unnormalise_preds(
        surf_preds,
        surf_vars,
        lambda slice_, name: unnormalise_surf_var(slice_, name),
    )


def _unnormalise_atmos_preds(
    atmos_preds: torch.Tensor,
    atmos_vars: Sequence[str],
    levels: torch.Tensor,
) -> torch.Tensor:
    return _unnormalise_preds(
        atmos_preds,
        atmos_vars,
        lambda slice_, name: unnormalise_atmos_var(slice_, name, levels),
    )


def _unnormalise_preds(
    preds: torch.Tensor,
    var_names: Sequence[str],
    normaliser: Callable[[torch.Tensor, str], torch.Tensor],
) -> torch.Tensor:
    restored: list[torch.Tensor] = []
    for idx, name in enumerate(var_names):
        slice_ = preds[:, idx: idx + 1]
        restored.append(normaliser(slice_, name))
    return torch.cat(restored, dim=1)


def _prepare_region_from_patch_grid(
    patch_grid_path: Path,
    latents_h5: Path,
    timestamp_label: str,
    mode: str,
) -> tuple[
    torch.Tensor,
    int,
    int,
    dict[str, tuple[float, float]],
    tuple[float, float, float, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Load region latents and derive bounds/extents using saved patch_grid + attrs.

    Returns:
        region_latents, patch_rows, patch_cols, region_bounds, extent,
        latitudes, longitudes
    """
    # Load patch grid metadata
    grid_obj = torch.load(patch_grid_path)
    patch_grid = grid_obj["patch_grid"]
    metadata = grid_obj["metadata"]
    lat_shape = metadata["lat_shape"]
    lon_shape = metadata["lon_shape"]
    atmos_levels = metadata["atmos_levels"]

    centres = patch_grid["centres"]  # (patch_count, 2) [lat, lon]
    patch_shape = patch_grid["patch_shape"]  # (lat_patches, lon_patches)

    centres_np = centres.detach().cpu().numpy()
    lat_centres = centres_np[:, 0]
    lon_centres = centres_np[:, 1]
    lat_min, lat_max = float(lat_centres.min()), float(lat_centres.max())
    lon_min, lon_max = float(lon_centres.min()), float(lon_centres.max())

    latitudes = torch.linspace(lat_min, lat_max, lat_shape[0])
    longitudes = torch.linspace(lon_min, lon_max, lon_shape[0])

    # Load region latents indices + data
    with h5py.File(latents_h5, "r") as handle:
        row_start = int(handle.attrs["row_start"])
        row_end = int(handle.attrs["row_end"])
        col_start = int(handle.attrs["col_start"])
        col_end = int(handle.attrs["col_end"])

        if mode == "surface":
            group = handle["surface_latents_region"]
        elif mode == "atmos":
            group = handle["pressure_latents_region"]
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if timestamp_label not in group:
            raise KeyError(
                f"Timestamp {timestamp_label!r} not found in {mode} latents group."
            )
        # (1, patches_region, levels, dim)
        region_array = group[timestamp_label][...]

    # Derive region bounds from the subset of patch centres
    all_indices = np.arange(centres.shape[0])
    lat_ids, lon_ids = np.unravel_index(all_indices, patch_shape)

    mask = (
        (lat_ids >= row_start)
        & (lat_ids <= row_end)
        & (lon_ids >= col_start)
        & (lon_ids <= col_end)
    )
    used_centres = centres[mask]
    used_np = used_centres.detach().cpu().numpy()
    used_lat = used_np[:, 0]
    used_lon = used_np[:, 1]
    region_bounds = {
        "lat": (float(used_lat.min()), float(used_lat.max())),
        "lon": (float(used_lon.min()), float(used_lon.max())),
    }
    extent = _bounds_to_extent(region_bounds)

    # Convert region_array into torch.Tensor
    region_latents = torch.from_numpy(region_array)
    patch_rows = row_end - row_start + 1
    patch_cols = col_end - col_start + 1

    return (
        region_latents,
        patch_rows,
        patch_cols,
        region_bounds,
        extent,
        latitudes,
        longitudes,
        atmos_levels,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode precomputed regional Aurora latents and plot them "
            "on global + zoomed maps."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/run_single_forward_latents_region"),
        help=(
            "Root directory used by run_single_forward_with_latents_region.py "
            "(containing a 'latents' subdirectory)."
        ),
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        # required=True,
        default="2020-01-01T18-00-00",
        help=(
            "Timestamp label inside the HDF5 file, e.g. 2020-01-01T12-00-00 "
            "(must match the label written by the forward script)."
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["surface", "atmos"],
        default="surface",
        help="Which latents to decode.",
    )
    parser.add_argument(
        "--var-name",
        type=str,
        default="2t",
        help="Variable to decode, e.g. '2t' for surface or 't' for atmos.",
    )
    parser.add_argument(
        "--latents-file-name",
        type=str,
        default="pressure_surface_latents_region.h5",
        help="Name of the HDF5 file under <work-dir>/latents/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    model.eval()

    latents_dir = args.work_dir.expanduser() / "latents"
    patch_grid_path = latents_dir / "patch_grid.pt"
    latents_h5 = latents_dir / args.latents_file_name

    if not patch_grid_path.is_file():
        raise FileNotFoundError(
            f"patch_grid.pt not found at {patch_grid_path}")
    if not latents_h5.is_file():
        raise FileNotFoundError(f"Latents HDF5 file not found at {latents_h5}")

    print(f"Loading regional latents from {latents_h5}")
    (
        region_latents,
        patch_rows,
        patch_cols,
        region_bounds,
        extent,
        latitudes,
        longitudes,
        atmos_levels,
    ) = _prepare_region_from_patch_grid(
        patch_grid_path,
        latents_h5,
        args.timestamp,
        args.mode,
    )

    var_names = [args.var_name]
    if args.mode == "surface":
        decoded = _decode_subset(
            model.decoder,
            region_latents.to(device),
            var_names,
            patch_rows,
            patch_cols,
            head_builder=lambda name: model.decoder.surf_heads[name](
                region_latents.to(device)
            ),
            unnormalise=_unnormalise_surface_preds,
        )
        mode_label = "surface"
    else:
        if atmos_levels is None:
            raise RuntimeError(
                "Atmospheric levels not available in patch_grid metadata; "
                "re-run the forward script after adding 'atmos_levels'."
            )

        decoded = _decode_subset(
            model.decoder,
            region_latents.to(device),
            var_names,
            patch_rows,
            patch_cols,
            head_builder=lambda name: model.decoder.atmos_heads[name](
                region_latents.to(device), levels=atmos_levels.to(device)
            ),
            unnormalise=lambda tensor, names: _unnormalise_atmos_preds(
                tensor.squeeze(2), names, atmos_levels
            ),
        )
        mode_label = "atmos"

    print(f"Decoded {mode_label} variable {args.var_name}: {decoded.shape}")

    region_field = decoded[0, 0]

    color_limits = _compute_color_limits(region_field)

    _plot_region_only(
        region_field[None, None, ...],  # shape (1,1,H,W) for consistency
        extent,
        mode_label,
        args.var_name,
        None,  # timestamp (optional)
        color_limits=color_limits,
    )


if __name__ == "__main__":
    main()
