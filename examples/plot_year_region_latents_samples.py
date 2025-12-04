#!/usr/bin/env python3
"""Plot the first N samples from multi-year regional latent datasets.

This script reads regional latents saved by `run_year_forward_with_latents_region.py`,
decodes a single variable for multiple timestamps, and produces figures for each
sample consisting of a global map with the region overlaid plus a zoomed-in regional map.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch

from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.util import unpatchify
from aurora.normalisation import unnormalise_atmos_var, unnormalise_surf_var
from examples.init_exploring.utils import load_model
from examples.init_exploring.helpers_plot_region import (
    _compute_color_limits,
    _plot_region_with_location,
    _bounds_to_extent,
    _plot_patchiness_comparison,
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


def get_timestamps_from_h5(latents_h5: Path, mode: str, max_samples: int) -> list[str]:
    """Extract the first N timestamp labels from the H5 file."""
    with h5py.File(latents_h5, "r") as handle:
        if mode == "surface":
            group = handle.get("surface_latents_region")
        elif mode == "atmos":
            group = handle.get("pressure_latents_region")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if group is None:
            return []
        
        # Get all timestamps and sort them
        timestamps = sorted(list(group.keys()))
        return timestamps[:max_samples]


def find_h5_files_with_samples(
    latents_dir: Path,
    start_year: int,
    end_year: int,
    mode: str,
    max_samples: int,
) -> list[tuple[Path, str]]:
    """Find H5 files and their timestamps up to max_samples total."""
    samples = []
     
    for year in range(start_year, end_year + 1): #TODO: better to have all years in 1 dataset?
        year_dir = latents_dir / str(year)
        h5_file = year_dir / "pressure_surface_latents_region.h5"
        
        if not h5_file.exists():
            print(f"  Warning: {h5_file} not found, skipping year {year}")
            continue
        
        timestamps = get_timestamps_from_h5(h5_file, mode, max_samples - len(samples))
        for ts in timestamps:
            samples.append((h5_file, ts))
            if len(samples) >= max_samples:
                return samples
    
    return samples


def _load_region_artifacts_from_h5(
    latents_h5: Path,
    patch_grid_path: Path,
    timestamp_label: str,
    mode: str,
):
    """Load regional latents and metadata from H5 file."""
    if not patch_grid_path.is_file():
        raise FileNotFoundError(f"patch_grid.pt not found at {patch_grid_path}")
    if not latents_h5.is_file():
        raise FileNotFoundError(f"Latents HDF5 file not found at {latents_h5}")

    grid_obj = torch.load(patch_grid_path)
    patch_grid = grid_obj["patch_grid"]
    metadata = grid_obj["metadata"]
    lat_shape = metadata["lat_shape"]
    lon_shape = metadata["lon_shape"]
    atmos_levels_meta = metadata.get("atmos_levels")
    atmos_levels = (
        torch.tensor(atmos_levels_meta)
        if atmos_levels_meta is not None
        else None
    )

    centres = patch_grid["centres"]
    patch_shape = patch_grid["patch_shape"]

    # Reshape centres to 2D to determine latitude orientation
    lat_patches, lon_patches = patch_shape
    centres_2d = centres.reshape(lat_patches, lon_patches, 2)
    lat_row_0 = float(centres_2d[0, 0, 0])
    lat_row_last = float(centres_2d[-1, 0, 0])
    lat_decreasing = lat_row_0 > lat_row_last

    centres_np = centres.detach().cpu().numpy()
    lat_centres = centres_np[:, 0]
    lon_centres = centres_np[:, 1]
    lat_min, lat_max = float(lat_centres.min()), float(lat_centres.max())
    lon_min, lon_max = float(lon_centres.min()), float(lon_centres.max())

    latitudes = torch.linspace(lat_min, lat_max, lat_shape[0])
    longitudes = torch.linspace(lon_min, lon_max, lon_shape[0])

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
        region_array = group[timestamp_label][...] #TODO: what does this do

    all_indices = np.arange(centres.shape[0])
    lat_ids, lon_ids = np.unravel_index(all_indices, patch_shape)
    mask = (
        (lat_ids >= row_start)
        & (lat_ids <= row_end)
        & (lon_ids >= col_start)
        & (lon_ids <= col_end)
    )
    used_centres = centres[mask] #TODO: check logic here
    used_np = used_centres.detach().cpu().numpy()
    used_lat = used_np[:, 0]
    used_lon = used_np[:, 1]
    region_bounds = {
        "lat": (float(used_lat.min()), float(used_lat.max())),
        "lon": (float(used_lon.min()), float(used_lon.max())),
    }
    extent = _bounds_to_extent(region_bounds)

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
        patch_shape,
        row_start,
        col_start,
        lat_decreasing,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the first N samples from multi-year regional latent datasets."
        )
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/run_year_forward_latents_region"),
        help=(
            "Root directory used by run_year_forward_with_latents_region.py "
            "(containing a 'latents_dataset' subdirectory)."
        ),
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2018,
        help="First year to look for samples",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2020,
        help="Last year to look for samples",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum number of samples to plot",
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
        "--show-patchiness-diagnostic",
        action="store_true",
        help="Show diagnostic plot comparing narrow vs wide color scales.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    model.eval()

    work_dir = args.work_dir.expanduser()
    latents_dir = work_dir / "latents_dataset"
    patch_grid_path = latents_dir / "patch_grid.pt"

    if not patch_grid_path.exists():
        raise FileNotFoundError(
            f"patch_grid.pt not found at {patch_grid_path}. "
            "Please run run_year_forward_with_latents_region.py first."
        )

    print(f"\nSearching for samples in years {args.start_year}-{args.end_year}...")
    samples = find_h5_files_with_samples(
        latents_dir,
        args.start_year,
        args.end_year,
        args.mode,
        args.max_samples,
    )

    if not samples:
        print("No samples found! Please run run_year_forward_with_latents_region.py first.")
        return

    print(f"Found {len(samples)} sample(s) to plot:\n")
    for i, (h5_file, timestamp) in enumerate(samples, 1):
        print(f"  {i}. {timestamp} (from {h5_file.parent.name}/{h5_file.name})")

    var_names = [args.var_name]

    for sample_idx, (h5_file, timestamp_label) in enumerate(samples, 1):
        print(f"\n{'='*60}")
        print(f"Processing sample {sample_idx}/{len(samples)}: {timestamp_label}")
        print(f"{'='*60}")

        try:
            (
                region_latents,
                patch_rows,
                patch_cols,
                region_bounds,
                extent,
                latitudes,
                longitudes,
                atmos_levels,
                patch_shape,
                row_start,
                col_start,
                lat_decreasing,
            ) = _load_region_artifacts_from_h5(
                h5_file,
                patch_grid_path,
                timestamp_label,
                args.mode,
            )
        except Exception as exc:
            print(f"  ✗ Failed to load artifacts: {exc}")
            continue

        plot_origin = "upper" if lat_decreasing else "lower"

        # Decode the variable
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
                print(f"  ✗ Atmospheric levels not available, skipping")
                continue

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

        print(f"  ✓ Decoded {mode_label} variable {args.var_name}: {decoded.shape}")

        region_field = decoded[0, 0]
        color_limits = _compute_color_limits(region_field.detach().cpu())

        timestamp_dt = datetime.strptime(timestamp_label, "%Y-%m-%dT%H-%M-%S")

        # Plot region with world map
        _plot_region_with_location(
            region_field[None, None, ...].detach().cpu(),
            extent,
            region_bounds,
            mode_label,
            args.var_name,
            timestamp_dt,
            color_limits=color_limits,
            origin=plot_origin,
        )

        # Optional diagnostic plot
        if args.show_patchiness_diagnostic:
            _plot_patchiness_comparison(
                region_field.detach().cpu(),
                extent,
                mode_label,
                args.var_name,
                patch_size=model.decoder.patch_size,
                origin=plot_origin,
            )

    print(f"\n{'='*60}")
    print(f"✓ Successfully plotted {len(samples)} sample(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# Usage examples:
#
# Plot first 3 samples (default):
#   python examples/plot_year_region_latents_samples.py \
#       --start-year 2018 --end-year 2020
#
# Plot first 5 samples of a specific variable:
#   python examples/plot_year_region_latents_samples.py \
#       --start-year 2018 --end-year 2020 \
#       --max-samples 5 --var-name "10u"
#
# Plot atmospheric variable:
#   python examples/plot_year_region_latents_samples.py \
#       --mode atmos --var-name "t" --max-samples 3
#
# With patchiness diagnostic:
#   python examples/plot_year_region_latents_samples.py \
#       --max-samples 3 --show-patchiness-diagnostic --start-year 2018 --end-year 2020 --var-name "2t"
