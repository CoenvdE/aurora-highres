#!/usr/bin/env python3
"""Decode regional latents from Zarr and plot with global context.

This script reads regional latents saved by `run_year_forward_with_latents_zarr.py`,
decodes a single variable, and produces a figure with global context.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import xarray as xr
import zarr

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


def load_latents_from_zarr(
    zarr_path: Path,
    time_idx: int | None = None,
    timestamp: datetime | str | None = None,
    mode: str = "surface",
) -> tuple:
    """Load latents from Zarr dataset.
    
    Args:
        zarr_path: Path to the Zarr store
        time_idx: Integer index to load (if specified)
        timestamp: Datetime or string to select (if time_idx not specified)
        mode: "surface" or "atmos"
    
    Returns:
        Tuple of (latents_tensor, patch_rows, patch_cols, region_bounds, extent, 
                  atmos_levels, actual_time)
    """
    ds = xr.open_zarr(str(zarr_path))
    store = zarr.open(str(zarr_path), mode="r")
    
    # Determine which timestep to load
    if time_idx is not None:
        idx = time_idx
    elif timestamp is not None:
        if isinstance(timestamp, str):
            # Try parsing various formats
            for fmt in ["%Y-%m-%dT%H-%M-%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"]:
                try:
                    timestamp = datetime.strptime(timestamp, fmt)
                    break
                except ValueError:
                    continue
        # Find matching index
        time_values = ds.time.values
        target = np.datetime64(timestamp)
        idx = int(np.argmin(np.abs(time_values - target)))
    else:
        idx = 0  # Default to first timestep
    
    # Get actual time for this index
    actual_time = ds.time.values[idx]
    actual_time = np.datetime64(actual_time, 'us').astype(datetime)
    
    # Load latents based on mode
    if mode == "surface":
        # Shape: (lat, lon, channels)
        latent_data = ds.surface_latents.isel(time=idx).values
        # Reshape to (1, lat*lon, channels) for decoder
        rows, cols, channels = latent_data.shape
        latents = latent_data.reshape(1, rows * cols, channels)
    else:  # atmos
        # Shape: (level, lat, lon, channels)
        latent_data = ds.pressure_latents.isel(time=idx).values
        # Reshape to (1, level, lat*lon, channels) for decoder
        levels, rows, cols, channels = latent_data.shape
        latents = latent_data.reshape(1, levels, rows * cols, channels)
    
    latents_tensor = torch.from_numpy(latents).float()
    
    # Get metadata - support both old (lat_min) and new (region_lat_min) naming
    patch_rows = store.attrs["surface_shape"][0]
    patch_cols = store.attrs["surface_shape"][1]
    
    # Try new naming first, fall back to old
    if "region_lat_min" in store.attrs:
        region_bounds = {
            "lat": (store.attrs["region_lat_min"], store.attrs["region_lat_max"]),
            "lon": (store.attrs["region_lon_min"], store.attrs["region_lon_max"]),
        }
    else:
        region_bounds = {
            "lat": (store.attrs["lat_min"], store.attrs["lat_max"]),
            "lon": (store.attrs["lon_min"], store.attrs["lon_max"]),
        }
    
    extent = _bounds_to_extent(region_bounds)
    atmos_levels = torch.tensor(store.attrs["atmos_levels"])
    
    ds.close()
    
    return (
        latents_tensor,
        patch_rows,
        patch_cols,
        region_bounds,
        extent,
        atmos_levels,
        actual_time,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Aurora latents from Zarr and plot with global context."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("examples/latents_europe_2018_2020.zarr"),
        help="Path to the latents Zarr store",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=None,
        help="Integer index of timestep to plot (0-based)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp to plot, e.g. '2020-01-01T12:00' or '2020-01-01T12-00-00'",
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
    parser.add_argument(
        "--list-timesteps",
        action="store_true",
        help="List available timesteps and exit",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1,
        help="Number of samples to plot (starting from time-idx or first available)",
    )
    return parser.parse_args()


def list_available_timesteps(zarr_path: Path, n_show: int = 20) -> None:
    """Print available timesteps in the Zarr store."""
    ds = xr.open_zarr(str(zarr_path))
    store = zarr.open(str(zarr_path), mode="r")
    
    n_total = len(ds.time)
    processed = store["processed"][:]
    n_processed = int(np.sum(processed))
    
    print(f"\n{'='*60}")
    print(f"Zarr Store: {zarr_path}")
    print(f"Total timesteps: {n_total}")
    print(f"Processed: {n_processed}")
    print(f"{'='*60}")
    
    print(f"\nFirst {min(n_show, n_total)} timesteps:")
    for i in range(min(n_show, n_total)):
        time_val = ds.time.values[i]
        status = "✓" if processed[i] else "✗"
        print(f"  [{i:4d}] {status} {time_val}")
    
    if n_total > n_show:
        print(f"  ... ({n_total - n_show} more timesteps)")
    
    ds.close()


def main() -> None:
    args = parse_args()
    
    if not args.zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {args.zarr_path}")
    
    if args.list_timesteps:
        list_available_timesteps(args.zarr_path)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(device)
    model.eval()

    # Determine starting index
    start_idx = args.time_idx if args.time_idx is not None else 0
    
    for sample_offset in range(args.n_samples):
        current_idx = start_idx + sample_offset
        
        print(f"\n{'='*60}")
        print(f"Processing sample {sample_offset + 1}/{args.n_samples} (idx={current_idx})")
        print(f"{'='*60}")
        
        (
            region_latents,
            patch_rows,
            patch_cols,
            region_bounds,
            extent,
            atmos_levels,
            actual_time,
        ) = load_latents_from_zarr(
            args.zarr_path,
            time_idx=current_idx,
            timestamp=args.timestamp if sample_offset == 0 else None,
            mode=args.mode,
        )
        
        print(f"Loaded latents for {actual_time}")
        print(f"  Shape: {region_latents.shape}")
        print(f"  Patch grid: {patch_rows} x {patch_cols}")

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
        color_limits = _compute_color_limits(region_field.detach().cpu())

        # Plot
        _plot_region_with_location(
            region_field[None, None, ...].detach().cpu(),
            extent,
            region_bounds,
            mode_label,
            args.var_name,
            actual_time,
            color_limits=color_limits,
            origin="upper",  # ERA5 is north to south
        )

        if args.show_patchiness_diagnostic:
            _plot_patchiness_comparison(
                region_field.detach().cpu(),
                extent,
                mode_label,
                args.var_name,
                patch_size=model.decoder.patch_size,
                origin="upper",
            )


if __name__ == "__main__":
    main()

# Usage examples:
#
# List available timesteps:
#   python examples/plot_region_latents_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --list-timesteps
#
# Plot first timestep:
#   python examples/plot_region_latents_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --time-idx 0 --var-name 2t
#
# Plot specific timestamp:
#   python examples/plot_region_latents_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --timestamp "2020-01-01T12:00" --var-name 2t
#
# Plot multiple samples:
#   python examples/plot_region_latents_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --time-idx 0 --n-samples 5
#
# Plot atmospheric variable:
#   python examples/plot_region_latents_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --mode atmos --var-name t --time-idx 0
