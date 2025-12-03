#!/usr/bin/env python3
"""Decode *regional* deaggregated latents from a single forward pass.

This script complements ``run_single_forward_with_latents_regional.py`` by
loading one timestamp's regional latent tensor from the HDF5 dataset and
decoding it back to a physical field for visual inspection, reusing the
helpers from ``examples/init_exploring/decode_deag_latents_region.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from aurora.batch import Batch
from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.init_exploring.utils import ensure_static_dataset, load_model
from examples.init_exploring.decode_deag_latents_region import (
    _decode_atmos_subset,
)
from examples.init_exploring.helpers_plot_region import (
    _bounds_to_extent,
    _compute_color_limits,
    _plot_world_and_region,
)


DEFAULT_ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/"
    "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode and visualise regional deaggregated latents from a "
            "single forward pass."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("examples/run_single_forward_latents_regional"),
        help="Root directory where regional latents were written",
    )
    parser.add_argument(
        "--latents-file",
        type=Path,
        default=None,
        help=(
            "Path to HDF5 file with regional latents. If omitted, "
            "defaults to <work-dir>/latents/deaggregated_latents_regional.h5"
        ),
    )
    parser.add_argument(
        "--timestamp-label",
        type=str,
        default=None,
        help=(
            "Timestamp label to decode (e.g. 2020-01-01T12-00-00). "
            "If omitted, the latest label in the file is used."
        ),
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=DEFAULT_ZARR_PATH,
        help="Path to the ERA5 Zarr archive (for global context)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default="2020-01-01",
        help="Date (YYYY-MM-DD) to extract at 12Z for context",
    )
    parser.add_argument(
        "--var",
        type=str,
        default="t",
        help="Atmospheric variable name to decode (e.g. 't')",
    )
    return parser.parse_args()


def _select_timestamp_label(latents_group: h5py.Group, label: str | None) -> str:
    """Choose a timestamp label from the HDF5 ``latents`` group."""

    keys = sorted(latents_group.keys())
    if not keys:
        raise RuntimeError("No latents found in HDF5 file.")
    if label is None:
        return keys[-1]
    if label not in latents_group:
        raise KeyError(
            f"Requested timestamp_label {label!r} not found. "
            f"Available labels: {keys!r}"
        )
    return label


def main() -> None:
    args = parse_args()

    work_dir = args.work_dir.expanduser()
    latents_file = (
        args.latents_file
        if args.latents_file is not None
        else work_dir / "latents" / "deaggregated_latents_regional.h5"
    )

    if not latents_file.exists():
        raise SystemExit(f"Latents file not found: {latents_file}")

    print(f"Loading regional latents from {latents_file}")
    with h5py.File(latents_file, "r") as handle:
        latents_group = handle["latents"]
        timestamp_label = _select_timestamp_label(
            latents_group, args.timestamp_label
        )
        ds = latents_group[timestamp_label]
        region_latents_np = ds[...]
        timestamp_iso = ds.attrs["timestamp"]
        patch_rows = int(ds.attrs["patch_rows"])
        patch_cols = int(ds.attrs["patch_cols"])
        patch_size = int(ds.attrs["patch_size"])
        region_bounds = {
            "lat": (
                float(ds.attrs["region_lat_min"]),
                float(ds.attrs["region_lat_max"]),
            ),
            "lon": (
                float(ds.attrs["region_lon_min"]),
                float(ds.attrs["region_lon_max"]),
            ),
        }

    # Convert to tensor on device for decoding.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    region_latents = torch.from_numpy(region_latents_np).to(device)

    # Load model and a small ERA5 batch to provide atmospheric level metadata
    # and global context for plotting.
    static_path = ensure_static_dataset(work_dir)
    print("Loading context batch from ERA5 Zarr...")
    batch = load_batch_from_zarr(
        zarr_path=args.zarr_path,
        static_path=str(static_path),
        date_str=args.date,
    )

    batch_device = batch.to(device)
    model = load_model(device)
    model.eval()

    # Run a global forward pass to obtain a reference prediction for
    # plotting. Keep this as a full `Batch` so we retain metadata.
    with torch.no_grad():
        print("Running global forward pass for reference field...")
        global_prediction = model(batch_device)

    # Prepare inputs for the decoder helper.
    atmos_vars = [args.var]
    region_latents = region_latents.to(device)

    print("Decoding regional atmospheric field from latents...")
    region_pred = _decode_atmos_subset(
        model.decoder,
        region_latents,
        atmos_vars,
        patch_rows,
        patch_cols,
        batch,
    )

    # Move decoded field to CPU for plotting.
    region_pred = region_pred.to("cpu")
    print(f"Decoded regional field shape: {region_pred.shape}")

    # Collapse decoded regional field to a single 2D map for plotting.
    # region_pred has shape [batch, vars, levels, H, W]; take first batch,
    # first variable, and average over levels.
    region_field = region_pred[0, 0].mean(dim=0)  # [H, W]

    # Extract global reference surface field and coordinates, mirroring
    # decode_deag_latents_region.py's usage pattern. We only need the
    # relevant atmospheric variable from the prediction for plotting.
    var_name = args.var
    if var_name not in global_prediction.atmos_vars:
        raise KeyError(
            f"Variable {var_name!r} not found in model atmospheric outputs. "
            f"Available variables: {sorted(global_prediction.atmos_vars.keys())!r}"
        )

    global_pred = global_prediction.atmos_vars[var_name].to("cpu")
    # global_pred is typically [batch, history, levels, H, W]; collapse to
    # a single 2D map in the same way as the regional field.
    global_field = global_pred[0, 0].mean(dim=0)  # [H, W]

    # Convert coordinates to NumPy arrays for plotting helpers that use NumPy
    # APIs and normalise longitudes to [-180, 180) for Cartopy.
    latitudes = batch.metadata.lat.detach().cpu().numpy()
    longitudes_raw = batch.metadata.lon.detach().cpu().numpy()
    longitudes = ((longitudes_raw + 180.0) % 360.0) - 180.0

    # Derive extent from stored region bounds.
    extent = _bounds_to_extent(region_bounds)

    print(
        f"Plotting regional decode for {args.var!r} at {timestamp_iso} "
        f"with bounds {region_bounds}"
    )

    _plot_world_and_region(
        region_field,
        extent,
        region_bounds,
        "atmos",
        args.var,
        batch.metadata.time[0],
        global_field,
        latitudes,
        longitudes,
        color_limits=_compute_color_limits(global_field, region_field),
    )


if __name__ == "__main__":
    main()

# usage:
# python examples/decode_deag_latents_region_single.py \
#   --work-dir ./output_dir \
#   --timestamp-label 2020-01-01T12-00-00 \
#   --var t
