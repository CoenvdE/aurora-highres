"""Shared helpers for selecting regional latents in patch space.

This centralises the logic for mapping geographic bounds to patch indices
and slicing rectangular regions from latent tensors.
"""

from __future__ import annotations

from typing import Dict, Tuple, TypedDict

import numpy as np
import torch

from examples.init_exploring.helpers_plot_region import _bounds_to_extent


class RegionBounds(TypedDict):
    lat: Tuple[float, float]
    lon: Tuple[float, float]


def select_region_latents(
    latents: torch.Tensor,
    patch_shape: tuple[int, int],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> tuple[torch.Tensor, int, int]:
    """Slice a rectangular subset of latents in patch space.

    Latents are expected to have shape ``(batch, patches, levels, embed_dim)``,
    where ``patches = lat_patches * lon_patches`` and ``patch_shape`` is the
    tuple ``(lat_patches, lon_patches)``.
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


def compute_region_indices_from_bounds(
    requested_bounds: Dict[str, Tuple[float, float]],
    patch_grid: dict,
) -> tuple[int, int, int, int, RegionBounds]:
    """Map geographic bounds to a rectangular window in patch space.

    ``patch_grid`` is expected to be the output of ``compute_patch_grid`` and
    must contain ``"centres"`` (lat/lon of patch centres) and ``"patch_shape"``
    (``lat_patches``, ``lon_patches``).
    
    Handles longitude wraparound: if patch centers are in 0-360° format and
    requested bounds use negative longitudes (like -30° to 50°), the function
    normalizes both to the same coordinate system before matching.
    """

    centres = patch_grid["centres"]  # (patch_count, 2) [lat, lon]
    patch_shape = patch_grid["patch_shape"]  # (lat_patches, lon_patches)

    lat_min_req, lat_max_req = requested_bounds["lat"]
    lon_min_req, lon_max_req = requested_bounds["lon"]

    centres_np = centres.detach().cpu().numpy()
    lat_centres = centres_np[:, 0]
    lon_centres = centres_np[:, 1]
    
    # Handle longitude wraparound: normalize patch centers to -180° to 180° range
    # This ensures negative longitude requests (e.g., -30°) work correctly when
    # patch centers are stored in 0-360° format (where -30° = 330°)
    lon_centres_normalized = np.where(
        lon_centres > 180, lon_centres - 360, lon_centres
    )

    lat_mask = (lat_centres >= lat_min_req) & (lat_centres <= lat_max_req)
    
    # Handle longitude wraparound for regions crossing the prime meridian
    # E.g., Europe from -30° to 50° needs patches from 330°-360° AND 0°-50°
    if lon_min_req < 0 and lon_max_req > 0:
        # Region crosses prime meridian: use OR logic
        # lon_centres_normalized will be in -180 to 180 range
        lon_mask = (lon_centres_normalized >= lon_min_req) & (lon_centres_normalized <= lon_max_req)
    elif lon_min_req > lon_max_req:
        # Alternative wraparound case (e.g., 330° to 50° in 0-360 format)
        lon_mask = (lon_centres_normalized >= lon_min_req) | (lon_centres_normalized <= lon_max_req)
    else:
        # Normal case: no wraparound
        lon_mask = (lon_centres_normalized >= lon_min_req) & (lon_centres_normalized <= lon_max_req)
    
    mask = lat_mask & lon_mask
    if not np.any(mask):
        raise ValueError("Requested bounds do not overlap any patches.")

    indices = np.where(mask)[0]
    lat_indices, lon_indices = np.unravel_index(indices, patch_shape)
    row_start = int(lat_indices.min())
    row_end = int(lat_indices.max())
    col_start = int(lon_indices.min())
    col_end = int(lon_indices.max())

    lat_used = lat_centres[indices]
    # Use normalized longitudes for the region bounds output
    lon_used = lon_centres_normalized[indices]
    region_bounds: RegionBounds = {
        "lat": (float(lat_used.min()), float(lat_used.max())),
        "lon": (float(lon_used.min()), float(lon_used.max())),
    }

    return row_start, row_end, col_start, col_end, region_bounds


def prepare_region_for_capture(
    *,
    mode: str,
    requested_bounds: Dict[str, Tuple[float, float]],
    patch_grid: dict,
    surface_latents: torch.Tensor | None = None,
    atmos_latents: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    int,
    int,
    RegionBounds,
    int,
    int,
    int,
    int,
]:
    """Slice latents and derive bounds + indices for capture-time scripts.

    Returns the region latents, patch grid dimensions and effective region
    bounds, together with the integer patch indices so that downstream code
    can store them on disk.
    """

    (
        row_start,
        row_end,
        col_start,
        col_end,
        region_bounds,
    ) = compute_region_indices_from_bounds(requested_bounds, patch_grid)

    patch_shape = patch_grid["patch_shape"]

    if mode == "surface":
        if surface_latents is None:
            raise ValueError("Surface latents are required for surface mode.")
        region_latents, patch_rows, patch_cols = select_region_latents(
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
        region_latents, patch_rows, patch_cols = select_region_latents(
            atmos_latents,
            patch_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

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


def prepare_region_for_decode(
    *,
    mode: str,
    requested_bounds: Dict[str, Tuple[float, float]],
    patch_grid: dict,
    surface_latents: torch.Tensor | None = None,
    atmos_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, RegionBounds, tuple[float, float, float, float]]:
    """Slice latents and derive bounds + extent for decode-time scripts."""

    (
        row_start,
        row_end,
        col_start,
        col_end,
        region_bounds,
    ) = compute_region_indices_from_bounds(requested_bounds, patch_grid)

    patch_shape = patch_grid["patch_shape"]

    if mode == "surface":
        if surface_latents is None:
            raise ValueError("Surface latents are required for surface mode.")
        region_latents, patch_rows, patch_cols = select_region_latents(
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
        region_latents, patch_rows, patch_cols = select_region_latents(
            atmos_latents,
            patch_shape,
            row_start,
            row_end,
            col_start,
            col_end,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")

    extent = _bounds_to_extent(region_bounds)

    return region_latents, patch_rows, patch_cols, region_bounds, extent
