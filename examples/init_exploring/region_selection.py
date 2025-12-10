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


def normalize_lon_to_180(lon: np.ndarray) -> np.ndarray:
    """Convert longitude from 0-360° to -180° to 180° range."""
    return np.where(lon > 180, lon - 360, lon)


def compute_lon_roll_shift(patch_grid: dict) -> int:
    """Compute the number of columns to roll to convert from 0-360° to -180°-180° centered.

    Aurora uses 0-360° longitude format. To convert to -180° to 180°, we need to
    roll the longitude axis so that the 180° meridian becomes the edge instead of 0°.

    Returns the number of columns to roll (for use with np.roll or torch.roll).
    A positive value means roll left (shift data from end to beginning).
    """
    centres = patch_grid["centres"]  # (patch_count, 2) [lat, lon]
    patch_shape = patch_grid["patch_shape"]  # (lat_patches, lon_patches)

    centres_np = centres.detach().cpu().numpy()
    lon_centres = centres_np[:, 1]

    # Reshape to 2D grid to find column indices
    lat_patches, lon_patches = patch_shape
    lon_2d = lon_centres.reshape(lat_patches, lon_patches)

    # Take first row's longitudes (all rows have same lon pattern)
    lon_row = lon_2d[0, :]

    # Find the column index where longitude is closest to 180°
    # We want to roll so that 180° becomes the left edge (in -180 to 180, that's -180)
    idx_180 = np.argmin(np.abs(lon_row - 180))

    # Roll by this amount: columns 0..idx_180-1 stay, columns idx_180..end go to front
    # After rolling, lon=180 (which becomes -180) is at index 0
    return idx_180


def roll_latents_to_180(latents: torch.Tensor, patch_shape: tuple, roll_shift: int) -> torch.Tensor:
    """Roll latents along the longitude dimension to convert from 0-360° to -180°-180° centered.

    Args:
        latents: Shape (batch, patches, levels, embed_dim) where patches = lat*lon
        patch_shape: (lat_patches, lon_patches)
        roll_shift: Number of columns to roll (from compute_lon_roll_shift)

    Returns:
        Rolled latents with same shape, but longitude-ordered as -180° to 180°
    """
    lat_patches, lon_patches = patch_shape
    batch_size, n_patches, levels, embed_dim = latents.shape

    # Reshape to (batch, lat, lon, levels, embed)
    latents_2d = latents.reshape(
        batch_size, lat_patches, lon_patches, levels, embed_dim)

    # Roll along the longitude dimension (dim=2)
    # Negative shift moves data from end to beginning
    latents_rolled = torch.roll(latents_2d, shifts=-roll_shift, dims=2)

    # Reshape back to (batch, patches, levels, embed)
    return latents_rolled.reshape(batch_size, n_patches, levels, embed_dim)


def get_rolled_patch_grid(patch_grid: dict, roll_shift: int) -> dict:
    """Create a new patch_grid with longitudes rolled and converted to -180° to 180°.

    This creates a copy of patch_grid where:
    - Longitude columns are rolled by roll_shift
    - All longitude values are converted from 0-360° to -180° to 180°
    """
    centres = patch_grid["centres"].clone()  # (patch_count, 2) [lat, lon]
    # (patch_count, 4) [lat_min, lat_max, lon_min, lon_max]
    bounds = patch_grid["bounds"].clone()
    patch_shape = patch_grid["patch_shape"]
    lat_patches, lon_patches = patch_shape

    # Reshape to 2D grid
    centres_2d = centres.reshape(lat_patches, lon_patches, 2)
    bounds_2d = bounds.reshape(lat_patches, lon_patches, 4)

    # Roll along longitude dimension
    centres_2d = torch.roll(centres_2d, shifts=-roll_shift, dims=1)
    bounds_2d = torch.roll(bounds_2d, shifts=-roll_shift, dims=1)

    # Convert longitudes from 0-360° to -180° to 180°
    # centres[:, 1] is longitude
    centres_2d[:, :, 1] = torch.where(
        centres_2d[:, :, 1] > 180,
        centres_2d[:, :, 1] - 360,
        centres_2d[:, :, 1]
    )
    # bounds[:, 2] is lon_min, bounds[:, 3] is lon_max
    bounds_2d[:, :, 2] = torch.where(
        bounds_2d[:, :, 2] > 180,
        bounds_2d[:, :, 2] - 360,
        bounds_2d[:, :, 2]
    )
    bounds_2d[:, :, 3] = torch.where(
        bounds_2d[:, :, 3] > 180,
        bounds_2d[:, :, 3] - 360,
        bounds_2d[:, :, 3]
    )

    # Reshape back to flat
    total_patches = lat_patches * lon_patches
    new_grid = patch_grid.copy()
    new_grid["centres"] = centres_2d.reshape(total_patches, 2)
    new_grid["bounds"] = bounds_2d.reshape(total_patches, 4)

    return new_grid


def compute_region_indices_from_bounds(
    requested_bounds: Dict[str, Tuple[float, float]],
    patch_grid: dict,
    *,
    normalize_lon: bool = True,
) -> tuple[int, int, int, int, RegionBounds]:
    """Map geographic bounds to a rectangular window in patch space.

    ``patch_grid`` is expected to be the output of ``compute_patch_grid`` and
    must contain ``"centres"`` (lat/lon of patch centres) and ``"patch_shape"``
    (``lat_patches``, ``lon_patches``).

    Args:
        requested_bounds: Dictionary with 'lat' and 'lon' tuples specifying the region.
        patch_grid: Output of compute_patch_grid or get_rolled_patch_grid.
        normalize_lon: If True (default), normalizes longitude values from 0-360° to
            -180° to 180° internally for matching. Set to False if patch_grid is
            already in -180° to 180° format (e.g., from get_rolled_patch_grid).

    Note:
        When normalize_lon=True, this function works with Aurora's 0-360° format
        but the min/max index approach may not work correctly for regions that
        cross the 0° meridian. For such cases, use get_rolled_patch_grid and
        roll_latents_to_180 first, then set normalize_lon=False.
    """

    centres = patch_grid["centres"]  # (patch_count, 2) [lat, lon]
    patch_shape = patch_grid["patch_shape"]  # (lat_patches, lon_patches)

    lat_min_req, lat_max_req = requested_bounds["lat"]
    lon_min_req, lon_max_req = requested_bounds["lon"]

    centres_np = centres.detach().cpu().numpy()
    lat_centres = centres_np[:, 0]
    lon_centres = centres_np[:, 1]

    # Normalize longitude to -180° to 180° if needed
    if normalize_lon:
        lon_centres = np.where(
            lon_centres > 180, lon_centres - 360, lon_centres)

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

    lat_used = lat_centres[indices]
    lon_used = lon_centres[indices]
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
    normalize_lon: bool = True,
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

    Args:
        mode: Either 'surface' or 'atmos'.
        requested_bounds: Dictionary with 'lat' and 'lon' tuples specifying the region.
        patch_grid: Output of compute_patch_grid or get_rolled_patch_grid.
        surface_latents: Surface latents tensor (required if mode='surface').
        atmos_latents: Atmospheric latents tensor (required if mode='atmos').
        normalize_lon: If True (default), normalizes longitude values internally.
            Set to False if using rolled patch_grid and latents.
    """

    (
        row_start,
        row_end,
        col_start,
        col_end,
        region_bounds,
    ) = compute_region_indices_from_bounds(
        requested_bounds, patch_grid, normalize_lon=normalize_lon
    )

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
    normalize_lon: bool = True,
) -> tuple[torch.Tensor, int, int, RegionBounds, tuple[float, float, float, float]]:
    """Slice latents and derive bounds + extent for decode-time scripts.

    Args:
        mode: Either 'surface' or 'atmos'.
        requested_bounds: Dictionary with 'lat' and 'lon' tuples specifying the region.
        patch_grid: Output of compute_patch_grid or get_rolled_patch_grid.
        surface_latents: Surface latents tensor (required if mode='surface').
        atmos_latents: Atmospheric latents tensor (required if mode='atmos').
        normalize_lon: If True (default), normalizes longitude values internally.
            Set to False if using rolled patch_grid and latents.
    """

    (
        row_start,
        row_end,
        col_start,
        col_end,
        region_bounds,
    ) = compute_region_indices_from_bounds(
        requested_bounds, patch_grid, normalize_lon=normalize_lon
    )

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
