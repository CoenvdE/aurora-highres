"""Plotting helpers for decoded region visualisations."""

from __future__ import annotations

import warnings
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle


def _normalize_lon(values: np.ndarray) -> np.ndarray:
    return ((values + 180) % 360) - 180


def _compute_color_limits(*arrays: np.ndarray | torch.Tensor) -> tuple[float, float]:
    flattened: list[np.ndarray] = []
    for array in arrays:
        if array is None:
            continue
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        np_array = np.asarray(array, dtype=np.float64)
        if np_array.size == 0:
            continue
        flattened.append(np_array.ravel())
    if not flattened:
        return (-1.0, 1.0)
    data = np.concatenate(flattened)
    vmin, vmax = np.nanpercentile(data, (2, 98))
    if vmin == vmax:
        offset = 0.5
        return float(vmin - offset), float(vmax + offset)
    return float(vmin), float(vmax)


def _draw_region_box(ax: plt.Axes, bounds: dict[str, tuple[float, float]]) -> None:
    lon_min, lon_max = bounds["lon"]
    lat_min, lat_max = bounds["lat"]
    lon_min = ((lon_min + 180) % 360) - 180
    lon_max = ((lon_max + 180) % 360) - 180
    boxes = []
    if lon_min <= lon_max:
        boxes.append((lon_min, lat_min, lon_max - lon_min, lat_max - lat_min))
    else:
        boxes.append((lon_min, lat_min, 180 - lon_min, lat_max - lat_min))
        boxes.append((-180.0, lat_min, lon_max + 180.0, lat_max - lat_min))
    for x, y, width, height in boxes:
        rect = Rectangle(
            (x, y),
            width,
            height,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="black",
            linewidth=2,
        )
        ax.add_patch(rect)


def _apply_land_ocean_mask(
    ax: plt.Axes,
    ocean_color: str = "#d0e7ff",
    land_color: str = "#f0f0f0",
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in buffer",
            category=RuntimeWarning,
            module="shapely",
        )
        ax.add_feature(cfeature.OCEAN, facecolor=ocean_color, zorder=0)
        ax.add_feature(cfeature.LAND, facecolor=land_color, zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4,
                       linestyle=":", zorder=2)


def _style_map_axis(
    ax: plt.Axes,
    *,
    ocean_color: str,
    land_color: str,
    show_labels: bool,
) -> None:
    _apply_land_ocean_mask(ax, ocean_color=ocean_color, land_color=land_color)
    gridlines = ax.gridlines(
        draw_labels=show_labels,
        linewidth=0.2,
        color="gray",
        linestyle="--",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False


def _plot_world_and_region(
    prediction: torch.Tensor | np.ndarray,
    extent: tuple[float, float, float, float],
    region_bounds: dict[str, tuple[float, float]],
    component: str,
    variable: str,
    time: np.datetime64,
    global_pred: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    color_limits: tuple[float, float] | None = None,
    origin: str | None = None,
) -> None:
    """Plot three panels showing the global forecast and the decoded subset.

    Args:
        origin: Optional override for imshow origin. If None, auto-detect from
            latitudes array. Use "upper" when latitudes decrease (north to south,
            typical for ERA5), "lower" when latitudes increase.
    """
    # Ensure latitude/longitude are NumPy arrays (they may be torch.Tensors).
    if isinstance(latitudes, torch.Tensor):
        latitudes = latitudes.detach().cpu().numpy()
    else:
        latitudes = np.asarray(latitudes)

    if isinstance(longitudes, torch.Tensor):
        longitudes = longitudes.detach().cpu().numpy()
    else:
        longitudes = np.asarray(longitudes)

    time_np = np.datetime64(time)
    if origin is not None:
        global_origin = origin
    else:
        lat_increasing = latitudes[0] < latitudes[-1]
        global_origin = "lower" if lat_increasing else "upper"
    normalized_lon = _normalize_lon(longitudes)
    sort_idx = np.argsort(normalized_lon)
    sorted_lon = normalized_lon[sort_idx]
    sorted_lon = np.asarray(sorted_lon, dtype=np.float64)
    lon_diffs = np.diff(sorted_lon)
    lon_step = float(np.nanmedian(lon_diffs) if lon_diffs.size > 0 else 0.0)
    lon_half_step = lon_step / 2.0 if lon_step > 0 else 0.0
    lat_diffs = np.diff(latitudes)
    lat_step = float(np.nanmedian(np.abs(lat_diffs))
                     if lat_diffs.size > 0 else 0.0)
    lat_half_step = lat_step / 2.0 if lat_step > 0 else 0.0
    global_extent = (
        float(sorted_lon[0] - lon_half_step),
        float(sorted_lon[-1] + lon_half_step),
        float(np.min(latitudes) - lat_half_step),
        float(np.max(latitudes) + lat_half_step),
    )

    if isinstance(prediction, torch.Tensor):
        prediction_arr = prediction.detach().cpu().numpy().squeeze()
    else:
        prediction_arr = np.asarray(prediction).squeeze()
    global_pred_arr = np.asarray(global_pred).squeeze()

    if color_limits is None:
        vmin, vmax = _compute_color_limits(global_pred_arr, prediction_arr)
    else:
        vmin, vmax = map(float, color_limits)
    origin_region = global_origin

    center_lon = float(
        (region_bounds["lon"][0] + region_bounds["lon"][1]) / 2.0)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(20, 8),
        subplot_kw={"projection": ccrs.PlateCarree(
            central_longitude=center_lon)},
    )
    ax_truth, ax_global_pred, ax_region = axes

    _style_map_axis(
        ax_truth,
        ocean_color="#d0e7ff",
        land_color="#f0f0f0",
        show_labels=True,
    )
    _style_map_axis(
        ax_global_pred,
        ocean_color="white",
        land_color="#f7f7f7",
        show_labels=True,
    )
    _style_map_axis(
        ax_region,
        ocean_color="#d0e7ff",
        land_color="#f0f0f0",
        show_labels=True,
    )

    im_full = ax_truth.imshow(
        global_pred_arr,
        extent=global_extent,
        origin=global_origin,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        zorder=2,
        alpha=0.9,
    )
    ax_truth.set_extent(global_extent, crs=ccrs.PlateCarree())
    ax_truth.set_title("Full model prediction (global)")
    _draw_region_box(ax_truth, region_bounds)

    ax_global_pred.set_extent(global_extent, crs=ccrs.PlateCarree())
    ax_global_pred.set_facecolor("white")
    ax_global_pred.set_title("Decoded region over global map")
    ax_global_pred.imshow(
        prediction_arr,
        extent=extent,
        origin=origin_region,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        zorder=4,
        alpha=0.9,
    )
    _draw_region_box(ax_global_pred, region_bounds)
    ax_global_pred.set_xticks([])
    ax_global_pred.set_yticks([])

    im_region = ax_region.imshow(
        prediction_arr,
        extent=extent,
        origin=origin_region,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        alpha=0.9,
    )
    region_title = f"Decoded {component} {variable} — {np.datetime_as_string(time_np, unit='m')}"
    ax_region.set_title(region_title)
    ax_region.set_extent(
        (extent[0], extent[1], extent[2], extent[3]), crs=ccrs.PlateCarree())
    ax_region.set_xticks([])
    ax_region.set_yticks([])

    fig.colorbar(
        im_full,
        ax=ax_truth,
        orientation="horizontal",
        pad=0.04,
        aspect=40,
    ).set_label("Kelvin (K)")
    fig.colorbar(
        im_region,
        ax=ax_region,
        orientation="horizontal",
        pad=0.04,
        aspect=40,
    ).set_label("Kelvin (K)")
    plt.tight_layout()

    output_dir = "examples/latents"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "decoded_region_map.png"))
    plt.show()


def _plot_region_only(
    prediction: torch.Tensor | np.ndarray,
    extent: tuple[float, float, float, float],
    component: str,
    variable: str,
    time: np.datetime64 | None,
    color_limits: tuple[float, float] | None = None,
    origin: str = "upper",
) -> None:
    """Plot a single regional panel (no global context).

    This treats the inputs as purely regional data and avoids any
    assumptions about global grids. It is intended for use when only
    regional latents have been decoded.

    Args:
        origin: Image origin for imshow. Use "upper" when latitudes decrease
            (north to south, typical for ERA5), "lower" when latitudes increase.
    """

    if isinstance(prediction, torch.Tensor):
        prediction_arr = prediction.detach().cpu().numpy().squeeze()
    else:
        prediction_arr = np.asarray(prediction).squeeze()

    if color_limits is None:
        vmin, vmax = _compute_color_limits(prediction_arr)
    else:
        vmin, vmax = map(float, color_limits)

    time_str = (
        np.datetime_as_string(np.datetime64(time), unit="m")
        if time is not None
        else "NaT"
    )
    region_title = f"Decoded {component} {variable} — {time_str}"

    fig, ax_region = plt.subplots(
        1,
        1,
        figsize=(10, 6),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    _style_map_axis(
        ax_region,
        ocean_color="#d0e7ff",
        land_color="#f0f0f0",
        show_labels=True,
    )

    im_region = ax_region.imshow(
        prediction_arr,
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        alpha=0.9,
    )
    ax_region.set_title(region_title)
    ax_region.set_extent(
        (extent[0], extent[1], extent[2], extent[3]),
        crs=ccrs.PlateCarree(),
    )
    ax_region.set_xticks([])
    ax_region.set_yticks([])

    fig.colorbar(
        im_region,
        ax=ax_region,
        orientation="horizontal",
        pad=0.04,
        aspect=40,
    ).set_label("Kelvin (K)")

    plt.tight_layout()

    output_dir = "examples/latents"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "decoded_region_only_map.png"))
    plt.show()


def _bounds_to_extent(bounds: dict[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    return (
        float(bounds["lon"][0]),
        float(bounds["lon"][1]),
        float(bounds["lat"][0]),
        float(bounds["lat"][1]),
    )


def _plot_region_with_location(
    prediction: torch.Tensor | np.ndarray,
    extent: tuple[float, float, float, float],
    region_bounds: dict[str, tuple[float, float]],
    component: str,
    variable: str,
    time: np.datetime64 | None,
    color_limits: tuple[float, float] | None = None,
    origin: str = "upper",
) -> None:
    """Plot regional data with a world map showing the region location.

    This creates a 2-panel figure:
    1. A world map with the region box highlighted (no data, just location)
    2. The decoded regional data zoomed in

    Use this when you only have regional latents and no full global prediction.
    """
    if isinstance(prediction, torch.Tensor):
        prediction_arr = prediction.detach().cpu().numpy().squeeze()
    else:
        prediction_arr = np.asarray(prediction).squeeze()

    if color_limits is None:
        vmin, vmax = _compute_color_limits(prediction_arr)
    else:
        vmin, vmax = map(float, color_limits)

    time_str = (
        np.datetime_as_string(np.datetime64(time), unit="m")
        if time is not None
        else "NaT"
    )

    center_lon = float(
        (region_bounds["lon"][0] + region_bounds["lon"][1]) / 2.0
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=center_lon)},
    )
    ax_world, ax_region = axes

    # Left panel: World map showing region location
    _style_map_axis(
        ax_world,
        ocean_color="#d0e7ff",
        land_color="#f0f0f0",
        show_labels=True,
    )
    ax_world.set_global()
    ax_world.set_title("Region location on global map")
    _draw_region_box(ax_world, region_bounds)

    # Right panel: Decoded regional data
    _style_map_axis(
        ax_region,
        ocean_color="#d0e7ff",
        land_color="#f0f0f0",
        show_labels=True,
    )
    im_region = ax_region.imshow(
        prediction_arr,
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        alpha=0.9,
    )
    region_title = f"Decoded {component} {variable} — {time_str}"
    ax_region.set_title(region_title)
    ax_region.set_extent(
        (extent[0], extent[1], extent[2], extent[3]),
        crs=ccrs.PlateCarree(),
    )

    fig.colorbar(
        im_region,
        ax=ax_region,
        orientation="horizontal",
        pad=0.04,
        aspect=40,
    ).set_label("Kelvin (K)")

    plt.tight_layout()

    output_dir = "examples/latents"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "decoded_region_with_location.png"))
    plt.show()


def _plot_patchiness_comparison(
    prediction: torch.Tensor | np.ndarray,
    extent: tuple[float, float, float, float],
    component: str,
    variable: str,
    patch_size: int,
    origin: str = "upper",
) -> None:
    """Diagnostic plot comparing narrow vs wide color scales and showing patch/pixel sizes.

    Creates a 2x2 figure:
    - Top left: Narrow color range (auto from data) - shows patchiness, with zoom box highlighted
    - Top right: Wide color range (250-300K) - hides patchiness
    - Bottom left: Zoomed crop showing individual pixels with patch grid overlay
    - Bottom right: Legend explaining patch vs pixel size
    """
    if isinstance(prediction, torch.Tensor):
        prediction_arr = prediction.detach().cpu().numpy().squeeze()
    else:
        prediction_arr = np.asarray(prediction).squeeze()

    # Compute narrow (data-driven) and wide (fixed) color limits
    vmin_narrow, vmax_narrow = _compute_color_limits(prediction_arr)
    vmin_wide, vmax_wide = 250.0, 300.0  # Same as Pipeline 1

    H, W = prediction_arr.shape
    patch_rows = H // patch_size
    patch_cols = W // patch_size

    # Calculate zoom region (3x3 patches from middle area)
    crop_patches = 3
    crop_size = crop_patches * patch_size
    start_row = H // 3  # Start from middle-ish area
    start_col = W // 3

    # Convert pixel coordinates to geographic coordinates for the zoom box
    lon_min, lon_max, lat_min, lat_max = extent
    lon_per_pixel = (lon_max - lon_min) / W
    lat_per_pixel = (lat_max - lat_min) / H

    if origin == "upper":
        # Row 0 = lat_max (north), Row H = lat_min (south)
        zoom_lat_max = lat_max - start_row * lat_per_pixel
        zoom_lat_min = lat_max - (start_row + crop_size) * lat_per_pixel
    else:
        # Row 0 = lat_min (south), Row H = lat_max (north)
        zoom_lat_min = lat_min + start_row * lat_per_pixel
        zoom_lat_max = lat_min + (start_row + crop_size) * lat_per_pixel

    zoom_lon_min = lon_min + start_col * lon_per_pixel
    zoom_lon_max = lon_min + (start_col + crop_size) * lon_per_pixel

    fig = plt.figure(figsize=(16, 14))

    # Top left: Narrow color range with zoom box
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
    _style_map_axis(ax1, ocean_color="#d0e7ff", land_color="#f0f0f0", show_labels=True)
    im1 = ax1.imshow(
        prediction_arr,
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin_narrow,
        vmax=vmax_narrow,
        zorder=3,
        alpha=0.9,
    )
    ax1.set_extent(extent, crs=ccrs.PlateCarree())

    # Draw zoom box (yellow rectangle showing where bottom-left panel comes from)
    zoom_rect = Rectangle(
        (zoom_lon_min, zoom_lat_min),
        zoom_lon_max - zoom_lon_min,
        zoom_lat_max - zoom_lat_min,
        transform=ccrs.PlateCarree(),
        facecolor="none",
        edgecolor="yellow",
        linewidth=3,
        linestyle="-",
        zorder=10,
    )
    ax1.add_patch(zoom_rect)
    ax1.annotate(
        "ZOOM\nAREA",
        xy=(zoom_lon_min, zoom_lat_max),
        xytext=(zoom_lon_min - 5, zoom_lat_max + 3),
        fontsize=10,
        fontweight="bold",
        color="yellow",
        transform=ccrs.PlateCarree(),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
    )

    ax1.set_title(
        f"Narrow color range ({vmin_narrow:.1f} - {vmax_narrow:.1f} K)\n"
        f"Range: {vmax_narrow - vmin_narrow:.1f} K — PATCHINESS VISIBLE\n"
        f"Yellow box = zoomed area in bottom-left"
    )
    fig.colorbar(im1, ax=ax1, orientation="horizontal", pad=0.05).set_label("Kelvin (K)")

    # Top right: Wide color range
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    _style_map_axis(ax2, ocean_color="#d0e7ff", land_color="#f0f0f0", show_labels=True)
    im2 = ax2.imshow(
        prediction_arr,
        extent=extent,
        origin=origin,
        transform=ccrs.PlateCarree(),
        cmap="coolwarm",
        vmin=vmin_wide,
        vmax=vmax_wide,
        zorder=3,
        alpha=0.9,
    )
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.set_title(
        f"Wide color range ({vmin_wide:.0f} - {vmax_wide:.0f} K)\n"
        f"Range: {vmax_wide - vmin_wide:.0f} K — PATCHINESS HIDDEN"
    )
    fig.colorbar(im2, ax=ax2, orientation="horizontal", pad=0.05).set_label("Kelvin (K)")

    # Bottom left: Zoomed crop showing pixels and patch grid
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Use same crop region as defined above for the zoom box
    crop = prediction_arr[start_row:start_row + crop_size, start_col:start_col + crop_size]
    
    im3 = ax3.imshow(crop, cmap="coolwarm", vmin=vmin_narrow, vmax=vmax_narrow, interpolation='nearest')
    
    # Draw patch boundaries (thick red lines)
    for i in range(crop_patches + 1):
        ax3.axhline(i * patch_size - 0.5, color='red', linewidth=2, linestyle='-')
        ax3.axvline(i * patch_size - 0.5, color='red', linewidth=2, linestyle='-')
    
    # Draw pixel boundaries (thin gray lines) for first patch only
    for i in range(patch_size + 1):
        ax3.axhline(i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax3.axvline(i - 0.5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    
    ax3.set_title(
        f"Zoomed view: {crop_patches}x{crop_patches} patches (yellow box on map)\n"
        f"Location: {zoom_lat_min:.1f}°-{zoom_lat_max:.1f}°N, {zoom_lon_min:.1f}°-{zoom_lon_max:.1f}°E\n"
        f"Red lines = patch boundaries ({patch_size}x{patch_size} pixels each)"
    )
    ax3.set_xlabel("Pixels (longitude direction)")
    ax3.set_ylabel("Pixels (latitude direction)")
    fig.colorbar(im3, ax=ax3, orientation="horizontal", pad=0.1).set_label("Kelvin (K)")

    # Bottom right: Size legend / info
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate geographic sizes
    lon_range = extent[1] - extent[0]
    lat_range = extent[3] - extent[2]
    pixel_lon_size = lon_range / W
    pixel_lat_size = lat_range / H
    patch_lon_size = lon_range / patch_cols
    patch_lat_size = lat_range / patch_rows
    
    info_text = f"""
    IMAGE DIMENSIONS
    ────────────────────────────────
    Total size: {H} x {W} pixels
    
    PATCH INFO
    ────────────────────────────────
    Patch size: {patch_size} x {patch_size} pixels
    Number of patches: {patch_rows} x {patch_cols} = {patch_rows * patch_cols}
    
    GEOGRAPHIC SIZE
    ────────────────────────────────
    1 pixel ≈ {pixel_lat_size:.3f}° lat x {pixel_lon_size:.3f}° lon
           ≈ {pixel_lat_size * 111:.1f} km x {pixel_lon_size * 111 * 0.7:.1f} km (at 45°N)
    
    1 patch ≈ {patch_lat_size:.2f}° lat x {patch_lon_size:.2f}° lon
           ≈ {patch_lat_size * 111:.0f} km x {patch_lon_size * 111 * 0.7:.0f} km (at 45°N)
    
    WHY PATCHINESS?
    ────────────────────────────────
    Each patch is decoded INDEPENDENTLY
    by a linear projection head.
    No smoothing across patch boundaries.
    
    Narrow color range amplifies small
    boundary discontinuities.
    """
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f"Patchiness Diagnostic: {component} {variable}", fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_dir = "examples/latents"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "patchiness_comparison.png"), dpi=150)
    plt.show()


__all__ = [
    "_bounds_to_extent",
    "_compute_color_limits",
    "_plot_world_and_region",
    "_plot_region_only",
    "_plot_region_with_location",
    "_plot_patchiness_comparison",
]
