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
) -> None:
    """Plot three panels showing the global forecast and the decoded subset."""
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


def _bounds_to_extent(bounds: dict[str, tuple[float, float]]) -> tuple[float, float, float, float]:
    return (
        float(bounds["lon"][0]),
        float(bounds["lon"][1]),
        float(bounds["lat"][0]),
        float(bounds["lat"][1]),
    )


__all__ = [
    "_bounds_to_extent",
    "_compute_color_limits",
    "_plot_world_and_region",
]
