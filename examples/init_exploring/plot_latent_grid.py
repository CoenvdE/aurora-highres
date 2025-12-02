"""Show captured Aurora latents on a world map."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from examples.init_exploring.utils import find_latest_latents_file, load_latents_info_and_grid


def _normalize_lon(lon: np.ndarray) -> np.ndarray:
    return ((lon + 180) % 360) - 180


def _plot_patch_edges(ax: plt.Axes, patch_grid: dict) -> None:
    bounds = np.asarray(patch_grid["bounds"]).reshape(-1, 4)
    polygons = []
    for lat_min, lat_max, lon_min, lon_max in bounds:
        lon_min = _normalize_lon(np.array(lon_min, dtype=np.float64))
        lon_max = _normalize_lon(np.array(lon_max, dtype=np.float64))
        polygons.append(
            [
                (lon_min, lat_min),
                (lon_min, lat_max),
                (lon_max, lat_max),
                (lon_max, lat_min),
            ]
        )
    collection = PolyCollection(
        polygons,
        facecolors="none",
        edgecolors="dimgray",
        linewidths=0.45,
        transform=ccrs.PlateCarree(),
    )
    ax.add_collection(collection)


def _plot_patch_centres(ax: plt.Axes, patch_grid: dict) -> None:
    centres = np.asarray(patch_grid["centres"])
    lats = centres[:, 0]
    lons = _normalize_lon(np.asarray(centres[:, 1], dtype=np.float64))
    ax.scatter(
        lons,
        lats,
        c="green",
        s=1,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )


def main(latents_path: Path | None = None) -> None:
    latents_path = latents_path or find_latest_latents_file()
    patch_grid = load_latents_info_and_grid(latents_path).patch_grid

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    _plot_patch_edges(ax, patch_grid)
    _plot_patch_centres(ax, patch_grid)

    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--")
    ax.set_title(f"Latent centres and bounds ({latents_path.name})", pad=8)
    plt.tight_layout()
    plt.savefig("examples/latents/patch_grid.png")
    plt.show()


if __name__ == "__main__":
    main()

# usage:
# python examples/plot_latent_grid.py
