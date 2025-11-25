"""Decode latents for a custom region and overlay the decoded field on a Cartopy map."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.util import unpatchify
from aurora.normalisation import (
    unnormalise_atmos_var,
    unnormalise_surf_var,
)
from examples.utils import (
    find_latest_latents_file,
    load_model,
    load_latents_info_and_grid,
)
from examples.helpers_plot_region import (
    _bounds_to_extent,
    _compute_color_limits,
    _plot_world_and_region,
)

def get_patch_window_from_coord_bounds(patch_grid: dict, bounds: dict[str, tuple[float, float]]) -> tuple[int, int, int, int]:
    grid_h, grid_w = patch_grid["patch_shape"]
    lat_min, lat_max = bounds["lat"]
    lon_min, lon_max = bounds["lon"]

    bounds_arr = np.asarray(patch_grid["bounds"]).reshape(grid_h, grid_w, 4)
    patch_lat_min = bounds_arr[..., 0]
    patch_lat_max = bounds_arr[..., 1]
    patch_lon_min = bounds_arr[..., 2]
    patch_lon_max = bounds_arr[..., 3]

    lat_mask = (patch_lat_max >= lat_min) & (patch_lat_min <= lat_max)
    # TODO: take into account longitude wrapping
    lon_mask = (patch_lon_max >= lon_min) & (patch_lon_min <= lon_max)
    mask = lat_mask & lon_mask

    row_idx = np.where(mask.any(axis=1))[0]
    col_idx = np.where(mask.any(axis=0))[0]
    if row_idx.size == 0 or col_idx.size == 0:
        raise RuntimeError("No patches overlap the requested region.")
    return row_idx[0], row_idx[-1], col_idx[0], col_idx[-1]


def _select_region_latents(
    latents: torch.Tensor,
    patch_shape: tuple[int, int],
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
) -> tuple[torch.Tensor, int, int]:
    grid_height, grid_width = patch_shape
    batch_size, _, levels, embed_dim = latents.shape
    patch_rows = row_end - row_start + 1
    patch_cols = col_end - col_start + 1
    latents = latents.reshape(batch_size, grid_height,
                              grid_width, levels, embed_dim)
    latents = latents[:, row_start:row_end + 1, col_start:col_end + 1, :, :]
    latents = latents.reshape(
        batch_size, patch_rows * patch_cols, levels, embed_dim)
    return latents, patch_rows, patch_cols


def _prepare_region_selection(
    mode: str,
    requested_bounds: dict[str, tuple[float, float]],
    patch_grid: dict,
    *,
    surface_latents: torch.Tensor | None = None,
    atmos_latents: torch.Tensor | None = None,
) -> tuple[torch.Tensor, int, int, dict[str, tuple[float, float]], tuple[float, float, float, float]]:
    """Slice latents and derive plotting metadata for the requested region."""
    row_start, row_end, col_start, col_end = get_patch_window_from_coord_bounds(
        patch_grid, requested_bounds)
    patch_shape = tuple(patch_grid["patch_shape"])

    bounds_arr = np.asarray(patch_grid["bounds"]).reshape(
        patch_shape[0], patch_shape[1], 4
    )
    selected_bounds = bounds_arr[
        row_start:row_end + 1, col_start:col_end + 1, :
    ]
    lat_min = float(selected_bounds[..., 0].min())
    lat_max = float(selected_bounds[..., 1].max())
    lon_min = float(selected_bounds[..., 2].min())
    lon_max = float(selected_bounds[..., 3].max())
    region_bounds = {"lat": (lat_min, lat_max), "lon": (lon_min, lon_max)}
    extent = _bounds_to_extent(region_bounds)

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

    return region_latents, patch_rows, patch_cols, region_bounds, extent


def _run_decoded_patches(
    decoder: Perceiver3DDecoder,
    latents: torch.Tensor,
    var_names: Sequence[str],
    patch_rows: int,
    patch_cols: int,
    head_builder: Callable[[str], torch.Tensor],
) -> torch.Tensor:

    patch_size = decoder.patch_size
    total_height = patch_rows * patch_size
    total_width = patch_cols * patch_size
    head_outputs = torch.stack(
        [head_builder(name) for name in var_names],
        dim=-1,
    )
    head_outputs = head_outputs.reshape(*head_outputs.shape[:3], -1)
    preds = unpatchify(head_outputs, len(var_names),
                       total_height, total_width, patch_size)
    print(f"preds shape: {preds.shape}")
    return preds


def _decode_subset(
    decoder: Perceiver3DDecoder,
    latents: torch.Tensor,
    var_names: Sequence[str],
    patch_rows: int,
    patch_cols: int,
    head_builder: Callable[[str], torch.Tensor],
    unnormalise: Callable[[torch.Tensor, Sequence[str]], torch.Tensor],
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    preds = _run_decoded_patches(
        decoder,
        latents,
        var_names,
        patch_rows,
        patch_cols,
        head_builder=head_builder,
    )
    if postprocess is not None:
        preds = postprocess(preds)
    return unnormalise(preds, var_names)


def _decode_surface_subset(
    decoder: Perceiver3DDecoder,
    latents: torch.Tensor,
    surf_vars: Sequence[str],
    patch_rows: int,
    patch_cols: int,
) -> torch.Tensor:
    return _decode_subset(
        decoder,
        latents,
        surf_vars,
        patch_rows,
        patch_cols,
        head_builder=lambda name: decoder.surf_heads[name](latents),
        unnormalise=_unnormalise_surface_preds,
    )


def _decode_atmos_subset(
    decoder: Perceiver3DDecoder,
    latents: torch.Tensor,
    atmos_vars: Sequence[str],
    patch_rows: int,
    patch_cols: int,
    batch: Batch,
) -> torch.Tensor:
    levels = batch.metadata.atmos_levels
    use_levels = decoder.level_condition is not None
    return _decode_subset(
        decoder,
        latents,
        atmos_vars,
        patch_rows,
        patch_cols,
        head_builder=lambda name: decoder.atmos_heads[name](
            latents, levels=levels
        ) if use_levels else decoder.atmos_heads[name](latents),
        unnormalise=lambda tensor, names: _unnormalise_atmos_preds(
            tensor.squeeze(2), names, batch
        ),
    )


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
    batch: Batch,
) -> torch.Tensor:
    levels = batch.metadata.atmos_levels
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


def main(latents_path: Path | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    model.eval()

    region_bounds = {
        "lat": (30.0, 70.0),    
        "lon": (-30.0, 50.0),
    }

    latents_path = latents_path or find_latest_latents_file()
    loaded = load_latents_info_and_grid(latents_path)
    preds = loaded.prediction
    patch_grid = loaded.patch_grid
    global_pred = loaded.global_surface_field
    latitudes = loaded.latitudes
    longitudes = loaded.longitudes
    surface_latents = loaded.surface_latents
    deagg_atmospheric_latents = loaded.deaggregated_atmos_latents

    mode = "surface"
    region_latents, patch_rows, patch_cols, region_bounds, extent = _prepare_region_selection(
        mode,
        region_bounds,
        patch_grid,
        surface_latents=surface_latents,
        atmos_latents=deagg_atmospheric_latents,
    )

    surface_vars = ["2t"]
    atmos_vars = ["t"]
    if mode == "surface":
        region_pred = _decode_surface_subset(
            model.decoder,
            region_latents,
            surface_vars,
            patch_rows,
            patch_cols,
        )
    elif mode == "atmos":
        region_pred = _decode_atmos_subset(
            model.decoder,
            region_latents,
            atmos_vars,
            patch_rows,
            patch_cols,
            preds,
        )
    else:
        raise ValueError(f"Invalid mode: {mode}")
    print(f"Decoded {mode} variables: {region_pred.shape}")

    _plot_world_and_region(
        region_pred,
        extent,
        region_bounds,
        mode,
        "2t",
        preds.metadata.time[0],
        global_pred,
        latitudes,
        longitudes,
        color_limits=_compute_color_limits(global_pred, region_pred)
    )


if __name__ == "__main__":
    main()

# usage:
# python -m examples.decode_deag_latents_region
