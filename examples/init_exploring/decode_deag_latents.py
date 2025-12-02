"""Simple decoder that takes `decoder.deaggregated_latents` and converts them to predictions.

Note: this script only supports decoders without level conditioning.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

import aurora as aurora_module
from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.util import unpatchify
from examples.init_exploring.utils import find_latest_latents_file, load_latents_info_and_grid

SURF_PHYS_PATH = Path("examples/downloads/era5/2023-01-01-surface-level.nc")
ATMOS_PHYS_PATH = Path("examples/downloads/era5/2023-01-01-atmospheric.nc")


def _decode_surface(
    decoder: Perceiver3DDecoder,
    surf_latents: torch.Tensor,
    batch: Batch,
) -> torch.Tensor:
    surf_vars = tuple(batch.surf_vars.keys())
    patch_size = decoder.patch_size
    height, width = batch.spatial_shape
    head_outputs = torch.stack(
        [decoder.surf_heads[name](surf_latents[..., :1, :])
         for name in surf_vars],
        dim=-1,
    )
    head_outputs = head_outputs.reshape(*head_outputs.shape[:3], -1)
    surf_preds = unpatchify(head_outputs, len(
        surf_vars), height, width, patch_size)
    surf_preds = surf_preds.squeeze(2)
    return surf_preds


def _decode_atmos(
    decoder: Perceiver3DDecoder,
    deagg_latents: torch.Tensor,
    batch: Batch,
) -> torch.Tensor:
    atmos_vars = tuple(batch.atmos_vars.keys())
    patch_size = decoder.patch_size
    height, width = batch.spatial_shape
    level_conditioned = decoder.level_condition is not None
    if level_conditioned:
        raise RuntimeError(
            "This script only supports decoders without level conditioning.")
    head_inputs = deagg_latents
    head_outputs = torch.stack(
        [decoder.atmos_heads[name](head_inputs) for name in atmos_vars],
        dim=-1,
    )
    head_outputs = head_outputs.reshape(*head_outputs.shape[:3], -1)
    atmos_preds = unpatchify(head_outputs, len(
        atmos_vars), height, width, patch_size)
    return atmos_preds


def _assemble_pred_batch(
    surf_preds: torch.Tensor,
    atmos_preds: torch.Tensor,
    batch: Batch,
) -> Batch:
    surf_vars = list(batch.surf_vars)
    atmos_vars = list(batch.atmos_vars)
    pred_batch = Batch(
        surf_vars={name: surf_preds[:, idx]
                   for idx, name in enumerate(surf_vars)},
        static_vars=batch.static_vars,
        atmos_vars={name: atmos_preds[:, idx]
                    for idx, name in enumerate(atmos_vars)},
        metadata=batch.metadata,
    )
    return pred_batch


def _load_surface_truth() -> xr.Dataset:
    return xr.open_dataset(SURF_PHYS_PATH, engine="h5netcdf")


def _load_atmos_truth() -> xr.Dataset:
    return xr.open_dataset(ATMOS_PHYS_PATH, engine="netcdf4")


def _plot(pred_tensor: torch.Tensor, truth: np.ndarray, timestr: str) -> None:
    decoded_map = pred_tensor[0, 0] - 273.15
    truth_map = truth - 273.15
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im = axes[0].imshow(decoded_map, vmin=-50, vmax=50, cmap="coolwarm")
    axes[0].set_title(f"Aurora decode ({timestr})")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].imshow(truth_map, vmin=-50, vmax=50, cmap="coolwarm")
    axes[1].set_title("ERA5 truth")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes, orientation="horizontal",
                 fraction=0.04, pad=0.08)
    plt.tight_layout()
    plt.show()


def _plot_surface_and_levels(
    pred_batch: Batch,
    surf_vars_ds: xr.Dataset,
    atmos_vars_ds: xr.Dataset,
) -> None:
    time = np.datetime64(pred_batch.metadata.time[0])
    surf_pred = pred_batch.surf_vars["2t"][0].cpu().numpy() - 273.15
    surf_truth = surf_vars_ds["t2m"].sel(valid_time=time).values - 273.15
    levels = pred_batch.metadata.atmos_levels
    atmos_pred = pred_batch.atmos_vars["t"][0].cpu().numpy() - 273.15

    nrows = len(levels) + 1
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 2.3 * nrows))
    axes = np.atleast_2d(axes)

    im = axes[0, 0].imshow(surf_pred, vmin=-50, vmax=35, cmap="coolwarm")
    axes[0, 0].set_title("Decoded 2t (surface)")
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    axes[0, 1].imshow(surf_truth, vmin=-50, vmax=35, cmap="coolwarm")
    axes[0, 1].set_title("ERA5 truth (2t)")
    axes[0, 1].set_xticks([])
    axes[0, 1].set_yticks([])

    for idx, level in enumerate(levels):
        level_truth = atmos_vars_ds["t"].sel(
            valid_time=time, pressure_level=level).values - 273.15
        pred = atmos_pred[idx]
        axes[idx + 1, 0].imshow(pred, vmin=-50, vmax=35, cmap="coolwarm")
        axes[idx + 1, 0].set_title(f"Decoded t at {level} hPa")
        axes[idx + 1, 0].set_xticks([])
        axes[idx + 1, 0].set_yticks([])
        axes[idx + 1, 1].imshow(level_truth, vmin=-50,
                                vmax=35, cmap="coolwarm")
        axes[idx + 1, 1].set_title(f"ERA5 t at {level} hPa")
        axes[idx + 1, 1].set_xticks([])
        axes[idx + 1, 1].set_yticks([])

    fig.colorbar(im, ax=axes[:, :].ravel().tolist(),
                 orientation="horizontal", fraction=0.02, pad=0.03)
    fig.suptitle(f"Surface + temperature levels ({time})", y=0.92)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


def main(latents_path: Path | None = None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = aurora_module.AuroraSmallPretrained().to(device)
    model.load_checkpoint()
    model.eval()

    latents_path = latents_path or find_latest_latents_file()
    loaded = load_latents_info_and_grid(latents_path)
    batch = loaded.prediction
    truth_surf_vars_ds = _load_surface_truth()
    truth_atmos_vars_ds = _load_atmos_truth()

    surf_latents = loaded.surface_latents.to(device)
    deagg_latents = loaded.deaggregated_atmos_latents.to(device)

    print("Surface latents:", tuple(surf_latents.shape))
    print("Deaggregated latents:", tuple(deagg_latents.shape))

    with torch.no_grad():
        surf_preds = _decode_surface(model.decoder, surf_latents, batch)
        atmos_preds = _decode_atmos(model.decoder, deagg_latents, batch)
        pred_batch = _assemble_pred_batch(surf_preds, atmos_preds, batch)
        pred_batch = pred_batch.unnormalise(model.surf_stats)

    print("Decoded surface vars:", tuple(batch.surf_vars.keys()))
    print("Decoded atmospheric vars:", tuple(batch.atmos_vars.keys()))

    time = np.datetime64(batch.metadata.time[0])
    truth_surf = truth_surf_vars_ds["t2m"].sel(valid_time=time).values
    _plot(pred_batch.surf_vars["2t"][:, None], truth_surf, str(time))
    _plot_surface_and_levels(
        pred_batch, truth_surf_vars_ds, truth_atmos_vars_ds)


if __name__ == "__main__":
    main()

# usage:
# python examples/decode_deag_latents.py
