import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from huggingface_hub import hf_hub_download

import aurora as aurora_module
from aurora.batch import Batch
from aurora.model.posencoding import lat_lon_meshgrid, patch_root_area
from torch.serialization import safe_globals
import types
from typing import Callable
import warnings


LATENTS_DIR = Path("examples/latents")
DEFAULT_STATIC_REPO = "microsoft/aurora"
DEFAULT_STATIC_FILENAME = "aurora-0.25-static.pickle"
DEFAULT_STATIC_CANDIDATES: tuple[Path, ...] = (
    Path("examples/downloads/era5/static.nc"),
    Path("examples/downloads/hres/static.nc"),
)


def ensure_static_dataset(
    cache_dir: Path,
    repo_id: str = DEFAULT_STATIC_REPO,
    filename: str = DEFAULT_STATIC_FILENAME,
) -> Path:
    """Download Aurora static fields and materialize them as ``static.nc``."""

    cache_dir.mkdir(parents=True, exist_ok=True)

    for candidate in DEFAULT_STATIC_CANDIDATES:
        candidate = candidate.expanduser()
        if _is_usable_static_file(candidate):
            return candidate

    static_path = cache_dir / "static.nc"
    if _is_usable_static_file(static_path):
        return static_path
    if static_path.exists():
        static_path.unlink()

    pickle_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )

    with open(pickle_path, "rb") as handle:
        static_vars = pickle.load(handle)

    latitudes = np.linspace(90, -90, 721)
    longitudes = np.linspace(0, 360, 1440, endpoint=False)
    dataset = xr.Dataset(
        data_vars={
            name: (("latitude", "longitude"), values)
            for name, values in static_vars.items()
        },
        coords={
            "latitude": ("latitude", latitudes),
            "longitude": ("longitude", longitudes),
        },
    )
    dataset.to_netcdf(static_path, engine="scipy")
    return static_path


def _is_usable_static_file(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path, engine="netcdf4") as ds:  # type: ignore[misc]
            print(f'Loading {path} with netcdf4 engine')
            ds.load()
        return True
    except Exception:
        try:
            with xr.open_dataset(path, engine="scipy") as ds:
                print(f'Loading {path} with scipy engine')
                ds.load()
            return True
        except Exception:
            return False


@dataclass
class LatentsLoadResult:
    captures: dict[str, torch.Tensor]
    patch_grid: dict[str, Any]
    prediction: Batch
    surface_latents: torch.Tensor
    deaggregated_atmos_latents: torch.Tensor
    global_surface_field: np.ndarray
    latitudes: np.ndarray
    longitudes: np.ndarray


def expand_patch_bounds(
    lat_min: torch.Tensor,
    lat_max: torch.Tensor,
    lon_min: torch.Tensor,
    lon_max: torch.Tensor,
    lat_step: torch.Tensor,
    lon_step: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extend patch bounds to cover the full cell by adding half a grid spacing."""
    lat_half = lat_step.to(lat_min.dtype) / 2
    lon_half = lon_step.to(lon_min.dtype) / 2

    lat_min = torch.clamp(lat_min - lat_half, min=-90.0, max=90.0)
    lat_max = torch.clamp(lat_max + lat_half, min=-90.0, max=90.0)
    lon_min = lon_min - lon_half
    lon_max = lon_max + lon_half

    return lat_min, lat_max, lon_min, lon_max


def _pool2d(tensor: torch.Tensor, pool_fn, kernel: tuple[int, int]) -> torch.Tensor:
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    pooled = pool_fn(tensor, kernel)
    return pooled[0, 0]


def compute_patch_grid(
    lat: torch.Tensor, lon: torch.Tensor, patch_size: int
) -> dict[str, torch.Tensor | int | tuple[int, int]]:
    """Recreate the patch centres and bounds that the encoder uses for every latent."""
    if lat.dim() == lon.dim() == 1:
        grid = lat_lon_meshgrid(lat, lon)
    elif lat.dim() == lon.dim() == 2:
        grid = torch.stack((lat, lon), dim=0)
    else:
        raise ValueError(
            "Lat and lon must both be vectors or both be matrices.")

    patch_dims = (patch_size, patch_size)
    lat_grid = grid[0].to(torch.float32)
    lon_grid = grid[1].to(torch.float32)

    centres = torch.stack(
        (
            _pool2d(lat_grid, F.avg_pool2d, patch_dims),
            _pool2d(lon_grid, F.avg_pool2d, patch_dims),
        ),
        dim=-1,
    )

    lat_max = _pool2d(lat_grid, F.max_pool2d, patch_dims)
    lat_min = -_pool2d(-lat_grid, F.max_pool2d, patch_dims)
    lon_max = _pool2d(lon_grid, F.max_pool2d, patch_dims)
    lon_min = -_pool2d(-lon_grid, F.max_pool2d, patch_dims)

    lat_step = (
        (lat_grid[1:, :] - lat_grid[:-1, :]).abs().mean()
        if lat_grid.shape[0] > 1
        else lat_grid.new_tensor(0.0)
    )
    lon_step = (
        (lon_grid[:, 1:] - lon_grid[:, :-1]).abs().mean()
        if lon_grid.shape[1] > 1
        else lon_grid.new_tensor(0.0)
    )

    lat_min, lat_max, lon_min, lon_max = expand_patch_bounds(
        lat_min, lat_max, lon_min, lon_max, lat_step, lon_step
    )

    bounds = torch.stack((lat_min, lat_max, lon_min, lon_max), dim=-1)
    root_area = patch_root_area(lat_min, lon_min, lat_max, lon_max)

    grid_height, grid_width = centres.shape[:2]
    total_patches = grid_height * grid_width

    return {
        "centres": centres.reshape(total_patches, 2),
        "bounds": bounds.reshape(total_patches, 4),
        "root_area": root_area.reshape(total_patches),
        "patch_shape": (grid_height, grid_width),
        "patch_size": patch_size,
        "patch_count": total_patches,
    }


def load_model(device: torch.device) -> aurora_module.Aurora:
    model = aurora_module.AuroraSmallPretrained()
    model = model.to(device)
    model.load_checkpoint()
    model.eval()
    return model


def load_latents_info_and_grid(
    latents_path: Path,
) -> LatentsLoadResult:
    with safe_globals([Batch, datetime]):
        captures_prediction_and_grid = torch.load(
            latents_path, map_location="cpu", weights_only=False)

    captures = captures_prediction_and_grid.get("captures")
    patch_grid = captures_prediction_and_grid.get("patch_grid")
    preds = captures_prediction_and_grid.get("prediction")

    surface_latents = captures["decoder.surface_latents"]
    deaggregated_atmos_latents = captures["decoder.deaggregated_atmospheric_latents"]
    global_surface_field = preds.surf_vars["2t"][0].detach().cpu().numpy()
    latitudes = preds.metadata.lat.detach().cpu().numpy()
    longitudes = preds.metadata.lon.detach().cpu().numpy()

    return LatentsLoadResult(
        captures=captures,
        patch_grid=patch_grid,
        prediction=preds,
        surface_latents=surface_latents,
        deaggregated_atmos_latents=deaggregated_atmos_latents,
        global_surface_field=global_surface_field,
        latitudes=latitudes,
        longitudes=longitudes,
    )


def format_latents_filename(timestamp: datetime) -> str:
    """Serialize metadata time to a filename-safe string."""
    return f"aurora_latents_{timestamp.strftime('%Y-%m-%dT%H-%M-%S')}.pt"


def find_latest_latents_file(
    latents_dir: Path | None = None,
    pattern: str = "aurora_latents*.pt",
) -> Path:
    """Return the most recently modified latents checkpoint."""
    directory = (latents_dir or LATENTS_DIR).expanduser()
    if not directory.exists():
        raise FileNotFoundError(
            f"Latents directory {directory} does not exist.")

    candidates = sorted(
        directory.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No latents files matching {pattern!r} inside {directory}.")

    return candidates[0]

def register_latent_hooks(model: aurora_module.Aurora, captures: dict[str, torch.Tensor]):
    """Attach hooks so we capture the required latent snapshots."""
    handles = []

    def make_forward_hook(name: str):
        def hook(_, __, output):
            captures[name] = output.detach().cpu()

        return hook

    handles.append(model.encoder.register_forward_hook(
        make_forward_hook("after_encoder")))
    handles.append(model.backbone.register_forward_hook(
        make_forward_hook("after_processor")))

    def decoder_callback(latents: dict[str, torch.Tensor]):
        for key, tensor in latents.items():
            captures[f"decoder.{key}"] = tensor.detach().cpu()

    decoder_cleanup = register_decoder_latent_callback(
        model.decoder, decoder_callback)
    return handles, decoder_cleanup


def register_decoder_latent_callback(
    decoder: torch.nn.Module, callback: Callable[[dict[str, torch.Tensor]], None]
) -> Callable[[], None]:
    """Install `callback` so every emitted decoder latent is captured."""
    if hasattr(decoder, "register_latent_callback"):
        decoder.register_latent_callback(callback)
        return decoder.clear_latent_callback

    original_emit = getattr(decoder, "_emit_latents", None)
    if original_emit is None:
        warnings.warn(
            "Decoder does not expose `_emit_latents`, so decoder latents will not be captured."
        )
        return lambda: None

    def patched_emit(self, **latents: torch.Tensor):
        callback(latents)
        return original_emit(**latents)

    decoder._emit_latents = types.MethodType(patched_emit, decoder)

    def cleanup() -> None:
        decoder._emit_latents = original_emit

    return cleanup
