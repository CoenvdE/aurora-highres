#!/usr/bin/env python3
"""Decode regional latents from Zarr and plot in side-by-side comparison style.

This script reads regional latents saved by `run_year_forward_with_latents_zarr.py`,
decodes them, and visualizes them using the same style as the run_simple_forward_visualize scripts:
- 2 panels side-by-side: Decoded Prediction and Ground Truth (ERA5)
- Fixed color scale (-10 to 30°C)
- Simple Cartopy styling (no ocean coloring, just coastlines and borders)
- Regional extent only (no global context map)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# Force unbuffered output for SLURM job logs
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.util import unpatchify
from aurora.normalisation import unnormalise_atmos_var, unnormalise_surf_var
from examples.init_exploring.utils import load_model

# Try to import Cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not found. Plotting without map projections.")

# Default output directory
OUTPUT_DIR = Path("examples/forward_pass_plots")


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
        latent_data = ds.surface_latents.isel(time=idx).values
        rows, cols, channels = latent_data.shape
        latents = latent_data.reshape(1, rows * cols, 1, channels)
    else:  # atmos
        latent_data = ds.pressure_latents.isel(time=idx).values
        n_levels, rows, cols, channels = latent_data.shape
        latent_data = latent_data.transpose(1, 2, 0, 3)
        latents = latent_data.reshape(1, rows * cols, n_levels, channels)
    
    latents_tensor = torch.from_numpy(latents).float()
    
    # Get metadata
    patch_rows = ds.attrs["surface_shape"][0]
    patch_cols = ds.attrs["surface_shape"][1]
    
    # Get region bounds
    if "region_lat_min" in ds.attrs:
        region_bounds = {
            "lat": (ds.attrs["region_lat_min"], ds.attrs["region_lat_max"]),
            "lon": (ds.attrs["region_lon_min"], ds.attrs["region_lon_max"]),
        }
    else:
        region_bounds = {
            "lat": (ds.attrs["lat_min"], ds.attrs["lat_max"]),
            "lon": (ds.attrs["lon_min"], ds.attrs["lon_max"]),
        }
    
    # Calculate extent from actual lat/lon coordinates
    lat_values = ds.lat.values
    lon_values = ds.lon.values
    lat_bounds = ds.lat_bounds.values
    lon_bounds = ds.lon_bounds.values
    
    extent = (
        float(lon_bounds[0, 0]),
        float(lon_bounds[-1, 1]),
        float(lat_bounds[-1, 0]),
        float(lat_bounds[0, 1]),
    )
    
    atmos_levels = torch.tensor(ds.attrs["atmos_levels"])
    
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


def load_era5_ground_truth(
    zarr_path: str,
    time: datetime,
    var_name: str,
    region_bounds: dict,
    mode: str = "surface",
    level: int = 850,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Load ERA5 ground truth from the global Zarr dataset.
    
    Args:
        zarr_path: Path to ERA5 Zarr store
        time: Timestamp to load
        var_name: Variable name (e.g. '2t', 't')
        region_bounds: Dict with 'lat' and 'lon' bounds as tuples
        mode: 'surface' or 'atmos'
        level: Pressure level (for atmos mode)
    
    Returns:
        Tuple of (data_array, extent)
    """
    ds = xr.open_zarr(zarr_path)
    
    # Convert time to numpy datetime64
    target_time = np.datetime64(time)
    
    # Select the timestep
    ds_time = ds.sel(time=target_time, method='nearest')
    
    # Get latitude and longitude
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    # Extract region bounds
    lat_min, lat_max = region_bounds['lat']
    lon_min, lon_max = region_bounds['lon']
    
    # Wrap longitude to 0-360 if needed (ERA5 uses 0-360)
    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360
    
    # Select region
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    
    # Load data based on mode
    if mode == "surface":
        # Map variable names (2t -> t2m, etc.)
        var_map = {'2t': 't2m', '10u': 'u10', '10v': 'v10', 'msl': 'msl'}
        era5_var = var_map.get(var_name, var_name)
        
        data = ds_time[era5_var].values
        data_region = data[lat_mask, :][:, lon_mask]
    else:  # atmos
        # Map variable names if needed
        var_map = {'t': 't', 'u': 'u', 'v': 'v', 'q': 'q', 'z': 'z'}
        era5_var = var_map.get(var_name, var_name)
        
        # Select pressure level
        ds_level = ds_time.sel(pressure_level=level)
        data = ds_level[era5_var].values
        data_region = data[lat_mask, :][:, lon_mask]
    
    # Calculate extent (convert back to -180 to 180 for plotting)
    lon_region = lon[lon_mask]
    lat_region = lat[lat_mask]
    
    # Wrap back to -180 to 180
    lon_region_wrapped = ((lon_region + 180) % 360) - 180
    sort_idx = np.argsort(lon_region_wrapped)
    lon_region_sorted = lon_region_wrapped[sort_idx]
    data_region_sorted = data_region[:, sort_idx]
    
    extent = (
        float(lon_region_sorted[0]),
        float(lon_region_sorted[-1]),
        float(lat_region[0]),
        float(lat_region[-1]),
    )
    
    ds.close()
    
    return data_region_sorted, extent


def visualize_comparison(
    decoded_data: np.ndarray,
    era5_data: np.ndarray,
    extent: tuple[float, float, float, float],
    var_name: str,
    time: datetime,
    output_dir: Path,
    mode: str = "surface",
    level: int | None = None,
    vmin: float = -10.0,
    vmax: float = 30.0,
) -> None:
    """Visualize decoded latents vs ERA5 in side-by-side comparison.
    
    Uses the same style as run_simple_forward_visualize scripts:
    - 2 panels: Prediction (decoded) and Ground Truth (ERA5)
    - Fixed color scale
    - Simple Cartopy styling
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to Celsius if temperature
    if var_name in ['2t', 't', 't2m']:
        decoded_celsius = decoded_data - 273.15
        era5_celsius = era5_data - 273.15
    else:
        decoded_celsius = decoded_data
        era5_celsius = era5_data
    
    # Create figure
    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            1, 2,
            figsize=(14, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    data_list = [
        (decoded_celsius, f"Decoded Prediction: {time}"),
        (era5_celsius, f"Ground Truth (ERA5): {time}")
    ]
    
    for ax, (data, title) in zip(axes, data_list):
        if HAS_CARTOPY:
            # Simple styling - no ocean coloring like the visualize scripts
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=2)
            ax.coastlines(resolution='50m', zorder=3)
            ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
            
            im = ax.imshow(
                data,
                extent=extent,
                transform=ccrs.PlateCarree(),
                origin="upper",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                interpolation='none',  # No interpolation
            )
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            im = ax.imshow(
                data,
                extent=extent,
                origin="upper",
                cmap="coolwarm",
                vmin=vmin,
                vmax=vmax,
                interpolation='none',
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, label="Temperature (°C)")
    
    # Create title
    if mode == "atmos" and level is not None:
        suptitle = f"Decoded Latents vs ERA5: {var_name} at {level} hPa"
    else:
        suptitle = f"Decoded Latents vs ERA5: {var_name}"
    
    plt.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    # Save with timestamp
    time_str = time.strftime("%Y-%m-%dT%H-%M-%S")
    if mode == "atmos" and level is not None:
        filename = f"latents_comparison_{mode}_{var_name}_{level}hPa_{time_str}.png"
    else:
        filename = f"latents_comparison_{mode}_{var_name}_{time_str}.png"
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode Aurora latents from Zarr and plot side-by-side comparison with ERA5."
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        default=Path("examples/latents_europe_2018_2020.zarr"),
        help="Path to the latents Zarr store",
    )
    parser.add_argument(
        "--era5-zarr",
        type=str,
        default="/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2",
        help="Path to ERA5 Zarr store for ground truth",
    )
    parser.add_argument(
        "--time-idx",
        type=int,
        default=0,
        help="Integer index of timestep to plot (0-based)",
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
        "--level",
        type=int,
        default=850,
        help="Pressure level to plot for atmos mode (default: 850 hPa)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory for plots",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-10.0,
        help="Min temperature for color scale (°C)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=30.0,
        help="Max temperature for color scale (°C)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if not args.zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {args.zarr_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(device)
    model.eval()
    
    print(f"\nLoading latents from {args.zarr_path}")
    print(f"Time index: {args.time_idx}")
    print(f"Mode: {args.mode}, Variable: {args.var_name}")
    
    # Load latents
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
        time_idx=args.time_idx,
        mode=args.mode,
    )
    
    print(f"\nLoaded latents for {actual_time}")
    print(f"  Shape: {region_latents.shape}")
    print(f"  Patch grid: {patch_rows} x {patch_cols}")
    print(f"  Region: lat {region_bounds['lat']}, lon {region_bounds['lon']}")
    
    # Decode latents
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
    else:
        decoded = _decode_subset(
            model.decoder,
            region_latents.to(device),
            var_names,
            patch_rows,
            patch_cols,
            head_builder=lambda name: model.decoder.atmos_heads[name](
                region_latents.to(device)
            ),
            unnormalise=lambda tensor, names: _unnormalise_atmos_preds(
                tensor.squeeze(2), names, atmos_levels
            ),
        )
    
    print(f"\nDecoded {args.mode} variable {args.var_name}: {decoded.shape}")
    
    # Extract the appropriate data
    if args.mode == "atmos":
        level_idx = list(atmos_levels.numpy()).index(args.level)
        decoded_field = decoded[0, 0, level_idx].cpu().numpy()
        print(f"  Selected level {args.level} hPa (index {level_idx})")
    else:
        decoded_field = decoded[0, 0].cpu().numpy()
        if decoded_field.ndim > 2:
            decoded_field = decoded_field.squeeze()
    
    # Load ERA5 ground truth
    print(f"\nLoading ERA5 ground truth from {args.era5_zarr}")
    era5_data, era5_extent = load_era5_ground_truth(
        args.era5_zarr,
        actual_time,
        args.var_name,
        region_bounds,
        mode=args.mode,
        level=args.level if args.mode == "atmos" else None,
    )
    
    print(f"  ERA5 shape: {era5_data.shape}")
    print(f"  ERA5 extent: {era5_extent}")
    
    # Visualize comparison
    print(f"\nCreating comparison visualization...")
    visualize_comparison(
        decoded_data=decoded_field,
        era5_data=era5_data,
        extent=extent,
        var_name=args.var_name,
        time=actual_time,
        output_dir=args.output_dir,
        mode=args.mode,
        level=args.level if args.mode == "atmos" else None,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()

# Usage examples:
#
# Plot surface variable (2m temperature):
#   python examples/plot_latents_comparison_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --time-idx 0 --var-name 2t
#
# Plot atmospheric variable:
#   python examples/plot_latents_comparison_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --mode atmos --var-name t --level 850 --time-idx 0
#
# Custom color scale:
#   python examples/plot_latents_comparison_from_zarr.py \
#       --zarr-path examples/latents_europe_2018_2020.zarr \
#       --time-idx 0 --vmin -20 --vmax 40
