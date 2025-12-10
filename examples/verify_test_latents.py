#!/usr/bin/env python3
"""Verify latents test pipeline by loading, decoding, and visualizing the Zarr dataset.

This script:
1. Loads the extracted latents Zarr dataset
2. Prints dataset info and statistics  
3. Decodes latents using Aurora decoder (like plot_region_latents_from_zarr.py)
4. Creates visualization plots with proper geographic extent

Based on the working code from plot_region_latents_from_zarr.py.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.util import unpatchify
from aurora.normalisation import unnormalise_atmos_var, unnormalise_surf_var
from examples.init_exploring.utils import load_model
from examples.init_exploring.helpers_plot_region import (
    _compute_color_limits,
    _plot_region_with_location,
    _bounds_to_extent,
)


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
    """Decode a subset of latents using Aurora decoder heads."""
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
    time_idx: int = 0,
    mode: str = "surface",
) -> tuple:
    """Load latents from Zarr dataset.
    
    Args:
        zarr_path: Path to the Zarr store
        time_idx: Integer index to load
        mode: "surface" or "atmos"
    
    Returns:
        Tuple of (latents_tensor, patch_rows, patch_cols, region_bounds, extent, 
                  atmos_levels, actual_time)
    """
    ds = xr.open_zarr(str(zarr_path))
    
    # Get actual time for this index
    actual_time = ds.time.values[time_idx]
    actual_time = np.datetime64(actual_time, 'us').astype(datetime)
    
    # Load latents based on mode
    if mode == "surface":
        # Zarr shape: (lat, lon, channels)
        latent_data = ds.surface_latents.isel(time=time_idx).values
        rows, cols, channels = latent_data.shape
        # Reshape to (1, patches, 1, channels) - decoder expects levels dim
        latents = latent_data.reshape(1, rows * cols, 1, channels)
    else:  # atmos
        # Zarr shape: (level, lat, lon, channels)
        latent_data = ds.pressure_latents.isel(time=time_idx).values
        n_levels, rows, cols, channels = latent_data.shape
        # Reshape to (lat, lon, level, channel) then to (1, patches, levels, channels)
        latent_data = latent_data.transpose(1, 2, 0, 3)  # (lat, lon, level, channel)
        latents = latent_data.reshape(1, rows * cols, n_levels, channels)
    
    latents_tensor = torch.from_numpy(latents).float()
    
    # Get metadata
    patch_rows = ds.attrs["surface_shape"][0]
    patch_cols = ds.attrs["surface_shape"][1]
    
    # Get region bounds for the box on world map
    if "region_lat_min" in ds.attrs:
        region_bounds = {
            "lat": (ds.attrs["region_lat_min"], ds.attrs["region_lat_max"]),
            "lon": (ds.attrs["region_lon_min"], ds.attrs["region_lon_max"]),
        }
    else:  # NOTE: for backwards compatibility with older versions
        region_bounds = {
            "lat": (ds.attrs["lat_min"], ds.attrs["lat_max"]),
            "lon": (ds.attrs["lon_min"], ds.attrs["lon_max"]),
        }
    
    # Calculate extent from ACTUAL lat/lon coordinates
    lat_bounds = ds.lat_bounds.values  # (n_lat, 2) with [min, max]
    lon_bounds = ds.lon_bounds.values  # (n_lon, 2) with [min, max]
    
    # Extent is (lon_min, lon_max, lat_min, lat_max)
    extent = (
        float(lon_bounds[0, 0]),   # First lon patch min
        float(lon_bounds[-1, 1]),  # Last lon patch max
        float(lat_bounds[-1, 0]),  # Last lat patch min (southern)
        float(lat_bounds[0, 1]),   # First lat patch max (northern)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify latents test pipeline")
    parser.add_argument("--latents-zarr", type=Path, required=True, help="Path to latents Zarr")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--var-name", type=str, default="2t", help="Surface variable to decode (default: 2t)")
    parser.add_argument("--atmos-var", type=str, default="t", help="Atmospheric variable to decode (default: t)")
    parser.add_argument("--level", type=int, default=850, help="Pressure level for atmos plot (default: 850 hPa)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LATENTS TEST PIPELINE VERIFICATION")
    print("=" * 60)
    
    # Check if Zarr exists
    print("\n--- Loading Latents Zarr ---")
    if not args.latents_zarr.exists():
        print(f"ERROR: Latents Zarr not found at {args.latents_zarr}")
        return
    
    # Load and print dataset info
    ds = xr.open_zarr(str(args.latents_zarr))
    print(f"Dataset dimensions: {dict(ds.sizes)}")
    print(f"Variables: {list(ds.data_vars)}")
    
    # Check processed timesteps
    processed = ds["processed"].values
    processed = np.nan_to_num(processed, nan=0.0).astype(bool)
    n_processed = int(processed.sum())
    print(f"\nProcessed timesteps: {n_processed} / {len(processed)}")
    
    if n_processed == 0:
        print("ERROR: No timesteps were processed!")
        return
    
    # Print coordinate info
    print(f"\nLat centers: {ds.lat.values[0]:.2f} to {ds.lat.values[-1]:.2f} ({len(ds.lat)} patches)")
    print(f"Lon centers: {ds.lon.values[0]:.2f} to {ds.lon.values[-1]:.2f} ({len(ds.lon)} patches)")
    print(f"Levels: {ds.level.values}")
    print(f"Time range: {ds.time.values[0]} to {ds.time.values[-1]}")
    
    # Print region bounds
    print(f"\nRegion bounds:")
    print(f"  Lat: [{ds.attrs.get('region_lat_min', 'N/A')}, {ds.attrs.get('region_lat_max', 'N/A')}]")
    print(f"  Lon: [{ds.attrs.get('region_lon_min', 'N/A')}, {ds.attrs.get('region_lon_max', 'N/A')}]")
    print(f"Surface shape: {ds.attrs.get('surface_shape', 'N/A')}")
    print(f"Pressure shape: {ds.attrs.get('pressure_shape', 'N/A')}")
    
    ds.close()
    
    # Load model for decoding
    print("\n--- Loading Aurora Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(device)
    model.eval()
    
    # Find first processed timestep
    first_idx = int(np.argmax(processed))
    
    print(f"\n--- Decoding Surface Latents (idx={first_idx}) ---")
    try:
        (
            region_latents,
            patch_rows,
            patch_cols,
            region_bounds,
            extent,
            atmos_levels,
            actual_time,
        ) = load_latents_from_zarr(
            args.latents_zarr,
            time_idx=first_idx,
            mode="surface",
        )
        
        print(f"Loaded latents for {actual_time}")
        print(f"  Shape: {region_latents.shape}")
        print(f"  Patch grid: {patch_rows} x {patch_cols}")
        print(f"  Extent: {extent}")
        
        # Decode surface variable
        var_names = [args.var_name]
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
        
        region_field = decoded[0, 0]
        # Handle case where there's an extra dim from surface levels=1
        if region_field.dim() > 2:
            region_field = region_field.squeeze(0)
        
        print(f"Decoded surface variable {args.var_name}: {decoded.shape}")
        print(f"  Min: {region_field.min().item():.2f}, Max: {region_field.max().item():.2f}")
        
        color_limits = _compute_color_limits(region_field.detach().cpu())
        
        # Use extent-based bounds for consistent alignment
        extent_as_bounds = {
            "lat": (extent[2], extent[3]),
            "lon": (extent[0], extent[1]),
        }
        
        # Plot decoded surface variable with geographic context
        fig = plt.figure(figsize=(16, 6))
        
        # Create the plot using the helper function (saves to examples/latents/)
        # But we also want to save to output_dir, so we'll do a simpler plot here
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        center_lon = float((region_bounds["lon"][0] + region_bounds["lon"][1]) / 2.0)
        
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(central_longitude=center_lon))
        ax1.set_global()
        ax1.add_feature(cfeature.OCEAN, facecolor="#d0e7ff", zorder=0)
        ax1.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=1)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=2)
        ax1.set_title("Region location on global map")
        
        # Draw region box
        from matplotlib.patches import Rectangle
        lon_min, lon_max = region_bounds["lon"]
        lat_min, lat_max = region_bounds["lat"]
        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        ax1.add_patch(rect)
        
        # Right panel: Decoded data
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.add_feature(cfeature.OCEAN, facecolor="#d0e7ff", zorder=0)
        ax2.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=1)
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=2)
        
        prediction_arr = region_field.detach().cpu().numpy()
        im = ax2.imshow(
            prediction_arr,
            extent=extent,
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=color_limits[0],
            vmax=color_limits[1],
            zorder=3,
            alpha=0.9,
        )
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.set_title(f"Decoded {args.var_name} — {actual_time}")
        
        plt.colorbar(im, ax=ax2, orientation="horizontal", pad=0.05, label="K")
        plt.tight_layout()
        
        plot_path = args.output_dir / "decoded_surface_2t.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"ERROR decoding surface latents: {e}")
        import traceback
        traceback.print_exc()
    
    # Decode atmospheric variable
    print(f"\n--- Decoding Atmospheric Latents (idx={first_idx}) ---")
    try:
        (
            region_latents,
            patch_rows,
            patch_cols,
            region_bounds,
            extent,
            atmos_levels,
            actual_time,
        ) = load_latents_from_zarr(
            args.latents_zarr,
            time_idx=first_idx,
            mode="atmos",
        )
        
        print(f"Loaded latents for {actual_time}")
        print(f"  Shape: {region_latents.shape}")
        print(f"  Patch grid: {patch_rows} x {patch_cols}")
        print(f"  Levels: {atmos_levels.numpy()}")
        
        # Decode atmospheric variable
        var_names = [args.atmos_var]
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
        
        print(f"Decoded atmospheric variable {args.atmos_var}: {decoded.shape}")
        
        # Select specific pressure level
        level_idx = list(atmos_levels.numpy()).index(args.level)
        region_field = decoded[0, 0, level_idx]
        print(f"  Selected level {args.level} hPa (index {level_idx})")
        print(f"  Min: {region_field.min().item():.2f}, Max: {region_field.max().item():.2f}")
        
        color_limits = _compute_color_limits(region_field.detach().cpu())
        
        # Plot decoded atmospheric variable
        fig = plt.figure(figsize=(16, 6))
        
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        
        center_lon = float((region_bounds["lon"][0] + region_bounds["lon"][1]) / 2.0)
        
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree(central_longitude=center_lon))
        ax1.set_global()
        ax1.add_feature(cfeature.OCEAN, facecolor="#d0e7ff", zorder=0)
        ax1.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=1)
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=2)
        ax1.set_title("Region location on global map")
        
        # Draw region box
        from matplotlib.patches import Rectangle
        lon_min, lon_max = region_bounds["lon"]
        lat_min, lat_max = region_bounds["lat"]
        rect = Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            transform=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor="red",
            linewidth=2,
        )
        ax1.add_patch(rect)
        
        # Right panel: Decoded data
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.add_feature(cfeature.OCEAN, facecolor="#d0e7ff", zorder=0)
        ax2.add_feature(cfeature.LAND, facecolor="#f0f0f0", zorder=1)
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":", zorder=2)
        
        prediction_arr = region_field.detach().cpu().numpy()
        im = ax2.imshow(
            prediction_arr,
            extent=extent,
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=color_limits[0],
            vmax=color_limits[1],
            zorder=3,
            alpha=0.9,
        )
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.set_title(f"Decoded {args.atmos_var} at {args.level} hPa — {actual_time}")
        
        plt.colorbar(im, ax=ax2, orientation="horizontal", pad=0.05, label="K")
        plt.tight_layout()
        
        plot_path = args.output_dir / f"decoded_atmos_{args.atmos_var}_{args.level}hPa.png"
        plt.savefig(plot_path, dpi=150)
        print(f"  Saved: {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"ERROR decoding atmospheric latents: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ VERIFICATION COMPLETE")
    print(f"  Plots saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
