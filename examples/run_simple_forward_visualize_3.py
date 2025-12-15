#!/usr/bin/env python3
"""Simple Aurora forward pass with Cartopy visualization.

This script:
1. Loads ERA5 sample data
2. Runs a forward pass with AuroraSmallPretrained
3. Visualizes predictions with Cartopy (coastlines, borders, projections)
4. Supports regional plotting with configurable bounds

Based on the Cartopy plotting logic with proper geographic projections.
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from aurora import Aurora, AuroraSmall, Batch, Metadata

# Try to import Cartopy
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("WARNING: Cartopy not found. Plotting without map projections.")

# Path to downloaded ERA5 data
DOWNLOAD_PATH = Path("examples/downloads/era5")
OUTPUT_DIR = Path("examples/forward_pass_plots")


def load_era5_batch(download_path: Path, device: torch.device) -> tuple[Batch, xr.Dataset]:
    """Load ERA5 sample data into an Aurora Batch.
    
    Returns:
        batch: Aurora Batch for model input
        surf_ds: Surface dataset for ground truth comparison
    """
    print("Loading ERA5 data...")
    
    static_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_ds = xr.open_dataset(download_path / "2018-01-01-surface-level.nc", engine="netcdf4")
    atmos_ds = xr.open_dataset(download_path / "2018-01-01-atmospheric.nc", engine="netcdf4")
    
    # Take first two timesteps (00:00 and 06:00) for history
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_ds["t2m"].values[:2][None]),
            "10u": torch.from_numpy(surf_ds["u10"].values[:2][None]),
            "10v": torch.from_numpy(surf_ds["v10"].values[:2][None]),
            "msl": torch.from_numpy(surf_ds["msl"].values[:2][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_ds["z"].values[0]),
            "slt": torch.from_numpy(static_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_ds["t"].values[:2][None]),
            "u": torch.from_numpy(atmos_ds["u"].values[:2][None]),
            "v": torch.from_numpy(atmos_ds["v"].values[:2][None]),
            "q": torch.from_numpy(atmos_ds["q"].values[:2][None]),
            "z": torch.from_numpy(atmos_ds["z"].values[:2][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values),
            lon=torch.from_numpy(surf_ds.longitude.values),
            time=(surf_ds.valid_time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=tuple(int(level) for level in atmos_ds.pressure_level.values),
        ),
    )
    
    print(f"  Input time: {batch.metadata.time[0]}")
    print(f"  Grid shape: {batch.metadata.lat.shape[0]} x {batch.metadata.lon.shape[0]}")
    print(f"  Atmospheric levels: {batch.metadata.atmos_levels}")
    
    return batch.to(device), surf_ds


def load_model(device: torch.device) -> Aurora:
    """Load the small pretrained Aurora model."""
    print("\nLoading AuroraSmall pretrained model...")
    model = AuroraSmall()
    model = model.to(device)
    model.load_checkpoint("microsoft/aurora", "aurora-0.25-small-pretrained.ckpt")
    model.eval()
    print("  Model loaded successfully!")
    return model


def run_forward_pass(model: Aurora, batch: Batch) -> Batch:
    """Run a single forward pass."""
    print("\nRunning forward pass...")
    with torch.no_grad():
        prediction = model(batch)
    print(f"  Prediction time: {prediction.metadata.time[0]}")
    return prediction


def plot_temperature_cartopy(
    temp_data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    title: str,
    output_path: Path,
    region: tuple = None,
    vmin: float = None,
    vmax: float = None,
):
    """Plot temperature using Cartopy with coastlines and borders.
    
    Args:
        temp_data: 2D temperature array (lat, lon) in Kelvin
        lat: Latitude coordinates
        lon: Longitude coordinates  
        title: Plot title
        output_path: Output file path
        region: Optional (lon_min, lon_max, lat_min, lat_max) for regional plot
        vmin, vmax: Optional color scale bounds
    """
    # Convert to Celsius
    temp_celsius = temp_data - 273.15
    
    # Handle Longitude Wrapping (0..360 -> -180..180) and Sorting
    lon_wrapped = ((lon + 180) % 360) - 180
    lon_order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[lon_order]
    temp_sorted = temp_celsius[:, lon_order]
    
    # Region Slicing
    extent = None
    if region:
        lon_min, lon_max, lat_min, lat_max = region
        
        # Find indices for latitude and longitude
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon_sorted >= lon_min) & (lon_sorted <= lon_max))[0]
        
        if len(lat_idx) > 0 and len(lon_idx) > 0:
            lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
            lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)
            
            temp_plot = temp_sorted[lat_slice, lon_slice]
            
            current_lats = lat[lat_slice]
            current_lons = lon_sorted[lon_slice]
            
            extent = [current_lons[0], current_lons[-1], current_lats[-1], current_lats[0]]
        else:
            print("Region selection resulted in empty array. Plotting global.")
            temp_plot = temp_sorted
            extent = [lon_sorted[0], lon_sorted[-1], lat[-1], lat[0]]
    else:
        temp_plot = temp_sorted
        extent = [lon_sorted[0], lon_sorted[-1], lat[-1], lat[0]]

    # Plotting
    fig = plt.figure(figsize=(12, 8))
    
    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=2)
        ax.coastlines(resolution='50m', zorder=3)
        ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
        
        im = ax.imshow(
            temp_plot, 
            extent=extent, 
            transform=ccrs.PlateCarree(),
            cmap='coolwarm', 
            origin='upper', 
            vmin=vmin, 
            vmax=vmax
        )
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax = plt.axes()
        im = ax.imshow(
            temp_plot, 
            extent=extent, 
            cmap='coolwarm', 
            origin='upper', 
            aspect='auto', 
            vmin=vmin, 
            vmax=vmax
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05)
    cbar.set_label("Temperature (°C)")
    plt.title(title)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving plot to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_comparison_cartopy(
    pred_data: np.ndarray,
    era5_data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    pred_time: str,
    output_path: Path,
    region: tuple = None,
    vmin: float = None,
    vmax: float = None,
):
    """Plot Aurora prediction vs ERA5 with Cartopy.
    
    Creates a 2-panel figure: Prediction and Ground Truth.
    """
    # Convert to Celsius
    pred_celsius = pred_data - 273.15
    era5_celsius = era5_data - 273.15
    
    # Handle Longitude Wrapping
    lon_wrapped = ((lon + 180) % 360) - 180
    lon_order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[lon_order]
    pred_sorted = pred_celsius[:, lon_order]
    era5_sorted = era5_celsius[:, lon_order]
    
    # Region Slicing
    if region:
        lon_min, lon_max, lat_min, lat_max = region
        lat_idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
        lon_idx = np.where((lon_sorted >= lon_min) & (lon_sorted <= lon_max))[0]
        
        if len(lat_idx) > 0 and len(lon_idx) > 0:
            lat_slice = slice(lat_idx[0], lat_idx[-1] + 1)
            lon_slice = slice(lon_idx[0], lon_idx[-1] + 1)
            
            pred_plot = pred_sorted[lat_slice, lon_slice]
            era5_plot = era5_sorted[lat_slice, lon_slice]
            
            current_lats = lat[lat_slice]
            current_lons = lon_sorted[lon_slice]
            extent = [current_lons[0], current_lons[-1], current_lats[-1], current_lats[0]]
        else:
            pred_plot, era5_plot = pred_sorted, era5_sorted
            extent = [lon_sorted[0], lon_sorted[-1], lat[-1], lat[0]]
    else:
        pred_plot, era5_plot = pred_sorted, era5_sorted
        extent = [lon_sorted[0], lon_sorted[-1], lat[-1], lat[0]]

    # Auto color scale if not provided
    if vmin is None:
        vmin = min(pred_plot.min(), era5_plot.min())
    if vmax is None:
        vmax = max(pred_plot.max(), era5_plot.max())

    # Create figure with 2 panels
    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            1, 2, 
            figsize=(14, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    data_list = [
        (pred_plot, "Prediction", 'coolwarm', vmin, vmax),
        (era5_plot, "Ground Truth (ERA5)", 'coolwarm', vmin, vmax),
    ]
    
    for ax, (data, title, cmap, v_min, v_max) in zip(axes, data_list):
        if HAS_CARTOPY:
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=2)
            ax.coastlines(resolution='50m', zorder=3)
            ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
            im = ax.imshow(
                data, 
                extent=extent, 
                transform=ccrs.PlateCarree(),
                cmap=cmap, 
                origin='upper', 
                vmin=v_min, 
                vmax=v_max
            )
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            im = ax.imshow(
                data, 
                extent=extent, 
                cmap=cmap, 
                origin='upper', 
                aspect='auto',
                vmin=v_min, 
                vmax=v_max
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, label="°C")
    
    plt.suptitle(f"Aurora Forward Pass: 2m Temperature - {pred_time}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving comparison plot to {output_path}...")
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Aurora forward pass with Cartopy visualization")
    parser.add_argument("--output-dir", type=str, default="examples/forward_pass_plots")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # Default to Europe region
    parser.add_argument("--lon-min", type=float, default=-15.0)
    parser.add_argument("--lon-max", type=float, default=45.0)
    parser.add_argument("--lat-min", type=float, default=30.0)
    parser.add_argument("--lat-max", type=float, default=75.0)
    parser.add_argument("--vmin", type=float, default=-10.0, help="Min temperature for color scale (°C)")
    parser.add_argument("--vmax", type=float, default=30.0, help="Max temperature for color scale (°C)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    
    # Region settings (always regional)
    region = (args.lon_min, args.lon_max, args.lat_min, args.lat_max)
    region_suffix = "europe"
    
    # Load data
    batch, surf_ds = load_era5_batch(DOWNLOAD_PATH, device)
    
    # Load model
    model = load_model(device)
    
    # Run forward pass
    prediction = run_forward_pass(model, batch)
    
    # Move to CPU for visualization
    prediction = prediction.to("cpu")
    batch = batch.to("cpu")
    
    # Get lat/lon from prediction
    lat_pred = prediction.metadata.lat.numpy()
    lon_pred = prediction.metadata.lon.numpy()
    
    # Get prediction data
    pred_t2m = prediction.surf_vars["2t"][0, 0].numpy()
    
    # Get ERA5 ground truth (timestep 2 corresponds to prediction)
    era5_t2m_raw = surf_ds["t2m"][2].values
    lat_era5 = surf_ds.latitude.values
    lon_era5 = surf_ds.longitude.values
    
    # Use 720 latitude points directly (no interpolation)
    # ERA5 has 721 points, prediction has 720 - just use first 720 from ERA5
    if era5_t2m_raw.shape[0] == 721 and pred_t2m.shape[0] == 720:
        print(f"  Grid mismatch: ERA5 {era5_t2m_raw.shape} vs Prediction {pred_t2m.shape}")
        print("  Using first 720 latitude points from ERA5 (no interpolation)")
        era5_t2m = era5_t2m_raw[:720, :]
    else:
        era5_t2m = era5_t2m_raw
    
    # Plot single prediction
    plot_temperature_cartopy(
        temp_data=pred_t2m,
        lat=lat_pred,
        lon=lon_pred,
        title=f"Aurora Prediction: 2m Temperature - {prediction.metadata.time[0]}",
        output_path=output_dir / f"3_aurora_cartopy_pred_{region_suffix}.png",
        region=region,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    
    # Plot comparison (Prediction vs ERA5 vs Error)
    plot_comparison_cartopy(
        pred_data=pred_t2m,
        era5_data=era5_t2m,
        lat=lat_pred,
        lon=lon_pred,
        pred_time=str(prediction.metadata.time[0]),
        output_path=output_dir / f"3_aurora_cartopy_comparison_{region_suffix}.png",
        region=region,
        vmin=args.vmin,
        vmax=args.vmax,
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
