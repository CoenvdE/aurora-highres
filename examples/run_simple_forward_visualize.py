#!/usr/bin/env python3
"""Simple Aurora forward pass with visualization.

This script:
1. Loads ERA5 sample data (already downloaded)
2. Runs a forward pass with AuroraSmallPretrained
3. Visualizes the input and prediction for 2m temperature
"""

import matplotlib
matplotlib.use('Agg')

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


def load_era5_batch(download_path: Path, device: torch.device) -> Batch:
    """Load ERA5 sample data into an Aurora Batch."""
    print("Loading ERA5 data...")
    
    static_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_ds = xr.open_dataset(download_path / "2020-01-01-surface-level.nc", engine="netcdf4")
    atmos_ds = xr.open_dataset(download_path / "2020-01-01-atmospheric.nc", engine="netcdf4")
    
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
    
    return batch.to(device), static_ds


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


def visualize_results(batch: Batch, prediction: Batch, static_ds, output_dir: Path):
    """Visualize regional prediction and ground truth for 2m temperature."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data - use prediction's own coordinates since grids may differ
    input_t2m = batch.surf_vars["2t"][0, 1].cpu().numpy()  # Second timestep of input (ground truth)
    pred_t2m = prediction.surf_vars["2t"][0, 0].cpu().numpy()  # Prediction
    
    # Input grid
    lat_in = batch.metadata.lat.cpu().numpy()
    lon_in = batch.metadata.lon.cpu().numpy()
    
    # Prediction grid (may differ slightly)
    lat_pred = prediction.metadata.lat.cpu().numpy()
    lon_pred = prediction.metadata.lon.cpu().numpy()
    
    # Convert to Celsius
    input_t2m_c = input_t2m - 273.15
    pred_t2m_c = pred_t2m - 273.15
    
    # Europe bounds (same as HRES target region)
    lat_min_eu, lat_max_eu = 30, 70
    lon_min_eu, lon_max_eu = -30, 50
    
    # Wrap longitude from 0-360 to -180-180 for proper Europe slicing
    lon_pred_wrapped = ((lon_pred + 180) % 360) - 180
    lon_in_wrapped = ((lon_in + 180) % 360) - 180
    
    # Sort by wrapped longitude to get correct order
    lon_order_pred = np.argsort(lon_pred_wrapped)
    lon_order_in = np.argsort(lon_in_wrapped)
    
    lon_pred_sorted = lon_pred_wrapped[lon_order_pred]
    lon_in_sorted = lon_in_wrapped[lon_order_in]
    
    # Re-order the data arrays to match sorted longitude
    pred_t2m_c_sorted = pred_t2m_c[:, lon_order_pred]
    input_t2m_c_sorted = input_t2m_c[:, lon_order_in]
    
    # Find indices for prediction grid
    lat_mask_pred = (lat_pred >= lat_min_eu) & (lat_pred <= lat_max_eu)
    lon_mask_pred = (lon_pred_sorted >= lon_min_eu) & (lon_pred_sorted <= lon_max_eu)
    lat_idx_pred = np.where(lat_mask_pred)[0]
    lon_idx_pred = np.where(lon_mask_pred)[0]
    
    # Find indices for input grid
    lat_mask_in = (lat_in >= lat_min_eu) & (lat_in <= lat_max_eu)
    lon_mask_in = (lon_in_sorted >= lon_min_eu) & (lon_in_sorted <= lon_max_eu)
    lat_idx_in = np.where(lat_mask_in)[0]
    lon_idx_in = np.where(lon_mask_in)[0]
    
    input_europe = input_t2m_c_sorted[lat_idx_in[0]:lat_idx_in[-1]+1, lon_idx_in[0]:lon_idx_in[-1]+1]
    pred_europe = pred_t2m_c_sorted[lat_idx_pred[0]:lat_idx_pred[-1]+1, lon_idx_pred[0]:lon_idx_pred[-1]+1]
    
    # Use consistent global color scale for comparability
    vmin_eu = -10  # °C - typical winter minimum for Europe
    vmax_eu = 30   # °C - typical summer maximum for Europe
    
    # Create figure with 2 panels: Prediction and Ground Truth
    extent = [lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu]
    
    if HAS_CARTOPY:
        fig, axes = plt.subplots(
            1, 2, 
            figsize=(14, 6),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    data_list = [
        (pred_europe, f"Prediction: {prediction.metadata.time[0]}"),
        (input_europe, f"Ground Truth: {batch.metadata.time[0]}")
    ]
    
    for ax, (data, title) in zip(axes, data_list):
        if HAS_CARTOPY:
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=2)
            ax.coastlines(resolution='50m', zorder=3)
            ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
            im = ax.imshow(
                data,
                extent=extent,
                transform=ccrs.PlateCarree(),
                origin="upper",
                cmap="coolwarm",
                vmin=vmin_eu, vmax=vmax_eu
            )
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            im = ax.imshow(
                data,
                extent=extent,
                origin="upper",
                cmap="coolwarm",
                vmin=vmin_eu, vmax=vmax_eu
            )
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        ax.set_title(title)
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, fraction=0.05, label="Temperature (°C)")
    
    plt.suptitle("Aurora Forward Pass: 2m Temperature (Europe)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_dir / "1_aurora_forward_pass_t2m_europe.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    batch, static_ds = load_era5_batch(DOWNLOAD_PATH, device)
    
    # Load model
    model = load_model(device)
    
    # Run forward pass
    prediction = run_forward_pass(model, batch)
    
    # Move prediction to CPU for visualization
    prediction = prediction.to("cpu")
    batch = batch.to("cpu")
    
    # Visualize
    visualize_results(batch, prediction, static_ds, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
