#!/usr/bin/env python3
"""Simple Aurora forward pass with visualization.

This script:
1. Loads ERA5 sample data (already downloaded)
2. Runs a forward pass with AuroraSmallPretrained
3. Visualizes the input and prediction for 2m temperature
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from aurora import Aurora, AuroraSmall, Batch, Metadata

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
    
    return batch.to(device)


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


def visualize_results(batch: Batch, prediction: Batch, output_dir: Path):
    """Visualize the input and prediction for 2m temperature."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get data - use prediction's own coordinates since grids may differ
    input_t2m = batch.surf_vars["2t"][0, 1].cpu().numpy()  # Second timestep of input
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
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Common colorbar range
    vmin = min(input_t2m_c.min(), pred_t2m_c.min())
    vmax = max(input_t2m_c.max(), pred_t2m_c.max())
    
    # Input
    im1 = axes[0].imshow(
        input_t2m_c,
        extent=[lon_in.min(), lon_in.max(), lat_in.min(), lat_in.max()],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin, vmax=vmax
    )
    axes[0].set_title(f"Input: {batch.metadata.time[0]}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="Temperature (°C)")
    
    # Prediction
    im2 = axes[1].imshow(
        pred_t2m_c,
        extent=[lon_pred.min(), lon_pred.max(), lat_pred.min(), lat_pred.max()],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin, vmax=vmax
    )
    axes[1].set_title(f"Prediction: {prediction.metadata.time[0]}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label="Temperature (°C)")
    
    # For difference, interpolate input to prediction grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((lat_in[::-1], lon_in), input_t2m_c[::-1], 
                                      bounds_error=False, fill_value=np.nan)
    lon_grid, lat_grid = np.meshgrid(lon_pred, lat_pred)
    input_interp = interp((lat_grid, lon_grid))
    
    diff = pred_t2m_c - input_interp
    diff_max = np.nanmax(np.abs(diff))
    im3 = axes[2].imshow(
        diff,
        extent=[lon_pred.min(), lon_pred.max(), lat_pred.min(), lat_pred.max()],
        origin="upper",
        cmap="RdBu_r",
        vmin=-diff_max, vmax=diff_max
    )
    axes[2].set_title("Prediction - Input (6h change)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[2], label="Temperature Change (°C)")
    
    plt.suptitle("Aurora Forward Pass: 2m Temperature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_dir / "1_aurora_forward_pass_t2m.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Also create a zoomed Europe view
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
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

    
    # Use interpolated diff for Europe too (also sorted by longitude)
    diff_sorted = diff[:, lon_order_pred]
    diff_europe = diff_sorted[lat_idx_pred[0]:lat_idx_pred[-1]+1, lon_idx_pred[0]:lon_idx_pred[-1]+1]
    
    # Use same color scale as global plots for comparison
    vmin_eu = vmin
    vmax_eu = vmax
    
    im1 = axes[0].imshow(
        input_europe,
        extent=[lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin_eu, vmax=vmax_eu
    )
    axes[0].set_title(f"Input: {batch.metadata.time[0]}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="Temperature (°C)")
    
    im2 = axes[1].imshow(
        pred_europe,
        extent=[lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin_eu, vmax=vmax_eu
    )
    axes[1].set_title(f"Prediction: {prediction.metadata.time[0]}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label="Temperature (°C)")
    
    # Use same diff scale as global plot
    diff_max_eu = diff_max
    im3 = axes[2].imshow(
        diff_europe,
        extent=[lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu],
        origin="upper",
        cmap="RdBu_r",
        vmin=-diff_max_eu, vmax=diff_max_eu
    )
    axes[2].set_title("Prediction - Input (6h change)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[2], label="Temperature Change (°C)")
    
    plt.suptitle("Aurora Forward Pass: 2m Temperature (Europe - Global Scale)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path_eu = output_dir / "1_aurora_forward_pass_t2m_europe_global_scale.png"
    plt.savefig(output_path_eu, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Europe (global scale) visualization saved to: {output_path_eu}")
    
    # Also create Europe view with LOCAL color scales
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Use local min/max for this version
    vmin_eu_local = min(input_europe.min(), pred_europe.min())
    vmax_eu_local = max(input_europe.max(), pred_europe.max())
    
    im1 = axes[0].imshow(
        input_europe,
        extent=[lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin_eu_local, vmax=vmax_eu_local
    )
    axes[0].set_title(f"Input: {batch.metadata.time[0]}")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    plt.colorbar(im1, ax=axes[0], label="Temperature (°C)")
    
    im2 = axes[1].imshow(
        pred_europe,
        extent=[lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu],
        origin="upper",
        cmap="RdYlBu_r",
        vmin=vmin_eu_local, vmax=vmax_eu_local
    )
    axes[1].set_title(f"Prediction: {prediction.metadata.time[0]}")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    plt.colorbar(im2, ax=axes[1], label="Temperature (°C)")
    
    diff_max_eu_local = np.nanmax(np.abs(diff_europe))
    im3 = axes[2].imshow(
        diff_europe,
        extent=[lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu],
        origin="upper",
        cmap="RdBu_r",
        vmin=-diff_max_eu_local, vmax=diff_max_eu_local
    )
    axes[2].set_title("Prediction - Input (6h change)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    plt.colorbar(im3, ax=axes[2], label="Temperature Change (°C)")
    
    plt.suptitle("Aurora Forward Pass: 2m Temperature (Europe - Local Scale)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path_eu_local = output_dir / "1_aurora_forward_pass_t2m_europe.png"
    plt.savefig(output_path_eu_local, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Europe (local scale) visualization saved to: {output_path_eu_local}")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    batch = load_era5_batch(DOWNLOAD_PATH, device)
    
    # Load model
    model = load_model(device)
    
    # Run forward pass
    prediction = run_forward_pass(model, batch)
    
    # Move prediction to CPU for visualization
    prediction = prediction.to("cpu")
    batch = batch.to("cpu")
    
    # Visualize
    visualize_results(batch, prediction, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
