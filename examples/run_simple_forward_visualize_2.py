#!/usr/bin/env python3
"""Simple Aurora forward pass with visualization (simplified version).

This script uses the simple visualization logic from the Aurora demo:
- Fixed color scale (vmin=-50, vmax=50 for temperature)
- No geographic extent (just raw array visualization)
- Side-by-side comparison of Aurora prediction vs ERA5

This script:
1. Loads ERA5 sample data (already downloaded)
2. Runs multiple forward passes with AuroraSmallPretrained
3. Visualizes predictions vs ERA5 ground truth
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from aurora import Aurora, AuroraSmall, Batch, Metadata

# Path to downloaded ERA5 data
DOWNLOAD_PATH = Path("examples/downloads/era5")
OUTPUT_DIR = Path("examples/forward_pass_plots")


def load_era5_data(download_path: Path):
    """Load ERA5 datasets."""
    print("Loading ERA5 data...")
    
    static_ds = xr.open_dataset(download_path / "static.nc", engine="netcdf4")
    surf_ds = xr.open_dataset(download_path / "2020-01-01-surface-level.nc", engine="netcdf4")
    atmos_ds = xr.open_dataset(download_path / "2020-01-01-atmospheric.nc", engine="netcdf4")
    
    return static_ds, surf_ds, atmos_ds


def create_batch(static_ds, surf_ds, atmos_ds, time_idx: int, device: torch.device) -> Batch:
    """Create a batch from ERA5 data starting at time_idx.
    
    Args:
        time_idx: Starting index for the two timesteps (uses time_idx and time_idx+1)
    """
    batch = Batch(
        surf_vars={
            "2t": torch.from_numpy(surf_ds["t2m"].values[time_idx:time_idx+2][None]),
            "10u": torch.from_numpy(surf_ds["u10"].values[time_idx:time_idx+2][None]),
            "10v": torch.from_numpy(surf_ds["v10"].values[time_idx:time_idx+2][None]),
            "msl": torch.from_numpy(surf_ds["msl"].values[time_idx:time_idx+2][None]),
        },
        static_vars={
            "z": torch.from_numpy(static_ds["z"].values[0]),
            "slt": torch.from_numpy(static_ds["slt"].values[0]),
            "lsm": torch.from_numpy(static_ds["lsm"].values[0]),
        },
        atmos_vars={
            "t": torch.from_numpy(atmos_ds["t"].values[time_idx:time_idx+2][None]),
            "u": torch.from_numpy(atmos_ds["u"].values[time_idx:time_idx+2][None]),
            "v": torch.from_numpy(atmos_ds["v"].values[time_idx:time_idx+2][None]),
            "q": torch.from_numpy(atmos_ds["q"].values[time_idx:time_idx+2][None]),
            "z": torch.from_numpy(atmos_ds["z"].values[time_idx:time_idx+2][None]),
        },
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values),
            lon=torch.from_numpy(surf_ds.longitude.values),
            time=(surf_ds.valid_time.values[time_idx+1].astype("datetime64[s]").tolist(),),
            atmos_levels=tuple(int(level) for level in atmos_ds.pressure_level.values),
        ),
    )
    
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


def run_forward_passes(model: Aurora, static_ds, surf_ds, atmos_ds, device: torch.device, n_steps: int = 2):
    """Run multiple forward passes and collect predictions."""
    preds = []
    
    print(f"\nRunning {n_steps} forward passes...")
    
    for step in range(n_steps):
        # Create batch starting at timestep (step)
        # step 0: uses timesteps 0,1 -> predicts timestep 2
        # step 1: uses timesteps 1,2 -> predicts timestep 3
        batch = create_batch(static_ds, surf_ds, atmos_ds, time_idx=step, device=device)
        
        with torch.no_grad():
            prediction = model(batch)
        
        prediction = prediction.to("cpu")
        preds.append(prediction)
        print(f"  Step {step+1}: Predicted {prediction.metadata.time[0]}")
    
    return preds


def visualize_simple(preds, surf_ds, output_dir: Path):
    """Visualize predictions using the simple Aurora demo logic.
    
    Uses fixed vmin/vmax and no geographic extent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_preds = len(preds)
    fig, ax = plt.subplots(n_preds, 2, figsize=(12, 6.5))
    
    for i in range(n_preds):
        pred = preds[i]
        
        # Aurora prediction (left column)
        ax[i, 0].imshow(pred.surf_vars["2t"][0, 0].numpy() - 273.15, vmin=-50, vmax=50, cmap="RdYlBu_r")
        ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
        if i == 0:
            ax[i, 0].set_title("Aurora Prediction")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        
        # ERA5 ground truth (right column)
        # Prediction at step i corresponds to ERA5 timestep (2 + i)
        # Because: step 0 predicts timestep 2, step 1 predicts timestep 3, etc.
        ax[i, 1].imshow(surf_ds["t2m"][2 + i].values - 273.15, vmin=-50, vmax=50, cmap="RdYlBu_r")
        if i == 0:
            ax[i, 1].set_title("ERA5")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
    
    plt.tight_layout()
    
    output_path = output_dir / "2_aurora_simple_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nSimple comparison saved to: {output_path}")


def visualize_with_error(preds, surf_ds, output_dir: Path):
    """Visualize predictions with error (difference) plot.
    
    Uses fixed vmin/vmax for temperature, symmetric for error.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_preds = len(preds)
    fig, ax = plt.subplots(n_preds, 3, figsize=(15, 6.5))
    
    for i in range(n_preds):
        pred = preds[i]
        
        pred_t2m = pred.surf_vars["2t"][0, 0].numpy() - 273.15
        era5_t2m_raw = surf_ds["t2m"][2 + i].values - 273.15
        
        # Handle grid mismatch by interpolating ERA5 to prediction grid
        if era5_t2m_raw.shape != pred_t2m.shape:
            lat_pred = pred.metadata.lat.numpy()
            lon_pred = pred.metadata.lon.numpy()
            lat_era5 = surf_ds.latitude.values
            lon_era5 = surf_ds.longitude.values
            
            # Create interpolator (ERA5 lat is descending 90 to -90)
            interp = RegularGridInterpolator(
                (lat_era5[::-1], lon_era5),
                era5_t2m_raw[::-1],
                bounds_error=False, 
                fill_value=np.nan
            )
            lon_grid, lat_grid = np.meshgrid(lon_pred, lat_pred)
            era5_t2m = interp((lat_grid, lon_grid))
        else:
            era5_t2m = era5_t2m_raw
        
        error = pred_t2m - era5_t2m
        
        # Aurora prediction (left column)
        ax[i, 0].imshow(pred_t2m, vmin=-50, vmax=50, cmap="RdYlBu_r")
        ax[i, 0].set_ylabel(str(pred.metadata.time[0]))
        if i == 0:
            ax[i, 0].set_title("Aurora Prediction")
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])
        
        # ERA5 ground truth (middle column)
        ax[i, 1].imshow(era5_t2m, vmin=-50, vmax=50, cmap="RdYlBu_r")
        if i == 0:
            ax[i, 1].set_title("ERA5")
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        
        # Error (right column)
        error_max = np.abs(error).max()
        im = ax[i, 2].imshow(error, vmin=-error_max, vmax=error_max, cmap="RdBu_r")
        if i == 0:
            ax[i, 2].set_title("Error (Pred - ERA5)")
        ax[i, 2].set_xticks([])
        ax[i, 2].set_yticks([])
        plt.colorbar(im, ax=ax[i, 2], label="°C")
    
    plt.tight_layout()
    
    output_path = output_dir / "2_aurora_simple_with_error.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Comparison with error saved to: {output_path}")


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    static_ds, surf_ds, atmos_ds = load_era5_data(DOWNLOAD_PATH)
    
    # Print available timesteps
    print(f"Available timesteps: {len(surf_ds.valid_time)}")
    for i, t in enumerate(surf_ds.valid_time.values):
        print(f"  [{i}] {t}")
    
    # Load model
    model = load_model(device)
    
    # Run forward passes (2 steps to match demo)
    preds = run_forward_passes(model, static_ds, surf_ds, atmos_ds, device, n_steps=2)
    
    # Visualize - simple version (matching demo logic)
    visualize_simple(preds, surf_ds, OUTPUT_DIR)
    
    # Visualize - with error plot
    visualize_with_error(preds, surf_ds, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
