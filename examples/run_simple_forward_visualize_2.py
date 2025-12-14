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


def visualize_regional(preds, surf_ds, static_ds, output_dir: Path):
    """Visualize regional predictions vs ground truth for Europe.
    
    Shows only Prediction and Ground Truth (2 panels per timestep).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Europe bounds
    lat_min_eu, lat_max_eu = 30, 70
    lon_min_eu, lon_max_eu = -30, 50
    
    lat = surf_ds.latitude.values
    lon = surf_ds.longitude.values
    
    # Wrap longitude from 0-360 to -180-180
    lon_wrapped = ((lon + 180) % 360) - 180
    lon_order = np.argsort(lon_wrapped)
    lon_sorted = lon_wrapped[lon_order]
    
    # Find indices for region
    lat_mask = (lat >= lat_min_eu) & (lat <= lat_max_eu)
    lon_mask = (lon_sorted >= lon_min_eu) & (lon_sorted <= lon_max_eu)
    lat_idx = np.where(lat_mask)[0]
    lon_idx = np.where(lon_mask)[0]
    
    n_preds = len(preds)
    extent = [lon_min_eu, lon_max_eu, lat_min_eu, lat_max_eu]
    
    if HAS_CARTOPY:
        fig, ax = plt.subplots(
            n_preds, 2, 
            figsize=(14, 6 * n_preds),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
    else:
        fig, ax = plt.subplots(n_preds, 2, figsize=(14, 6 * n_preds))
    
    for i in range(n_preds):
        pred = preds[i]
        
        pred_t2m = pred.surf_vars["2t"][0, 0].numpy() - 273.15
        era5_t2m_raw = surf_ds["t2m"][2 + i].values - 273.15
        
        # Use 720 latitude points directly (no interpolation)
        if era5_t2m_raw.shape[0] == 721 and pred_t2m.shape[0] == 720:
            era5_t2m = era5_t2m_raw[:720, :]
        else:
            era5_t2m = era5_t2m_raw
        
        # Sort by longitude and extract Europe region
        pred_sorted = pred_t2m[:, lon_order]
        era5_sorted = era5_t2m[:, lon_order]
        
        pred_europe = pred_sorted[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        era5_europe = era5_sorted[lat_idx[0]:lat_idx[-1]+1, lon_idx[0]:lon_idx[-1]+1]
        
        # Use consistent global color scale for comparability
        vmin = -10  # °C
        vmax = 30   # °C
        
        data_list = [
            (pred_europe, "Prediction", 0),
            (era5_europe, "Ground Truth (ERA5)", 1)
        ]
        
        for data, title, col_idx in data_list:
            current_ax = ax[i, col_idx] if n_preds > 1 else ax[col_idx]
            
            if HAS_CARTOPY:
                current_ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3, zorder=2)
                current_ax.coastlines(resolution='50m', zorder=3)
                current_ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=3)
                im = current_ax.imshow(
                    data,
                    extent=extent,
                    transform=ccrs.PlateCarree(),
                    origin="upper",
                    vmin=vmin, vmax=vmax,
                    cmap="coolwarm"
                )
                current_ax.set_extent(extent, crs=ccrs.PlateCarree())
            else:
                im = current_ax.imshow(
                    data,
                    extent=extent,
                    origin="upper",
                    vmin=vmin, vmax=vmax,
                    cmap="coolwarm"
                )
                current_ax.set_xlabel("Longitude")
            
            if col_idx == 0:
                current_ax.set_ylabel(str(pred.metadata.time[0]))
            if i == 0:
                current_ax.set_title(title)
    
    plt.suptitle("Aurora Forward Pass: 2m Temperature (Europe)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    output_path = output_dir / "2_aurora_regional_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nRegional comparison saved to: {output_path}")


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
    
    # Visualize regional prediction vs ground truth
    visualize_regional(preds, surf_ds, static_ds, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
