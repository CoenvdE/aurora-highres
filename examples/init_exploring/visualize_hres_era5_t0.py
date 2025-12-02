
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import numpy as np
from examples.init_exploring.load_era_batch_example import load_era_batch_example
from examples.init_exploring.load_hres_batch_example import load_hres_batch_example

def plot_comparison(hres_batch, era_batch, variable="2t", time_idx=0):
    """Plots HRES and ERA5 data side-by-side for a given variable and time index."""
    
    # Extract data
    hres_data = hres_batch.surf_vars[variable][0, time_idx].cpu().numpy()
    era_data = era_batch.surf_vars[variable][0, time_idx].cpu().numpy()
    
    hres_lat = hres_batch.metadata.lat.cpu().numpy()
    hres_lon = hres_batch.metadata.lon.cpu().numpy()
    era_lat = era_batch.metadata.lat.cpu().numpy()
    era_lon = era_batch.metadata.lon.cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Determine common color limits
    vmin = min(hres_data.min(), era_data.min())
    vmax = max(hres_data.max(), era_data.max())
    
    # Plot HRES
    ax = axes[0]
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    im = ax.pcolormesh(hres_lon, hres_lat, hres_data, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title(f"HRES (0.1°) - {variable} - t={time_idx}")
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

    # Plot ERA5
    ax = axes[1]
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    im = ax.pcolormesh(era_lon, era_lat, era_data, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title(f"ERA5 (0.25°) - {variable} - t={time_idx}")
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)

    plt.suptitle(f"Comparison of {variable} at Timestep {time_idx}")
    plt.tight_layout()
    output_path = f"examples/hres_vis/comparison_t{time_idx}_{variable}.png"
    plt.savefig(output_path)
    print(f"Saved comparison plot to {output_path}")

def main():
    device = torch.device("cpu") # Use CPU for visualization script to avoid memory issues if GPU is busy
    print("Loading HRES batch...")
    hres_batch = load_hres_batch_example(device)
    
    print("Loading ERA5 batch...")
    era_batch = load_era_batch_example(device)
    
    print("Plotting comparison...")
    plot_comparison(hres_batch, era_batch, variable="2t", time_idx=0)

if __name__ == "__main__":
    main()

# usage:
# python -m examples.visualize_hres_era5_t0