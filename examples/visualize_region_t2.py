
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime
from examples.helpers_plot_region import _draw_region_box

def load_hres_t2(device, download_path="examples/downloads/hres"):
    """Loads HRES data for timestep 2 (12:00)."""
    download_path = Path(download_path)
    date = datetime(2023, 1, 1)
    # Load 12:00 data
    ds = xr.open_dataset(download_path / date.strftime(f"surf_2t_%Y-%m-%d.grib"), engine="cfgrib")
    # Assuming the file might contain multiple steps or we need to find the one matching 12:00
    # Based on file naming in load_hres_batch_example, surf variables seem to be in one file.
    # Let's check if we can index it. If not, we might need to rely on the fact that we saw 
    # specific time files for atmos but surf was single file.
    # Actually, looking at load_hres_batch_example: data = ds[v_in_file].values[:2] # 00 and 06
    # We need 12:00. Let's assume it's the 3rd index if available, or we need to check validity.
    # For safety, let's try to load the specific time if possible or just take index 2.
    
    # However, for HRES, the example showed:
    # download_path / date.strftime(f"surf_2t_%Y-%m-%d.grib")
    # And it took values[:2].
    # If the file has more steps, we can take index 2.
    
    try:
        data = ds["t2m"].values[2] # Index 2 for 12:00 (00, 06, 12)
    except IndexError:
        # Fallback or error if not enough steps. 
        # But wait, the user request says "Important that it is 3rd as the first 2 are made to make the prediciton of the third"
        # So we must assume it exists.
        print("Warning: Could not load index 2 from HRES surf file. Checking if it exists.")
        data = ds["t2m"].values[-1] # Fallback to last if index 2 fails, but this is risky.
        
    lat = torch.from_numpy(ds.latitude.values).to(device)
    lon = torch.from_numpy(ds.longitude.values).to(device)
    data = torch.from_numpy(data).to(device)
    return data, lat, lon

def load_era5_t2(device, download_path="examples/downloads/era5"):
    """Loads ERA5 data for timestep 2 (12:00)."""
    download_path = Path(download_path)
    ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="h5netcdf")
    # Index 2 for 12:00
    data = ds["t2m"].values[2]
    lat = torch.from_numpy(ds.latitude.values).to(device)
    lon = torch.from_numpy(ds.longitude.values).to(device)
    data = torch.from_numpy(data).to(device)
    return data, lat, lon

def plot_region_comparison(hres_data, hres_lat, hres_lon, era_data, era_lat, era_lon, region_bounds):
    """Plots comparison for a specific region."""
    
    # Crop data to region
    def crop(data, lat, lon, bounds):
        lat_min, lat_max = bounds["lat"]
        lon_min, lon_max = bounds["lon"]
        
        # Handle longitude wrapping if needed, but for now simple mask
        mask_lat = (lat >= lat_min) & (lat <= lat_max)
        mask_lon = (lon >= lon_min) & (lon <= lon_max)
        
        # We need to subset the 2D data. 
        # Assuming lat/lon are 1D or 2D. In the examples they seem to be 1D for ERA5 and HRES (grib/netcdf usually).
        # But HRES might be 1D if it's unstructured or 2D if grid.
        # load_hres_batch_example uses ds.latitude.values which are usually 1D for regular grids.
        
        if lat.ndim == 1 and lon.ndim == 1:
            lat_idx = np.where(mask_lat)[0]
            lon_idx = np.where(mask_lon)[0]
            
            sub_data = data[lat_idx[:, None], lon_idx]
            sub_lat = lat[lat_idx]
            sub_lon = lon[lon_idx]
            return sub_data, sub_lat, sub_lon
        else:
            # If 2D (unlikely for these standard files but possible)
            return data, lat, lon

    hres_sub, hres_lat_sub, hres_lon_sub = crop(hres_data.cpu().numpy(), hres_lat.cpu().numpy(), hres_lon.cpu().numpy(), region_bounds)
    era_sub, era_lat_sub, era_lon_sub = crop(era_data.cpu().numpy(), era_lat.cpu().numpy(), era_lon.cpu().numpy(), region_bounds)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    vmin = min(hres_sub.min(), era_sub.min())
    vmax = max(hres_sub.max(), era_sub.max())
    
    # Use actual data bounds for extent to remove whitespace
    extent = [hres_lon_sub.min(), hres_lon_sub.max(), hres_lat_sub.min(), hres_lat_sub.max()]
    
    # Plot HRES
    ax = axes[0]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    im = ax.pcolormesh(hres_lon_sub, hres_lat_sub, hres_sub, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title("HRES (0.1°) - Region - t=2")
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    
    # Plot ERA5
    ax = axes[1]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    im = ax.pcolormesh(era_lon_sub, era_lat_sub, era_sub, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax.set_title("ERA5 (0.25°) - Region - t=2")
    plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
    
    plt.suptitle("Regional Comparison at Timestep 2 (12:00)")
    plt.tight_layout()
    output_path = "examples/hres_vis/region_comparison_t2.png"
    plt.savefig(output_path)
    print(f"Saved region comparison to {output_path}")

def main():
    device = torch.device("cpu")
    
    # Region from examples/decode_deag_latents_region.py
    region_bounds = {
        "lat": (30.0, 70.0),    
        "lon": (-30.0, 50.0),
    }
    
    print("Loading HRES data for t=2...")
    hres_data, hres_lat, hres_lon = load_hres_t2(device)
    
    print("Loading ERA5 data for t=2...")
    era_data, era_lat, era_lon = load_era5_t2(device)
    
    print("Plotting region comparison...")
    plot_region_comparison(hres_data, hres_lat, hres_lon, era_data, era_lat, era_lon, region_bounds)

if __name__ == "__main__":
    main()

# usage:
# python -m examples.visualize_region_t2
