
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime

def load_hres_t2(device, download_path="examples/downloads/hres"):
    download_path = Path(download_path)
    date = datetime(2023, 1, 1)
    ds = xr.open_dataset(download_path / date.strftime(f"surf_2t_%Y-%m-%d.grib"), engine="cfgrib")
    try:
        data = ds["t2m"].values[2] # Index 2 for 12:00
    except IndexError:
        data = ds["t2m"].values[-1]
    lat = torch.from_numpy(ds.latitude.values).to(device)
    lon = torch.from_numpy(ds.longitude.values).to(device)
    data = torch.from_numpy(data).to(device)
    return data, lat, lon

def load_era5_t2(device, download_path="examples/downloads/era5"):
    download_path = Path(download_path)
    ds = xr.open_dataset(download_path / "2023-01-01-surface-level.nc", engine="h5netcdf")
    data = ds["t2m"].values[2]
    lat = torch.from_numpy(ds.latitude.values).to(device)
    lon = torch.from_numpy(ds.longitude.values).to(device)
    data = torch.from_numpy(data).to(device)
    return data, lat, lon

def interpolate_and_compare(hres_data, hres_lat, hres_lon, era_data, era_lat, era_lon, region_bounds):
    """Interpolates ERA5 to HRES grid and compares."""
    
    # 1. Crop HRES to region first to define target grid
    lat_min, lat_max = region_bounds["lat"]
    lon_min, lon_max = region_bounds["lon"]
    
    mask_lat_hres = (hres_lat >= lat_min) & (hres_lat <= lat_max)
    mask_lon_hres = (hres_lon >= lon_min) & (hres_lon <= lon_max)
    
    hres_lat_idx = np.where(mask_lat_hres.cpu().numpy())[0]
    hres_lon_idx = np.where(mask_lon_hres.cpu().numpy())[0]
    
    target_hres_data = hres_data[hres_lat_idx[:, None], hres_lon_idx]
    target_lat = hres_lat[hres_lat_idx]
    target_lon = hres_lon[hres_lon_idx]
    
    # 2. Prepare ERA5 for interpolation
    from scipy.interpolate import RegularGridInterpolator
    
    era_lat_np = era_lat.cpu().numpy()
    era_lon_np = era_lon.cpu().numpy()
    era_data_np = era_data.cpu().numpy()
    
    # Handle potential descending latitude
    if era_lat_np[0] > era_lat_np[-1]:
        era_lat_np = era_lat_np[::-1]
        era_data_np = era_data_np[::-1, :]
        
    # Create meshgrid for target coordinates
    target_lat_grid, target_lon_grid = np.meshgrid(target_lat.cpu().numpy(), target_lon.cpu().numpy(), indexing='ij')

    # Define interpolation methods
    methods = ['linear', 'nearest', 'cubic']
    interpolated_results = {}
    
    for method in methods:
        # Note: 'cubic' requires scipy >= 1.9.0 for RegularGridInterpolator. 
        # If older, we might need RectBivariateSpline for cubic.
        # Assuming recent scipy.
        try:
            interpolator = RegularGridInterpolator((era_lat_np, era_lon_np), era_data_np, method=method, bounds_error=False, fill_value=None)
            interpolated_results[method] = interpolator((target_lat_grid, target_lon_grid))
        except ValueError as e:
            print(f"Warning: Method '{method}' failed (likely scipy version): {e}")
            # Fallback for cubic if needed, or just skip
            if method == 'cubic':
                 from scipy.interpolate import RectBivariateSpline
                 # RectBivariateSpline expects x, y, z. x is rows (lat), y is cols (lon)
                 rbs = RectBivariateSpline(era_lat_np, era_lon_np, era_data_np)
                 interpolated_results[method] = rbs(target_lat.cpu().numpy(), target_lon.cpu().numpy(), grid=True)

    
    # 4. Plot
    # Layout: 
    # Row 0: HRES (Ground Truth) - Spans all columns or just one? Let's do 3 rows.
    # Row 1: Interpolations (Linear, Nearest, Cubic)
    # Row 2: Differences (Linear, Nearest, Cubic)
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 18), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Use actual data bounds for extent
    extent = [target_lon.min().item(), target_lon.max().item(), target_lat.min().item(), target_lat.max().item()]
    
    vmin = target_hres_data.min().item()
    vmax = target_hres_data.max().item()
    
    # Row 0: Ground Truth (Center it or repeat? Let's put it in the middle)
    ax_gt = axes[0, 1]
    ax_gt.set_extent(extent, crs=ccrs.PlateCarree())
    ax_gt.add_feature(cfeature.COASTLINE)
    gl = ax_gt.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    im = ax_gt.pcolormesh(target_lon, target_lat, target_hres_data.cpu().numpy(), transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
    ax_gt.set_title("HRES (0.1°) - Ground Truth")
    plt.colorbar(im, ax=ax_gt, orientation='horizontal', pad=0.05)
    
    # Hide other axes in row 0
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    
    # Rows 1 & 2: Interpolations and Differences
    for i, method in enumerate(methods):
        if method not in interpolated_results:
            continue
            
        interp_data = interpolated_results[method]
        diff = interp_data - target_hres_data.cpu().numpy()
        
        # Interpolation Plot
        ax = axes[1, i]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        im = ax.pcolormesh(target_lon, target_lat, interp_data, transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title(f"ERA5 {method.capitalize()}")
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
        
        # Difference Plot
        ax = axes[2, i]
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.COASTLINE)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        
        diff_max = max(abs(diff.min()), abs(diff.max()))
        im = ax.pcolormesh(target_lon, target_lat, diff, transform=ccrs.PlateCarree(), cmap='bwr', vmin=-diff_max, vmax=diff_max)
        ax.set_title(f"Diff ({method.capitalize()} - HRES)")
        plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05).set_label("Difference (K)")
    
    plt.suptitle("Interpolation Methods Comparison at Timestep 2 (12:00)")
    plt.tight_layout()
    output_path = "examples/hres_vis/interpolation_comparison_t2.png"
    plt.savefig(output_path)
    print(f"Saved interpolation comparison to {output_path}")

def main():
    device = torch.device("cpu")
    
    region_bounds = {
        "lat": (30.0, 70.0),    
        "lon": (-30.0, 50.0),
    }
    
    print("Loading data...")
    hres_data, hres_lat, hres_lon = load_hres_t2(device)
    era_data, era_lat, era_lon = load_era5_t2(device)
    
    print("Interpolating and comparing...")
    interpolate_and_compare(hres_data, hres_lat, hres_lon, era_data, era_lat, era_lon, region_bounds)

if __name__ == "__main__":
    main()


# usage:
# python -m examples.interpolate_compare_t2
