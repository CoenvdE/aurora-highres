"""Quick script to inspect variable names in the ERA5 zarr file."""

import xarray as xr

# Path to the zarr file
zarr_path = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/"
    "1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"
)

print(f"Opening zarr file: {zarr_path}\n")

# Open zarr without loading data
ds = xr.open_zarr(zarr_path, consolidated=True)

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nDimensions: {dict(ds.dims)}")
print(f"\nCoordinates: {list(ds.coords)}")

print("\n" + "=" * 60)
print("DATA VARIABLES")
print("=" * 60)

for var_name in ds.data_vars:
    var = ds[var_name]
    print(f"\n{var_name}:")
    print(f"  Shape: {var.shape}")
    print(f"  Dims: {var.dims}")
    print(f"  Dtype: {var.dtype}")
    if 'long_name' in var.attrs:
        print(f"  Long name: {var.attrs['long_name']}")
    if 'units' in var.attrs:
        print(f"  Units: {var.attrs['units']}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total variables: {len(ds.data_vars)}")
print(f"Variable names: {list(ds.data_vars.keys())}")

ds.close()
