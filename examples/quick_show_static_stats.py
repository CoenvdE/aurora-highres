#!/usr/bin/env python3
"""Quick script to show mean and std of static variables."""

import zarr
import numpy as np
import sys

zarr_path = sys.argv[1] if len(sys.argv) > 1 else "/projects/prjs1858/static_hres.zarr"

print(f"Opening: {zarr_path}")
ds = zarr.open(zarr_path, mode='r')

print("\nStatic variable statistics:")
print("-" * 50)

for var_name in ds.keys():
    if var_name in ['lat', 'lon', 'latitude', 'longitude', 'time']:
        continue
    
    data = np.array(ds[var_name][:])
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) > 0:
        mean = np.mean(valid_data)
        std = np.std(valid_data)
        print(f"{var_name:10s}: mean={mean:12.4f}, std={std:12.4f}")
    else:
        print(f"{var_name:10s}: NO VALID DATA")

print("-" * 50)

