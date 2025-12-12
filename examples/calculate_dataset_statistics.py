#!/usr/bin/env python3
"""
Calculate mean and std per variable from zarr datasets and save to JSON.
"""

import argparse
import json
import zarr
import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_statistics(zarr_path, output_path=None, chunk_size=100):
    """
    Calculate mean and std for each variable in a zarr dataset.
    
    Args:
        zarr_path: Path to the zarr dataset
        output_path: Path to save the statistics JSON (default: same directory as zarr)
        chunk_size: Number of timesteps to process at once for memory efficiency
    """
    print(f"\nProcessing: {zarr_path}")
    
    # Open zarr dataset
    ds = zarr.open(zarr_path, mode='r')
    
    # Get all variable names (exclude coordinates and metadata)
    coord_names = {
        'time', 'lat', 'lon', 'latitude', 'longitude', 'level',
        'valid_time', 'step', 'number', 'surface', 'isobaricInhPa',
        'heightAboveGround', 'date', 'timestamp'
    }
    
    # Filter variables: only keep arrays with 3+ dimensions (time, lat, lon or time, level, lat, lon)
    var_names = []
    for key in ds.keys():
        if key in coord_names:
            continue
        var_array = ds[key]
        # Process variables with at least 2 dimensions (supports both static 2D and temporal 3D+ data)
        if hasattr(var_array, 'shape') and len(var_array.shape) >= 2:
            var_names.append(key)
    
    print(f"Found {len(var_names)} data variables: {var_names}")
    
    # Show excluded variables for reference
    excluded = [key for key in ds.keys() if key not in var_names]
    if excluded:
        print(f"Excluded metadata/coordinates: {excluded}")
    
    statistics = {}
    
    for var_name in var_names:
        print(f"\nCalculating statistics for {var_name}...")
        var_data = ds[var_name]
        
        # Get shape info
        shape = var_data.shape
        print(f"  Shape: {shape}")
        
        # Handle different dimensions
        # Typical shapes: (time, lat, lon) or (time, level, lat, lon)
        if len(shape) == 3:
            # (time, lat, lon)
            time_dim = 0
        elif len(shape) == 4:
            # (time, level, lat, lon)
            time_dim = 0
        else:
            print(f"  Warning: Unexpected shape {shape}, processing all data at once")
            time_dim = None
        
        # Calculate statistics incrementally to avoid memory issues
        if time_dim is not None and shape[time_dim] > chunk_size:
            print(f"  Using incremental calculation (chunk_size={chunk_size})")
            
            n_samples = 0
            running_sum = 0.0
            running_sq_sum = 0.0
            
            n_chunks = (shape[time_dim] + chunk_size - 1) // chunk_size
            
            for i in tqdm(range(0, shape[time_dim], chunk_size), desc=f"  {var_name}"):
                end_idx = min(i + chunk_size, shape[time_dim])
                
                if time_dim == 0:
                    chunk = var_data[i:end_idx]
                else:
                    chunk = var_data[:, i:end_idx]
                
                # Convert to numpy array
                chunk = np.array(chunk)
                
                # Mask NaN values
                valid_mask = ~np.isnan(chunk)
                valid_data = chunk[valid_mask]
                
                if len(valid_data) > 0:
                    n_samples += len(valid_data)
                    running_sum += np.sum(valid_data)
                    running_sq_sum += np.sum(valid_data ** 2)
            
            if n_samples > 0:
                mean = running_sum / n_samples
                variance = (running_sq_sum / n_samples) - (mean ** 2)
                std = np.sqrt(max(variance, 0))  # Ensure non-negative
            else:
                mean = np.nan
                std = np.nan
                
        else:
            # Load all data at once
            print(f"  Loading all data")
            data = np.array(var_data[:])
            
            # Mask NaN values
            valid_mask = ~np.isnan(data)
            valid_data = data[valid_mask]
            
            if len(valid_data) > 0:
                mean = float(np.mean(valid_data))
                std = float(np.std(valid_data))
            else:
                mean = np.nan
                std = np.nan
        
        statistics[var_name] = {
            'mean': float(mean),
            'std': float(std),
            'shape': shape,
            'n_dims': len(shape)
        }
        
        print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
    
    # Determine output path
    if output_path is None:
        zarr_dir = Path(zarr_path).parent
        zarr_name = Path(zarr_path).stem
        output_path = zarr_dir / f"{zarr_name}_statistics.json"
    else:
        output_path = Path(output_path)
    
    # Save statistics
    print(f"\nSaving statistics to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"\n✓ Statistics saved successfully!")
    
    return statistics


def main():
    parser = argparse.ArgumentParser(
        description='Calculate mean and std for variables in zarr datasets'
    )
    parser.add_argument(
        '--zarr-path',
        type=str,
        required=True,
        help='Path to the zarr dataset'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for statistics JSON (default: same directory as zarr with _statistics.json suffix)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of timesteps to process at once (default: 100)'
    )
    
    args = parser.parse_args()
    
    calculate_statistics(
        zarr_path=args.zarr_path,
        output_path=args.output,
        chunk_size=args.chunk_size
    )


if __name__ == '__main__':
    main()

