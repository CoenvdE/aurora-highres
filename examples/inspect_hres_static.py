#!/usr/bin/env python3
"""Download and inspect Aurora 0.1° HRES static variables from HuggingFace.

This script downloads the aurora-0.1-static.pickle file from the microsoft/aurora
HuggingFace repository and inspects its contents.

Static variables (0.1° resolution, 1801x3600 grid):
    - z: Surface geopotential (m²/s²)
    - slt: Soil type (categorical, 0-7)
    - lsm: Land-sea mask (0-1)

Usage:
    python examples/inspect_hres_static.py
    python examples/inspect_hres_static.py --plot  # Also generate plots
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

# HRES 0.1° static pickle from HuggingFace
STATIC_FILENAME = "aurora-0.1-static.pickle"


def download_static_pickle(cache_dir: Path | None = None) -> Path:
    """Download Aurora 0.1° static variables pickle from HuggingFace."""
    cache_str = str(cache_dir) if cache_dir else None

    print(f"Downloading {STATIC_FILENAME} from microsoft/aurora...")
    pickle_path = hf_hub_download(
        repo_id="microsoft/aurora",
        filename=STATIC_FILENAME,
        cache_dir=cache_str,
    )
    print(f"Downloaded to: {pickle_path}")
    return Path(pickle_path)


def inspect_static_vars(static_vars: dict[str, np.ndarray]) -> None:
    """Print detailed inspection of static variables."""
    print("\n" + "=" * 70)
    print("STATIC VARIABLES INSPECTION")
    print("=" * 70)

    print(f"\nNumber of variables: {len(static_vars)}")
    print(f"Variable names: {list(static_vars.keys())}")

    for name, arr in static_vars.items():
        print(f"\n{'─' * 50}")
        print(f"Variable: {name}")
        print(f"{'─' * 50}")
        print(f"  Shape: {arr.shape}")
        print(f"  Dtype: {arr.dtype}")
        print(f"  Memory: {arr.nbytes / 1024 / 1024:.2f} MB")
        print(f"  Min: {np.nanmin(arr):.6f}")
        print(f"  Max: {np.nanmax(arr):.6f}")
        print(f"  Mean: {np.nanmean(arr):.6f}")
        print(f"  Std: {np.nanstd(arr):.6f}")
        print(f"  NaN count: {np.isnan(arr).sum()}")
        # print(f"  Unique values: {len(np.unique(arr[~np.isnan(arr)]))}")

        # Show unique values for categorical data (like soil type)
        if name == "slt":
            unique_vals = np.unique(arr[~np.isnan(arr)])
            # print(f"  Unique soil types: {sorted(unique_vals)}")

        # Show percentiles
        if not np.all(np.isnan(arr)):
            percentiles = [0, 25, 50, 75, 100]
            pct_values = np.nanpercentile(arr, percentiles)
            print(
                f"  Percentiles (0/25/50/75/100): {[f'{v:.4f}' for v in pct_values]}")

    # Grid info (assuming 0.1° resolution)
    first_var = next(iter(static_vars.values()))
    nlat, nlon = first_var.shape
    expected_nlat = 1801  # 0.1° from 90 to -90
    expected_nlon = 3600  # 0.1° from 0 to 360

    print(f"\n{'─' * 50}")
    print("GRID INFORMATION")
    print(f"{'─' * 50}")
    print(f"  Actual grid size: {nlat} x {nlon}")
    print(f"  Expected 0.1° grid: {expected_nlat} x {expected_nlon}")
    print(
        f"  Grid match: {'✓' if (nlat == expected_nlat and nlon == expected_nlon) else '✗'}")

    if nlat == expected_nlat and nlon == expected_nlon:
        lat = np.linspace(90, -90, nlat)
        lon = np.linspace(0, 360, nlon, endpoint=False)
        print(f"  Latitude range: [{lat.min():.2f}°, {lat.max():.2f}°]")
        print(f"  Longitude range: [{lon.min():.2f}°, {lon.max():.2f}°]")
        print(f"  Resolution: 0.1° ({111 * 0.1:.1f} km at equator)")


def plot_static_vars(static_vars: dict[str, np.ndarray], output_dir: Path) -> None:
    """Generate statistical visualization plots for static variables."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in static_vars.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Aurora HRES Static Variable: {name}", fontsize=16, fontweight='bold')

        # Flatten array and remove NaNs for statistics
        data = arr.flatten()
        data_clean = data[~np.isnan(data)]

        # 1. Histogram
        ax = axes[0, 0]
        if name == "slt":
            # For categorical data, use bar plot
            unique_vals, counts = np.unique(data_clean, return_counts=True)
            ax.bar(unique_vals, counts, color='steelblue',
                   edgecolor='black', alpha=0.7)
            ax.set_xlabel("Soil Type Category")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Soil Types")
            ax.set_xticks(unique_vals)
        else:
            # For continuous data, use histogram
            ax.hist(data_clean, bins=100, color='steelblue',
                    edgecolor='black', alpha=0.7)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution (Histogram)")
            ax.grid(True, alpha=0.3)

        # 2. Box plot
        ax = axes[0, 1]
        bp = ax.boxplot(data_clean, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax.set_ylabel("Value")
        ax.set_title("Box Plot")
        ax.grid(True, alpha=0.3)

        # 3. Statistics table
        ax = axes[1, 0]
        ax.axis('off')
        stats = [
            ["Statistic", "Value"],
            ["Count", f"{len(data_clean):,}"],
            ["Mean", f"{np.mean(data_clean):.6f}"],
            ["Std Dev", f"{np.std(data_clean):.6f}"],
            ["Min", f"{np.min(data_clean):.6f}"],
            ["25%", f"{np.percentile(data_clean, 25):.6f}"],
            ["50% (Median)", f"{np.percentile(data_clean, 50):.6f}"],
            ["75%", f"{np.percentile(data_clean, 75):.6f}"],
            ["Max", f"{np.max(data_clean):.6f}"],
            ["NaN count", f"{np.isnan(data).sum():,}"],
            ["Unique values", f"{len(np.unique(data_clean)):,}"],
        ]
        table = ax.table(cellText=stats, cellLoc='left', loc='center',
                         colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header row
        for i in range(2):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        ax.set_title("Statistics Summary")

        # 4. Percentile plot
        ax = axes[1, 1]
        percentiles = np.linspace(0, 100, 101)
        values = np.percentile(data_clean, percentiles)
        ax.plot(percentiles, values, linewidth=2, color='steelblue')
        ax.fill_between(percentiles, values, alpha=0.3, color='steelblue')
        ax.set_xlabel("Percentile")
        ax.set_ylabel("Value")
        ax.set_title("Percentile Plot")
        ax.grid(True, alpha=0.3)

        # Add variable-specific info
        if name == "z":
            fig.text(0.5, 0.02, "Surface Geopotential [m²/s²]",
                     ha='center', fontsize=10, style='italic')
        elif name == "lsm":
            fig.text(0.5, 0.02, "Land-Sea Mask [0=sea, 1=land]",
                     ha='center', fontsize=10, style='italic')
        elif name == "slt":
            fig.text(0.5, 0.02, "Soil Type [0-7 categorical]",
                     ha='center', fontsize=10, style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        output_path = output_dir / f"static_{name}_stats.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and inspect Aurora 0.1° HRES static variables"
    )
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="HuggingFace cache directory")
    parser.add_argument("--plot", action="store_true",
                        help="Generate visualization plots")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("examples/downloads/static_plots"),
                        help="Directory for output plots")

    args = parser.parse_args()

    print("=" * 70)
    print("Aurora 0.1° HRES Static Variables Inspector")
    print("=" * 70)

    # Download the static pickle
    pickle_path = download_static_pickle(args.cache_dir)

    # Load the pickle
    print(f"\nLoading static variables from: {pickle_path}")
    with open(pickle_path, "rb") as f:
        static_vars = pickle.load(f)

    # Inspect the contents
    inspect_static_vars(static_vars)

    # Generate plots if requested
    if args.plot:
        print(f"\n{'─' * 50}")
        print("GENERATING PLOTS")
        print(f"{'─' * 50}")
        plot_static_vars(static_vars, args.output_dir)

    print("\n" + "=" * 70)
    print("✓ Inspection complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

# usage:
# python inspect_hres_static.py --plot
