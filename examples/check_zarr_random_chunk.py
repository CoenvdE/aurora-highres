"""Utility script to sanity-check ERA5 Zarr access by sampling a random chunk."""

import argparse
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_ZARR_PATH = (
    "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-"
    "chunk-1.zarr-v2"
)

COORD_CANDIDATES = {
    "lat": ("latitude", "lat"),
    "lon": ("longitude", "lon"),
    "time": ("time", "valid_time"),
}


def _format_range(values: np.ndarray) -> Tuple[str, str]:
    """Return a printable (min, max) tuple for numeric or datetime values."""
    if np.issubdtype(values.dtype, np.datetime64):
        start = pd.to_datetime(values.min()).to_pydatetime()
        end = pd.to_datetime(values.max()).to_pydatetime()
        return start.isoformat(), end.isoformat()

    start = float(values.min())
    end = float(values.max())
    return f"{start:.3f}", f"{end:.3f}"


def summarize_dataset(ds: xr.Dataset) -> None:
    print("=== Dataset Summary ===")
    print("Dimensions and approx chunk sizes:")
    for dim, size in ds.dims.items():
        chunks = ds.chunksizes.get(dim)
        if chunks:
            unique = sorted(set(int(c) for c in chunks))
            chunk_str = f" chunks~{unique[:3]}"
            if len(unique) > 3:
                chunk_str = f"{chunk_str}..."
        else:
            chunk_str = ""
        print(f"  - {dim}: {size}{chunk_str}")

    for key, candidates in COORD_CANDIDATES.items():
        coord_name = next(
            (name for name in candidates if name in ds.coords), None)
        if coord_name is None:
            continue
        coord_values = ds[coord_name].values
        start, end = _format_range(coord_values)
        print(f"  - {coord_name} range: {start} -> {end}")

    print("Variables:")
    for var in ds.data_vars:
        print(f"  - {var}")


def _random_chunk_bounds(size: int, chunks: Tuple[int, ...], rng: random.Random) -> Tuple[int, int]:
    if chunks:
        idx = rng.randrange(len(chunks))
        start = sum(chunks[:idx])
        stop = start + chunks[idx]
        return start, min(stop, size)

    width = min(size, 16)
    if width == size:
        return 0, size
    start = rng.randrange(0, size - width)
    return start, start + width


def select_random_chunk(ds: xr.Dataset, rng: random.Random) -> Dict[str, slice]:
    selections: Dict[str, slice] = {}
    for dim, size in ds.dims.items():
        chunks = tuple(int(c) for c in (ds.chunksizes.get(dim) or ()))
        start, stop = _random_chunk_bounds(size, chunks, rng)
        selections[dim] = slice(start, stop)
    return selections


def main(zarr_path: str, variable: str | None, seed: int | None, consolidated: bool) -> None:
    rng = random.Random(seed)
    ds = xr.open_zarr(zarr_path, consolidated=consolidated)

    summarize_dataset(ds)

    slices = select_random_chunk(ds, rng)
    print("\n=== Random Chunk Selection ===")
    for dim, sel in slices.items():
        print(f"  - {dim}: {sel.start}:{sel.stop} (size={sel.stop - sel.start})")

    chunk = ds.isel(**slices).load()

    if variable is None:
        if not chunk.data_vars:
            raise ValueError("Dataset contains no variables to inspect.")
        variable = rng.choice(list(chunk.data_vars))

    if variable not in chunk.data_vars:
        raise ValueError(f"Variable '{variable}' not found in dataset.")

    print(f"\nVariable '{variable}' values for the selected chunk:")
    values = chunk[variable].values
    print(np.array2string(values, threshold=20, edgeitems=2, floatmode="unique"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a random chunk from a Zarr store.")
    parser.add_argument("--zarr_path", type=str,
                        default=DEFAULT_ZARR_PATH, help="Path to ERA5 Zarr store")
    parser.add_argument("--variable", type=str, default=None,
                        help="Specific variable to print")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument(
        "--no_consolidated",
        action="store_true",
        help="Disable consolidated metadata when opening the store",
    )
    args = parser.parse_args()

    main(
        zarr_path=args.zarr_path,
        variable=args.variable,
        seed=args.seed,
        consolidated=not args.no_consolidated,
    )

# usage:
# python examples/check_zarr_random_chunk.py