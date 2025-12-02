#!/usr/bin/env python3
"""Inspect Aurora latent HDF5 files and print simple statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise Aurora latent HDF5 files in a compact, interpretable way."
    )
    parser.add_argument("latents_h5", type=Path, help="Path to the latents HDF5 file")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Limit the number of latent samples inspected per group. "
            "Useful when working with very large files."
        ),
    )
    return parser.parse_args()


class StatsAggregator:
    """Numerically stable summary stats across many tensors."""

    def __init__(self) -> None:
        self._count = 0
        self._total = 0.0
        self._total_sq = 0.0
        self._min: float | None = None
        self._max: float | None = None

    def update(self, array: np.ndarray) -> None:
        data = np.asarray(array, dtype=np.float64).ravel()
        if data.size == 0:
            return

        self._count += data.size
        self._total += float(data.sum())
        self._total_sq += float(np.square(data).sum())
        current_min = float(data.min())
        current_max = float(data.max())
        self._min = current_min if self._min is None else min(self._min, current_min)
        self._max = current_max if self._max is None else max(self._max, current_max)

    def as_dict(self) -> dict[str, float]:
        if self._count == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "std": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            }
        mean = self._total / self._count
        variance = max(self._total_sq / self._count - mean ** 2, 0.0)
        return {
            "count": float(self._count),
            "mean": mean,
            "std": variance ** 0.5,
            "min": self._min if self._min is not None else float("nan"),
            "max": self._max if self._max is not None else float("nan"),
        }


def nice_list(values: Iterable[str], limit: int = 3) -> str:
    values = list(values)
    if len(values) <= limit:
        return ", ".join(values)
    head = ", ".join(values[:limit])
    tail = ", ".join(values[-limit:])
    return f"{head}, ..., {tail}" if head != tail else head


def format_bytes(size: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} {units[-1]}"


def describe_patch_grid(group: h5py.Group) -> None:
    print("patch_grid:")
    for key, dataset in group.items():
        print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
    for attr in ("patch_shape", "patch_size", "patch_count"):
        if attr in group.attrs:
            print(f"  {attr}: {group.attrs[attr]}")


def summarise_latent_group(group_name: str, group: h5py.Group, max_samples: int | None) -> None:
    dataset_names = list(group.keys())
    print(f"{group_name}:")
    print(f"  datasets: {len(dataset_names)}")
    if not dataset_names:
        return

    total_storage = 0.0
    for name in dataset_names:
        dataset = group[name]
        try:
            total_storage += float(dataset.id.get_storage_size())
        except (AttributeError, ValueError):
            total_storage += float(dataset.size * dataset.dtype.itemsize)
    per_dataset = total_storage / len(dataset_names)
    print(f"  storage: total={format_bytes(total_storage)} per-sample={format_bytes(per_dataset)}")

    sample_names = dataset_names
    if max_samples is not None:
        sample_names = dataset_names[:max_samples]
        print(f"  inspecting: {len(sample_names)} (limit {max_samples})")
    else:
        print(f"  inspecting: {len(sample_names)} (all)")

    aggregator = StatsAggregator()
    shapes: set[tuple[int, ...]] = set()
    dtypes: set[str] = set()

    for name in sample_names:
        dataset = group[name]
        array = dataset[()]
        shapes.add(array.shape)
        dtypes.add(str(array.dtype))
        aggregator.update(array)

    stats = aggregator.as_dict()
    shape_text = nice_list([str(shape) for shape in sorted(shapes)])
    dtype_text = nice_list(sorted(dtypes))
    print(f"  shapes: {shape_text}")
    print(f"  dtypes: {dtype_text}")
    print(
        "  values: count={count:.0f} mean={mean:.5g} std={std:.5g} min={min:.5g} max={max:.5g}".format(
            **stats
        )
    )
    first_samples = nice_list(sample_names)
    print(f"  labels: {first_samples}")


def inspect_file(latents_h5: Path, max_samples: int | None) -> None:
    if not latents_h5.exists():
        raise SystemExit(f"File not found: {latents_h5}")

    print(f"Inspecting: {latents_h5}")
    try:
        file_size = latents_h5.stat().st_size
        print(f"file size: {format_bytes(file_size)}")
    except OSError:
        pass
    with h5py.File(latents_h5, "r") as handle:
        if "patch_grid" in handle:
            describe_patch_grid(handle["patch_grid"])
        else:
            print("patch_grid: <missing>")

        for group_name in ("pressure_latents", "surface_latents"):
            if group_name in handle:
                summarise_latent_group(group_name, handle[group_name], max_samples)
            else:
                print(f"{group_name}: <missing>")


def main() -> None:
    args = parse_args()
    inspect_file(args.latents_h5.expanduser(), args.max_samples)


if __name__ == "__main__":
    main()
