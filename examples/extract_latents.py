"""Utility to extract key latent representations from Aurora."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import aurora as aurora_module
import torch
import types
import warnings

from examples.init_exploring.load_era_batch_example import load_era_batch_example
from examples.init_exploring.utils import compute_patch_grid, load_model, format_latents_filename, register_latent_hooks


def main(save_dir: Path) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(device)

    print("Building batch...")
    batches = load_era_batch_example(device)  # TODO: replace with actual batches

    if not isinstance(batches, list):
        batches = [batches]

    print("Computing latent patch grid...")
    patch_grid = compute_patch_grid(
        batches[0].metadata.lat, batches[0].metadata.lon, model.patch_size)

    print("Registering latent hooks...")
    captures: dict[str, torch.Tensor] = {}
    handles, decoder_cleanup = register_latent_hooks(model, captures)

    print("Running inference...")
    try:
        for idx, batch in enumerate(batches):
            print(f"Processing batch {idx + 1} of {len(batches)}...")
            with torch.no_grad():
                preds = model(batch)  # NOTE: aurora returns unnormalised values
                preds = preds.to("cpu")
                timestamp = preds.metadata.time[0]
                target_file = save_dir / format_latents_filename(timestamp)

                print(f"Saving latents to {target_file}...")
                torch.save(
                    {
                        "captures": captures,
                        "patch_grid": patch_grid,
                        "prediction": preds,
                    },
                    target_file,
                )
    finally:
        for handle in handles:
            handle.remove()
        decoder_cleanup()

    print("Done.")


if __name__ == "__main__":
    main(save_dir=Path("examples/latents"))

# usage: python -m examples.extract_latents
