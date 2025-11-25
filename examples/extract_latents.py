"""Utility to extract key latent representations from Aurora."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import aurora as aurora_module
import torch
import types
import warnings

from examples.load_era_batch_example import load_era_batch_example
from examples.utils import compute_patch_grid, load_model, format_latents_filename

def register_latent_hooks(model: aurora_module.Aurora, captures: dict[str, torch.Tensor]):
    """Attach hooks so we capture the required latent snapshots."""
    handles = []

    def make_forward_hook(name: str):
        def hook(_, __, output):
            captures[name] = output.detach().cpu()

        return hook

    handles.append(model.encoder.register_forward_hook(
        make_forward_hook("after_encoder")))
    handles.append(model.backbone.register_forward_hook(
        make_forward_hook("after_processor")))

    def decoder_callback(latents: dict[str, torch.Tensor]):
        for key, tensor in latents.items():
            captures[f"decoder.{key}"] = tensor.detach().cpu()

    decoder_cleanup = register_decoder_latent_callback(
        model.decoder, decoder_callback)
    return handles, decoder_cleanup


def register_decoder_latent_callback(
    decoder: torch.nn.Module, callback: Callable[[dict[str, torch.Tensor]], None]
) -> Callable[[], None]:
    """Install `callback` so every emitted decoder latent is captured."""
    if hasattr(decoder, "register_latent_callback"):
        decoder.register_latent_callback(callback)
        return decoder.clear_latent_callback

    original_emit = getattr(decoder, "_emit_latents", None)
    if original_emit is None:
        warnings.warn(
            "Decoder does not expose `_emit_latents`, so decoder latents will not be captured."
        )
        return lambda: None

    def patched_emit(self, **latents: torch.Tensor):
        callback(latents)
        return original_emit(**latents)

    decoder._emit_latents = types.MethodType(patched_emit, decoder)

    def cleanup() -> None:
        decoder._emit_latents = original_emit

    return cleanup


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
