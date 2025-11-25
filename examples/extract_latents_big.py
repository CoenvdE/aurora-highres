
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
import pandas as pd

import aurora as aurora_module
import torch
import types
import warnings

from examples.load_era_batch_snellius import load_batch_from_zarr
from examples.utils import compute_patch_grid, load_model, format_latents_filename

# Default paths from the user's environment/examples
DEFAULT_ZARR_PATH = "/projects/2/managed_datasets/ERA5/era5-gcp-zarr/ar/1959-2022-wb13-6h-0p25deg-chunk-1.zarr-v2"

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


def main(
    start_year: int,
    end_year: int,
    save_dir: Path,
    zarr_path: str,
    static_path: str,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_model(device)
    model.eval()

    print("Registering latent hooks...")
    captures: dict[str, torch.Tensor] = {}
    handles, decoder_cleanup = register_latent_hooks(model, captures)

    try:
        #TODO: check every 6 hours instead of what it does now
        # Iterate through years
        for year in range(start_year, end_year + 1):
            print(f"Processing year {year}...")
            year_dir = save_dir / str(year)
            year_dir.mkdir(exist_ok=True)

            # Define date range for the year
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            # Iterate through days - assuming 6h intervals as per Zarr filename
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                print(f"  Processing {date_str}...")

                try:
                    # Load batch
                    batch = load_batch_from_zarr(zarr_path, static_path, date_str)
                    batch = batch.to(device)
                    
                    with torch.no_grad():
                        preds = model(batch)
                        preds = preds.to("cpu")
                        
                        # Timestamp from prediction might be different if the loader sets it.
                        # Using the one from the batch metadata.
                        timestamp = preds.metadata.time[0]
                        
                        # Format filename
                        filename = format_latents_filename(timestamp)
                        target_file = year_dir / filename

                        if target_file.exists():
                             print(f"    Skipping {target_file}, already exists.")
                        else:
                            print(f"    Saving latents to {target_file}...")
                            torch.save(
                                {
                                    "captures": captures,
                                    "patch_grid": compute_patch_grid(batch.metadata.lat, batch.metadata.lon, model.patch_size),
                                    "prediction": preds,
                                },
                                target_file,
                            )
                        
                        # Clear captures for next iteration
                        captures.clear()

                except Exception as e:
                    print(f"  Failed to process {date_str}: {e}")
                
                current_date += timedelta(days=1)

    finally:
        for handle in handles:
            handle.remove()
        decoder_cleanup()

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_year", type=int, default=2020, help="Start year")
    parser.add_argument("--end_year", type=int, default=2020, help="End year")
    parser.add_argument("--save_dir", type=Path, default=Path("examples/latents"), help="Directory to save latents")
    parser.add_argument("--zarr_path", type=str, default=DEFAULT_ZARR_PATH, help="Path to ERA5 Zarr")
    parser.add_argument("--static_path", type=str, required=True, help="Path to static data NetCDF")
    
    args = parser.parse_args()

    main(
        start_year=args.start_year,
        end_year=args.end_year,
        save_dir=args.save_dir,
        zarr_path=args.zarr_path,
        static_path=args.static_path,
    )
