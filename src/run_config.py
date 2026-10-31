"""
Run Configuration Helpers
=========================
@Description:
Every training run writes a `config.json` next to its weights. Index building,
evaluation and the Streamlit app read it back, so the encoder is always rebuilt
with exactly the architecture/feature settings it was trained with.

(Before this, --pooling/--out_dim/--num_layers had to be re-typed by hand for
every index build, and a mismatch either crashed or silently produced garbage.)

Checkpoints trained before config.json existed fall back to LEGACY_DEFAULTS,
which match the original training script defaults.
"""

import json
import pathlib

import torch

CONFIG_FILENAME = "config.json"

# Architecture keys needed to rebuild the encoder
ARCH_KEYS = ("hidden_dim", "out_dim", "num_layers", "pooling", "features")

LEGACY_DEFAULTS = {
    "hidden_dim": 128,
    "out_dim": 64,
    "num_layers": 3,
    "pooling": "mean",
    "features": "raw",
}


def save_run_config(save_dir, config):
    path = pathlib.Path(save_dir) / CONFIG_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def load_run_config(weights_dir):
    """Returns (config_dict, found_flag). Missing keys are filled from LEGACY_DEFAULTS."""
    path = pathlib.Path(weights_dir) / CONFIG_FILENAME
    config = dict(LEGACY_DEFAULTS)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
        return config, True
    return config, False


def resolve_arch(args, weights_dir):
    """Fills architecture args from config.json. Values in config.json win over CLI
    defaults, because they describe what the checkpoint actually is. Mutates and returns args."""
    config, found = load_run_config(weights_dir)
    if found:
        for key in ARCH_KEYS:
            setattr(args, key, config[key])
    else:
        for key in ARCH_KEYS:
            if getattr(args, key, None) is None:
                setattr(args, key, LEGACY_DEFAULTS[key])
    return args, found


def load_encoder_weights(encoder, weights_dir, device):
    """Loads final_encoder_weights.pth, falling back to best_contrastive_model.pth.
    Returns the path that was loaded."""
    weights_dir = pathlib.Path(weights_dir)
    final_path = weights_dir / "final_encoder_weights.pth"
    best_path = weights_dir / "best_contrastive_model.pth"

    if final_path.exists():
        state = torch.load(final_path, map_location=device, weights_only=True)
        path = final_path
    elif best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        path = best_path
    else:
        raise FileNotFoundError(f"No trained weights found in {weights_dir}. Run training first!")

    if isinstance(state, dict) and "encoder_state_dict" in state:
        state = state["encoder_state_dict"]
    encoder.load_state_dict(state)
    return path
