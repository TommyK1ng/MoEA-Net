"""Vision encoder loading utilities for MoEA-Net."""

import json
from pathlib import Path


def default_vision_encoder_path():
    """Use the repository-local checkpoint folder when no path is provided."""
    return Path(__file__).resolve().parents[1] / "checkpoints"


def resolve_vision_encoder_files(vision_encoder_path=None):
    """Return local BiomedCLIP config and weights paths."""
    root = Path(vision_encoder_path) if vision_encoder_path else default_vision_encoder_path()
    if root.is_file():
        weights_path = root
        config_path = root.with_name("open_clip_config.json")
    else:
        config_path = root / "open_clip_config.json"
        weights_path = root / "open_clip_pytorch_model.bin"

    missing = [str(path) for path in (config_path, weights_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing BiomedCLIP vision encoder files: " + ", ".join(missing)
        )
    return config_path, weights_path


def load_biomedclip_visual_encoder(vision_encoder_path=None):
    """Load a frozen BiomedCLIP visual encoder from local files."""
    from open_clip import create_model_and_transforms
    from open_clip.factory import _MODEL_CONFIGS

    config_path, weights_path = resolve_vision_encoder_files(vision_encoder_path)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)

    # Register the local OpenCLIP config so loading works in offline environments.
    _MODEL_CONFIGS["biomedclip_local"] = config["model_cfg"]
    clip_model, _, _ = create_model_and_transforms(
        model_name="biomedclip_local",
        pretrained=str(weights_path),
        **{f"image_{key}": value for key, value in config["preprocess_cfg"].items()},
    )
    visual_encoder = clip_model.visual
    # MoEA-Net trains experts and LoRA adapters; BiomedCLIP stays fixed.
    visual_encoder.requires_grad_(False)
    visual_encoder.eval()
    return visual_encoder
