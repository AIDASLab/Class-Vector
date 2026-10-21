"""Model and checkpoint loading."""

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class EncoderPair:
    pretrained: torch.nn.Module
    finetuned: torch.nn.Module
    train_preprocess: object
    val_preprocess: object


def _state_dict_from_checkpoint(path, device):
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("checkpoint must contain a non-empty state_dict: {}".format(path))
    if not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError("checkpoint contains non-tensor entries; provide a model state_dict")
    keys = list(checkpoint)
    if all(key.startswith("module.") for key in keys):
        checkpoint = {key[len("module.") :]: value for key, value in checkpoint.items()}
    if all(key.startswith("model.") for key in checkpoint):
        checkpoint = {key[len("model.") :]: value for key, value in checkpoint.items()}
    return checkpoint


def _build_encoder(name, pretrained, checkpoint_path, cache_dir, device):
    import open_clip

    initialization = None if checkpoint_path else pretrained
    pretrained_config = (open_clip.get_pretrained_cfg(name, pretrained) or {}) if pretrained else {}
    model, train_preprocess, val_preprocess = open_clip.create_model_and_transforms(
        model_name=name,
        pretrained=initialization,
        cache_dir=cache_dir,
        force_quick_gelu=bool(pretrained_config.get("quick_gelu", False)),
    )
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError("model checkpoint does not exist: {}".format(path))
        state = _state_dict_from_checkpoint(path, "cpu")
        if any(key.startswith("visual.") for key in state) and not any(
            key.startswith("transformer.") for key in state
        ):
            visual_state = {
                key[len("visual.") :]: value
                for key, value in state.items()
                if key.startswith("visual.")
            }
            model.visual.load_state_dict(visual_state, strict=True)
            model._class_vector_text_available = False
        else:
            model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, train_preprocess, val_preprocess


def load_encoder_pair(config) -> EncoderPair:
    """Load pretrained and fine-tuned encoders with one architecture setting."""
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained, train_tf, val_tf = _build_encoder(
        config.name,
        config.pretrained,
        config.pretrained_checkpoint,
        config.cache_dir,
        device,
    )
    if not config.finetuned_checkpoint and not config.finetuned_pretrained:
        raise ValueError("set model.finetuned_checkpoint or model.finetuned_pretrained")
    finetuned, _, _ = _build_encoder(
        config.name,
        config.finetuned_pretrained or (config.pretrained if config.finetuned_checkpoint else None),
        config.finetuned_checkpoint,
        config.cache_dir,
        device,
    )
    return EncoderPair(pretrained, finetuned, train_tf, val_tf)
