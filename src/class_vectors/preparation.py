"""Checkpoint and classification-head preparation."""

import copy
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .architecture_adapters import build_image_classifier, vision_feature_fn
from .config import data_section
from .core import _default_batch_to_xy
from .model_loader import _build_encoder
from .optimization import make_optimizer
from .runtime import ModelRuntime, device_for, ensure_head, make_loader, vision_preprocess


def _new_path(value, field):
    if not value:
        raise ValueError("{} is required".format(field))
    path = Path(value).expanduser()
    if path.exists():
        raise FileExistsError("destination already exists: {}".format(path))
    return path


def _save_state(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)


def prepare_cnn(config, max_steps=None):
    """Fine-tune a CNN and save its initial and trained tensor state dictionaries."""
    if config.model.family != "vision" or not config.model.num_classes:
        raise ValueError("prepare-cnn requires family: vision and model.num_classes")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive")
    if data_section(config, "support")["split"] != "train":
        raise ValueError("CNN preparation requires a training support split")
    initial_path = _new_path(config.model.pretrained_checkpoint, "model.pretrained_checkpoint")
    trained_path = _new_path(config.model.finetuned_checkpoint, "model.finetuned_checkpoint")
    if initial_path.resolve() == trained_path.resolve():
        raise ValueError("initial and fine-tuned checkpoint destinations must differ")
    options = config.raw.get("preparation", {})
    epochs, rate = int(options.get("epochs", 5)), float(options.get("learning_rate", 1e-4))
    if epochs < 1 or rate <= 0:
        raise ValueError("preparation.epochs and learning_rate must be positive")
    torch.manual_seed(int(config.raw.get("seed", 0)))
    device = device_for(config)
    model = build_image_classifier(
        config.model.name, config.model.num_classes, pretrained=bool(config.model.pretrained)
    )
    initial = copy.deepcopy(model).cpu()
    runtime = ModelRuntime(model, model, vision_feature_fn, vision_preprocess())
    source_loader, _ = make_loader(config, runtime, "support")
    loader = DataLoader(
        source_loader.dataset,
        batch_size=int(options.get("batch_size", source_loader.batch_size)),
        shuffle=True,
        num_workers=source_loader.num_workers,
    )
    model.to(device).train()
    optimizer = make_optimizer(
        model.parameters(),
        options.get("optimizer", "adam"),
        rate,
        float(options.get("weight_decay", 0)),
    )
    steps, losses = 0, []
    for _ in range(epochs):
        for batch in loader:
            images, labels = _default_batch_to_xy(batch)
            loss = F.cross_entropy(model(images.to(device)), labels.long().to(device))
            if not torch.isfinite(loss):
                raise ValueError("CNN fine-tuning produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        if max_steps is not None and steps >= max_steps:
            break
    if not losses:
        raise ValueError("CNN training data contains no samples")
    _save_state(initial, initial_path)
    _save_state(model, trained_path)
    return {
        "initial_checkpoint": str(initial_path),
        "finetuned_checkpoint": str(trained_path),
        "optimizer_steps": steps,
        "final_training_loss": losses[-1],
    }


def prepare_head(config):
    """Build a CLIP classification head using the pretrained text encoder."""
    if config.model.family != "clip":
        raise ValueError("prepare-head requires model.family: clip")
    destination = _new_path(config.model.head_checkpoint, "model.head_checkpoint")
    device = device_for(config)
    model, _, preprocess = _build_encoder(
        config.model.name,
        config.model.pretrained,
        config.model.pretrained_checkpoint,
        config.model.cache_dir,
        device,
    )
    runtime = ModelRuntime(
        model, model, lambda encoder, images: encoder.encode_image(images), preprocess
    )
    _, class_names = make_loader(config, runtime, "support")
    head = ensure_head(config, runtime, class_names)
    _save_state(head, destination)
    return {"head_checkpoint": str(destination), "num_classes": len(class_names)}
