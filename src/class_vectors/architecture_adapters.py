"""CNN and BERT feature extractors and classification heads."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn


def _safe_state_dict(path):
    state = torch.load(str(Path(path).expanduser()), map_location="cpu", weights_only=True)
    if isinstance(state, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if (
        not isinstance(state, dict)
        or not state
        or not all(torch.is_tensor(v) for v in state.values())
    ):
        raise ValueError("checkpoint must be a tensor-only state_dict")
    keys = list(state)
    if all(key.startswith("module.") for key in keys):
        state = {key[len("module.") :]: value for key, value in state.items()}
    return state


class VisionFeatureExtractor(nn.Module):
    """ResNet18/50 or ConvNeXt-Tiny feature trunk."""

    def __init__(self, architecture: str, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models

        architecture = architecture.lower().replace("-", "").replace("_", "")
        if architecture == "resnet18":
            model = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = model.fc.in_features
            self._kind = "resnet"
        elif architecture == "resnet50":
            model = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.feature_dim = model.fc.in_features
            self._kind = "resnet"
        elif architecture in ("convnext", "convnexttiny"):
            model = models.convnext_tiny(
                weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            )
            self.features = model.features
            self.pooler = model.avgpool
            self.feature_dim = model.classifier[-1].in_features
            self._kind = "convnext"
        else:
            raise ValueError("architecture must be resnet18, resnet50, or convnext_tiny")
        self.architecture = architecture

    def forward(self, images):
        features = self.features(images)
        if self._kind == "convnext":
            features = self.pooler(features)
        return features.flatten(start_dim=1)


class ImageClassifier(nn.Module):
    """Feature trunk with a linear classification head."""

    def __init__(self, feature_extractor: VisionFeatureExtractor, num_classes: int):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.feature_extractor = feature_extractor
        self.classifier = nn.Linear(feature_extractor.feature_dim, num_classes)

    def get_features(self, images):
        return self.feature_extractor(images)

    def forward(self, images):
        return self.classifier(self.get_features(images))


def build_image_classifier(
    architecture: str,
    num_classes: int,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    strict: bool = True,
):
    """Build a CNN classifier and load its state dictionary."""
    model = ImageClassifier(
        VisionFeatureExtractor(architecture, pretrained=pretrained), num_classes
    )
    if checkpoint_path:
        model.load_state_dict(_safe_state_dict(checkpoint_path), strict=strict)
    return model


def vision_feature_fn(model, images):
    """Feature function accepted by paired extraction and latent steering."""
    if not hasattr(model, "get_features"):
        raise TypeError("vision model must expose get_features(images)")
    return model.get_features(images)


@dataclass
class BertClassifierPair:
    pretrained_encoder: nn.Module
    finetuned_classifier: nn.Module
    tokenizer: object


def load_bert_classifier_pair(
    pretrained_model: str = "bert-base-uncased",
    finetuned_model: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load the base BERT encoder, task classifier, and tokenizer by explicit IDs."""
    if not finetuned_model:
        raise ValueError("finetuned_model must identify a task-finetuned BERT checkpoint")
    from transformers import (
        AutoModel,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    common = {"cache_dir": cache_dir, "local_files_only": local_files_only}
    encoder = AutoModel.from_pretrained(pretrained_model, **common)
    classifier = AutoModelForSequenceClassification.from_pretrained(finetuned_model, **common)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model, **common)
    return BertClassifierPair(encoder, classifier, tokenizer)


def _bert_body(model):
    """Resolve HF base model on either AutoModel or AutoModelForSequenceClassification."""
    base_prefix = getattr(model, "base_model_prefix", None)
    body = getattr(model, base_prefix, None) if base_prefix else None
    if body is None or body is model:
        body = getattr(model, "bert", None)
    if body is model or (body is None and hasattr(model, "pooler") and hasattr(model, "encoder")):
        body = model
    if body is None:
        raise TypeError("expected a BERT base encoder or sequence classifier")
    return body


def bert_feature_fn(model, token_batch):
    """Return the pooled CLS representation before the classification head."""
    if not isinstance(token_batch, dict):
        raise TypeError("BERT inputs must be a mapping of tokenizer tensors")
    accepted = {
        key: value
        for key, value in token_batch.items()
        if key in ("input_ids", "attention_mask", "token_type_ids", "position_ids")
    }
    if "input_ids" not in accepted:
        raise KeyError("BERT input mapping must include input_ids")
    outputs = _bert_body(model)(**accepted, return_dict=True)
    pooled = getattr(outputs, "pooler_output", None)
    if pooled is None:
        raise ValueError("BERT encoder must return pooler_output")
    return pooled


def bert_classifier_head(model):
    """Return the sequence classifier head compatible with bert_feature_fn output."""
    head = getattr(model, "classifier", None)
    if head is None:
        raise TypeError("sequence classifier does not expose a classifier head")
    return head


def move_batch_to_device(value, device):
    """Recursively move tensor leaves, preserving dict/list/tuple structure."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_batch_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_batch_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_batch_to_device(item, device) for item in value]
    return value


def forget_preserve_accuracy(labels, predictions, forget_class, num_classes=None):
    """Return target-class accuracy and accuracy over all remaining classes."""
    labels = torch.as_tensor(labels).reshape(-1).long().cpu()
    predictions = torch.as_tensor(predictions).reshape(-1).long().cpu()
    if labels.numel() != predictions.numel() or labels.numel() == 0:
        raise ValueError("labels and predictions must be non-empty and equally sized")
    classes = int(num_classes or max(int(labels.max()), int(predictions.max())) + 1)
    if forget_class < 0 or forget_class >= classes:
        raise ValueError("forget_class is outside the configured label space")
    per_class = []
    for class_id in range(classes):
        mask = labels == class_id
        per_class.append(
            float((predictions[mask] == labels[mask]).float().mean())
            if mask.any()
            else float("nan")
        )
    target = per_class[forget_class]
    retain_mask = labels != forget_class
    preserve = (
        float((predictions[retain_mask] == labels[retain_mask]).float().mean())
        if retain_mask.any()
        else float("nan")
    )
    return {
        "forget_accuracy": target,
        "retain_accuracy": preserve,
        "per_class_accuracy": per_class,
    }
