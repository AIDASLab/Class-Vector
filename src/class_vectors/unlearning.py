"""Feature extraction and latent unlearning for image and text classifiers."""

from typing import Callable, Iterable, Optional

import torch

from .architecture_adapters import forget_preserve_accuracy
from .core import (
    _default_batch_to_xy,
    _move_to_device,
    compute_class_vector_bank,
    extract_paired_features,
    latent_steer,
)


def extract_architecture_class_vectors(
    pretrained_encoder,
    finetuned_classifier,
    support_loader,
    feature_fn: Callable,
    class_ids: Optional[Iterable[int]] = None,
    device="cpu",
    batch_to_xy=None,
):
    """Extract aligned pre/finetuned features and compute per-class vectors.

    For image classifiers pass ``vision_feature_fn``; for BERT pass
    ``bert_feature_fn`` and a loader yielding tokenizer tensors plus labels.
    """
    pre, ft, labels = extract_paired_features(
        pretrained_encoder,
        finetuned_classifier,
        support_loader,
        device=device,
        pretrained_feature_fn=feature_fn,
        finetuned_feature_fn=feature_fn,
        batch_to_xy=batch_to_xy,
    )
    return compute_class_vector_bank(pre, ft, labels, class_ids=class_ids)


def evaluate_latent_unlearning(
    model,
    data_loader,
    class_vector_bank,
    class_id: int,
    feature_fn: Callable,
    classifier_head: Callable,
    scale: float = -1.5,
    cosine_threshold: float = 0.5,
    num_classes: Optional[int] = None,
    device="cpu",
    batch_to_xy=None,
):
    """Evaluate cosine-gated unlearning with fixed encoder weights."""
    if class_id not in class_vector_bank.vectors:
        raise KeyError("class {} is missing from the class-vector bank".format(class_id))
    batch_to_xy = batch_to_xy or _default_batch_to_xy
    model.to(device).eval()
    edit_vector = class_vector_bank.vectors[class_id].to(device) * float(scale)
    target_centroid = class_vector_bank.finetuned_centroids[class_id].to(device)
    labels_all, predictions_all = [], []
    gate_total = 0
    sample_total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch_to_xy(batch)
            inputs = _move_to_device(inputs, device)
            features = feature_fn(model, inputs)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            edited, gate, _ = latent_steer(
                features, edit_vector, target_centroid, threshold=cosine_threshold
            )
            logits = classifier_head(edited)
            predictions_all.append(logits.argmax(dim=-1).detach().cpu())
            labels_all.append(torch.as_tensor(labels).reshape(-1).detach().cpu())
            gate_total += int(gate.sum().item())
            sample_total += int(gate.numel())
    if not labels_all:
        raise ValueError("evaluation data loader produced no batches")
    labels = torch.cat(labels_all)
    predictions = torch.cat(predictions_all)
    metrics = forget_preserve_accuracy(labels, predictions, class_id, num_classes)
    metrics["gate_coverage"] = gate_total / sample_total
    return metrics
