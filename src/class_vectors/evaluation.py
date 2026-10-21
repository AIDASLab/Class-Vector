"""Classification metrics for clean and edited features."""

import torch

from .core import _default_batch_to_xy, _move_to_device, latent_steer


def classification_metrics(labels, predictions, target_class=None, destination_class=None):
    labels = torch.as_tensor(labels).long().reshape(-1)
    predictions = torch.as_tensor(predictions).long().reshape(-1)
    if labels.numel() == 0 or predictions.shape != labels.shape:
        raise ValueError("labels and predictions must have matching non-empty shapes")
    if (labels < 0).any():
        raise ValueError("labels must be nonnegative class IDs")
    correct = labels == predictions
    result = {
        "accuracy": float(correct.float().mean()),
        "sample_count": labels.numel(),
        "per_class_accuracy": {},
    }
    for value in labels.unique().tolist():
        result["per_class_accuracy"][str(value)] = float(correct[labels == value].float().mean())
    if target_class is not None:
        target = labels == target_class
        result["target_accuracy"] = float(correct[target].float().mean()) if target.any() else None
        result["retain_accuracy"] = (
            float(correct[~target].float().mean()) if (~target).any() else None
        )
        if destination_class is not None:
            result["transfer_success"] = (
                float((predictions[target] == destination_class).float().mean())
                if target.any()
                else None
            )
    return result


def evaluate_encoder(
    model,
    classifier_head,
    data_loader,
    feature_fn,
    *,
    edits=(),
    device="cpu",
    target_class=None,
    destination_class=None,
):
    model.to(device).eval()
    if hasattr(classifier_head, "to"):
        classifier_head.to(device).eval()
    labels_all, clean_all, edited_all = [], [], []
    covered = total = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = _default_batch_to_xy(batch)
            features = feature_fn(model, _move_to_device(inputs, device))
            clean_all.append(classifier_head(features).argmax(-1).cpu())
            shifted = features.clone()
            union = torch.zeros(features.shape[0], dtype=torch.bool, device=features.device)
            for vector, centroid, threshold in edits:
                if threshold is None:
                    vector = torch.as_tensor(vector, device=features.device, dtype=features.dtype)
                    if (
                        vector.ndim != 1
                        or vector.numel() != features.shape[-1]
                        or not torch.isfinite(vector).all()
                    ):
                        raise ValueError(
                            "edit vector must be finite and match the feature dimension"
                        )
                    local = features + vector
                    gate = torch.ones_like(union)
                else:
                    local, gate, _ = latent_steer(features, vector, centroid, threshold)
                shifted += local - features
                union |= gate
            edited_all.append(classifier_head(shifted).argmax(-1).cpu())
            labels = torch.as_tensor(labels)
            if labels.ndim != 1:
                raise ValueError("evaluation requires one class ID per example")
            labels_all.append(labels.cpu())
            covered += int(union.sum())
            total += len(labels)
    if not labels_all:
        raise ValueError("evaluation loader produced no samples")
    labels = torch.cat(labels_all)
    return {
        "clean": classification_metrics(
            labels, torch.cat(clean_all), target_class, destination_class
        ),
        "edited": classification_metrics(
            labels, torch.cat(edited_all), target_class, destination_class
        ),
        "gate_coverage": covered / total,
    }


def evaluate_trigger(
    model,
    head,
    loader,
    feature_fn,
    trigger,
    mask,
    *,
    source_class,
    destination_class,
    mode="patch",
    transparency=1.0,
    device="cpu",
):
    from .trigger import apply_trigger

    labels_all, clean_all, triggered_all = [], [], []
    model.to(device).eval()
    if hasattr(head, "to"):
        head.to(device).eval()
    trigger, mask = trigger.to(device), mask.to(device)
    with torch.no_grad():
        for batch in loader:
            images, labels = _default_batch_to_xy(batch)
            images = images.to(device)
            clean_all.append(head(feature_fn(model, images)).argmax(-1).cpu())
            patched = apply_trigger(images, trigger, mask, mode, transparency)
            triggered_all.append(head(feature_fn(model, patched)).argmax(-1).cpu())
            labels_all.append(torch.as_tensor(labels).cpu())
    if not labels_all:
        raise ValueError("trigger evaluation loader produced no samples")
    labels = torch.cat(labels_all)
    clean = classification_metrics(labels, torch.cat(clean_all), source_class, destination_class)
    triggered = classification_metrics(
        labels, torch.cat(triggered_all), source_class, destination_class
    )
    return {
        "clean": clean,
        "triggered": triggered,
        "clean_accuracy": clean["accuracy"],
        "attack_success_rate": triggered["transfer_success"],
    }
