"""Fit selected encoder parameters to latent editing targets."""

import copy
import fnmatch
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .core import _default_batch_to_xy, _move_to_device
from .optimization import make_optimizer


def _select_inputs(inputs, mask):
    if isinstance(inputs, dict):
        return {key: _select_inputs(value, mask) for key, value in inputs.items()}
    if isinstance(inputs, tuple):
        return tuple(_select_inputs(value, mask) for value in inputs)
    if isinstance(inputs, list):
        return [_select_inputs(value, mask) for value in inputs]
    if torch.is_tensor(inputs):
        return inputs[mask]
    raise TypeError("mapping inputs must contain batched tensors")


def build_mapping_target(reference_features, edit_vector):
    references = torch.as_tensor(reference_features)
    if references.ndim == 1:
        references = references.unsqueeze(0)
    edit = torch.as_tensor(edit_vector, device=references.device, dtype=references.dtype).reshape(
        -1
    )
    if references.ndim != 2 or references.shape[0] == 0:
        raise ValueError("reference_features must be a non-empty [N, D] matrix")
    if references.shape[1] != edit.numel():
        raise ValueError("edit vector dimension does not match references")
    return references.detach().mean(dim=0) + edit.detach()


def configure_trainable_parameters(model, patterns: Iterable[str]):
    """Freeze all parameters except names matching explicit glob patterns."""
    patterns = tuple(patterns)
    if not patterns:
        raise ValueError("at least one editable_parameter_pattern is required")
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(name)
    if not trainable:
        raise ValueError("editable patterns matched no model parameters: {}".format(patterns))
    return tuple(trainable)


def collect_reference_features(
    model,
    data_loader,
    feature_fn: Callable,
    class_id,
    n_references,
    device="cpu",
    batch_to_xy=None,
):
    """Collect a fixed N target-class embeddings before any weight updates."""
    if n_references < 1:
        raise ValueError("n_references must be positive")
    batch_to_xy = batch_to_xy or _default_batch_to_xy
    model.eval()
    collected = []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch_to_xy(batch)
            features = feature_fn(model, _move_to_device(inputs, device))
            if isinstance(features, (tuple, list)):
                features = features[-1]
            mask = torch.as_tensor(labels, device=features.device).reshape(-1) == class_id
            if mask.any():
                selected = features[mask].detach()
                remaining = n_references - sum(value.shape[0] for value in collected)
                collected.append(selected[:remaining])
            if sum(value.shape[0] for value in collected) >= n_references:
                break
    if sum(value.shape[0] for value in collected) < n_references:
        raise ValueError(
            "loader contains fewer than {} references for class {}".format(n_references, class_id)
        )
    return torch.cat(collected, dim=0)


def train_weight_mapping(
    model,
    data_loader,
    feature_fn: Callable,
    class_id,
    edit_vector,
    editable_parameter_patterns,
    n_references=1,
    epochs=1,
    learning_rate=1e-4,
    device="cpu",
    batch_to_xy=None,
    target_mode="fixed_mean",
    optimizer_name="adam",
    weight_decay=0.0,
    train_mode=False,
    reference_model=None,
):
    """Fit a fixed reference mean or frozen per-input features plus the edit."""
    if epochs < 1 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    batch_to_xy = batch_to_xy or _default_batch_to_xy
    model.to(device)
    if target_mode not in ("fixed_mean", "per_input"):
        raise ValueError("target_mode must be fixed_mean or per_input")
    target = None
    if target_mode == "fixed_mean":
        references = collect_reference_features(
            model, data_loader, feature_fn, class_id, n_references, device, batch_to_xy
        )
        target = build_mapping_target(references, edit_vector).to(device)
    else:
        if reference_model is model:
            raise ValueError("reference_model must be separate from the model being edited")
        reference_model = copy.deepcopy(model) if reference_model is None else reference_model
        reference_model.to(device).eval()
    edit = torch.as_tensor(edit_vector, device=device).detach().reshape(-1)
    if not torch.isfinite(edit).all():
        raise ValueError("edit vector must be finite")
    trainable_names = configure_trainable_parameters(model, editable_parameter_patterns)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = make_optimizer(parameters, optimizer_name, learning_rate, weight_decay)
    history = []
    model.train(train_mode)
    for _ in range(epochs):
        losses = []
        for batch in data_loader:
            inputs, labels = batch_to_xy(batch)
            inputs = _move_to_device(inputs, device)
            labels = torch.as_tensor(labels, device=device).reshape(-1)
            mask = labels == class_id
            if not mask.any():
                continue
            if target_mode == "per_input":
                inputs = _select_inputs(inputs, mask)
            features = feature_fn(model, inputs)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            if features.ndim != 2 or features.shape[-1] != edit.numel():
                raise ValueError("feature_fn must return [batch, feature_dim] embeddings")
            if target_mode == "per_input":
                with torch.no_grad():
                    reference = feature_fn(reference_model, inputs)
                    if isinstance(reference, (tuple, list)):
                        reference = reference[-1]
                    if reference.shape != features.shape:
                        raise ValueError("reference and edited feature shapes must match")
                    batch_target = reference + edit.to(reference.dtype)
                selected = features
            else:
                selected = features[mask]
                batch_target = target.expand_as(selected)
            loss = F.mse_loss(selected, batch_target)
            if not torch.isfinite(loss):
                raise ValueError("weight mapping produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if not losses:
            raise ValueError("training loader has no samples for class {}".format(class_id))
        history.append(sum(losses) / len(losses))
    return model, target.detach() if target is not None else None, history, trainable_names
