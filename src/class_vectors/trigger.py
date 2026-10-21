"""Masked pixel optimization against fixed representation targets."""

import torch
import torch.nn.functional as F

from .core import _default_batch_to_xy


def apply_trigger(images, trigger, mask, mode="patch", transparency=1.0):
    if mode not in ("patch", "blend", "additive"):
        raise ValueError("trigger mode must be patch, blend, or additive")
    if not 0 < transparency <= 1:
        raise ValueError("transparency must be in (0, 1]")
    if mode == "additive":
        return images + transparency * trigger * mask
    return images * (1 - transparency * mask) + transparency * trigger * mask


def optimize_trigger_loader(
    encoder,
    loader,
    target,
    initial_trigger,
    mask,
    feature_fn,
    *,
    source_class,
    epochs,
    learning_rate,
    mode="patch",
    transparency=1.0,
    lower_bound=None,
    upper_bound=None,
    max_steps=None,
    device="cpu",
):
    """Fit a shared pattern over source-class batches using a fixed target."""
    if epochs < 1 or learning_rate <= 0 or (max_steps is not None and max_steps < 1):
        raise ValueError("epochs, learning_rate, and max_steps must be positive")
    encoder.to(device).eval()
    trigger = torch.nn.Parameter(initial_trigger.detach().to(device).clone())
    mask = mask.to(device=device, dtype=trigger.dtype)
    target = target.detach().to(device=device, dtype=trigger.dtype).reshape(1, -1)
    optimizer = torch.optim.Adam([trigger], lr=learning_rate)
    losses = []
    for _ in range(epochs):
        updates = 0
        for batch in loader:
            inputs, labels = _default_batch_to_xy(batch)
            selected = torch.as_tensor(labels).reshape(-1) == source_class
            if not selected.any():
                continue
            images = inputs[selected].to(device)
            patched = apply_trigger(images, trigger, mask, mode, transparency)
            features = feature_fn(encoder, patched)
            loss = F.mse_loss(features, target.expand_as(features))
            if not torch.isfinite(loss):
                raise ValueError("trigger optimization produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            (trigger.grad,) = torch.autograd.grad(loss, trigger)
            optimizer.step()
            with torch.no_grad():
                trigger.clamp_(
                    min=lower_bound if lower_bound is not None else -torch.inf,
                    max=upper_bound if upper_bound is not None else torch.inf,
                )
            losses.append(float(loss.detach().cpu()))
            updates += 1
            if max_steps is not None and len(losses) >= max_steps:
                return trigger.detach().masked_fill(mask == 0, 0).cpu(), losses
        if not updates:
            raise ValueError("trigger data contains no source-class samples")
    return trigger.detach().masked_fill(mask == 0, 0).cpu(), losses


def optimize_shared_trigger(
    encoder,
    images,
    target_features,
    mask,
    feature_fn,
    *,
    steps,
    learning_rate,
    lower_bound=None,
    upper_bound=None,
    mode="patch",
    transparency=1.0,
    device="cpu",
):
    """Optimize one shared masked trigger against fixed target embeddings.

    `images` are preprocessed input tensors. `mask` contains values in [0, 1]
    and broadcasts to one image. Apply the returned pattern with the same mask
    and transparency. Supply bounds in model-input space when clipping is required.
    """
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    if mode not in ("patch", "blend", "additive"):
        raise ValueError("trigger mode must be patch, blend, or additive")
    if not 0 < transparency <= 1:
        raise ValueError("transparency must be in (0, 1]")
    images = torch.as_tensor(images, device=device).detach()
    targets = torch.as_tensor(target_features, device=device, dtype=images.dtype).detach()
    mask = torch.as_tensor(mask, device=device, dtype=images.dtype)
    if images.ndim < 3 or targets.ndim != 2 or targets.shape[0] != images.shape[0]:
        raise ValueError("images and target_features must have matching batch dimensions")
    while mask.ndim < images.ndim:
        mask = mask.unsqueeze(0)
    try:
        shape = torch.broadcast_shapes(images[:1].shape, mask.shape)
    except RuntimeError as exc:
        raise ValueError("mask must broadcast to a single image") from exc
    if (
        tuple(shape) != tuple(images[:1].shape)
        or not torch.isfinite(mask).all()
        or not ((mask >= 0) & (mask <= 1)).all()
    ):
        raise ValueError("mask must broadcast to one image with values in [0, 1]")

    encoder = encoder.to(device).eval()
    trigger = torch.nn.Parameter(torch.zeros_like(images[:1]))
    optimizer = torch.optim.Adam([trigger], lr=learning_rate)
    losses = []
    for _ in range(steps):
        patched = apply_trigger(images, trigger, mask, mode, transparency)
        features = feature_fn(encoder, patched)
        if isinstance(features, (tuple, list)):
            features = features[-1]
        if features.shape != targets.shape:
            raise ValueError("feature_fn output must match target_features shape")
        loss = F.mse_loss(features, targets)
        optimizer.zero_grad(set_to_none=True)
        (trigger.grad,) = torch.autograd.grad(loss, trigger)
        optimizer.step()
        with torch.no_grad():
            if lower_bound is not None or upper_bound is not None:
                trigger.clamp_(
                    min=lower_bound if lower_bound is not None else -torch.inf,
                    max=upper_bound if upper_bound is not None else torch.inf,
                )
        losses.append(float(loss.detach().cpu()))
    return trigger.detach().masked_fill(mask == 0, 0).cpu(), losses
