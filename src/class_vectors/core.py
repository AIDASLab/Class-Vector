"""Core Class Vector math following arXiv:2510.11268v2."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class ClassVectorBank:
    """Per-class vectors, paired encoder centroids, and support counts."""

    vectors: Dict[int, torch.Tensor]
    pretrained_centroids: Dict[int, torch.Tensor]
    finetuned_centroids: Dict[int, torch.Tensor]
    counts: Dict[int, int]

    def __post_init__(self):
        keys = set(self.vectors)
        if not keys or any(
            set(value) != keys
            for value in (self.pretrained_centroids, self.finetuned_centroids, self.counts)
        ):
            raise ValueError(
                "bank vectors, centroids, and counts must have identical non-empty class IDs"
            )
        dimension = None
        for key in keys:
            if (
                int(key) != key
                or key < 0
                or int(self.counts[key]) != self.counts[key]
                or self.counts[key] < 1
            ):
                raise ValueError("bank requires nonnegative class IDs and positive integer counts")
            for values in (self.vectors, self.pretrained_centroids, self.finetuned_centroids):
                tensor = values[key]
                if (
                    not torch.is_tensor(tensor)
                    or tensor.ndim != 1
                    or not tensor.numel()
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError("bank entries must be non-empty finite feature vectors")
                dimension = dimension or tensor.numel()
                if tensor.numel() != dimension:
                    raise ValueError("bank feature dimensions must match")

    def cpu(self):
        """Return a detached CPU copy suitable for portable serialization."""
        return ClassVectorBank(
            vectors={key: value.detach().cpu() for key, value in self.vectors.items()},
            pretrained_centroids={
                key: value.detach().cpu() for key, value in self.pretrained_centroids.items()
            },
            finetuned_centroids={
                key: value.detach().cpu() for key, value in self.finetuned_centroids.items()
            },
            counts=dict(self.counts),
        )


def _feature_matrix(features, name):
    value = torch.as_tensor(features)
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("{} must have shape [samples, features]".format(name))
    if not torch.is_floating_point(value):
        value = value.float()
    if not torch.isfinite(value).all():
        raise ValueError("{} contains non-finite values".format(name))
    return value


def compute_class_vector_bank(
    pretrained_features,
    finetuned_features,
    labels,
    class_ids: Optional[Iterable[int]] = None,
) -> ClassVectorBank:
    """Compute kappa_c = mean(f_ft|c) - mean(f_pre|c) on paired samples.

    Feature matrices must contain aligned representations of the same support samples.
    """
    pre = _feature_matrix(pretrained_features, "pretrained_features")
    ft = _feature_matrix(finetuned_features, "finetuned_features")
    labels = torch.as_tensor(labels, device=pre.device).reshape(-1)
    if (
        not torch.isfinite(labels).all()
        or not torch.equal(labels, labels.long().to(labels.dtype))
        or (labels < 0).any()
    ):
        raise ValueError("labels must be nonnegative integer class IDs")
    if pre.shape != ft.shape:
        raise ValueError("pretrained and fine-tuned features must have identical shapes")
    if pre.shape[0] != labels.numel():
        raise ValueError("feature rows and labels must have identical lengths")
    if pre.device != ft.device:
        raise ValueError("feature matrices must be on the same device")

    selected = list(class_ids) if class_ids is not None else torch.unique(labels).tolist()
    if not selected:
        raise ValueError("at least one class is required")
    if len(selected) != len(set(selected)):
        raise ValueError("class_ids must not contain duplicates")

    vectors, pre_centroids, ft_centroids, counts = {}, {}, {}, {}
    for class_id in selected:
        mask = labels == class_id
        count = int(mask.sum().item())
        if count == 0:
            raise ValueError("class {} has no support samples".format(class_id))
        pre_mean = pre[mask].mean(dim=0)
        ft_mean = ft[mask].mean(dim=0)
        pre_centroids[int(class_id)] = pre_mean
        ft_centroids[int(class_id)] = ft_mean
        vectors[int(class_id)] = ft_mean - pre_mean
        counts[int(class_id)] = count
    return ClassVectorBank(vectors, pre_centroids, ft_centroids, counts)


def compose_edit_vector(class_vectors: Dict[int, torch.Tensor], coefficients: Dict[int, float]):
    """Compose an edit, e.g. -kappa_target + kappa_destination."""
    if not coefficients:
        raise ValueError("at least one class-vector coefficient is required")
    result = None
    for class_id, coefficient in coefficients.items():
        if class_id not in class_vectors:
            raise KeyError("class vector {} is missing".format(class_id))
        term = class_vectors[class_id] * float(coefficient)
        if (
            not math.isfinite(float(coefficient))
            or term.ndim != 1
            or not torch.isfinite(term).all()
        ):
            raise ValueError("coefficients and class vectors must be finite")
        if result is not None and term.shape != result.shape:
            raise ValueError("class vectors must have matching dimensions")
        result = term.clone() if result is None else result + term
    return result


def latent_steer(features, edit_vector, target_centroid, threshold=0.5):
    """Apply an edit where cosine(features, target_centroid) > threshold.

    Features and the target-class centroid use the fine-tuned encoder's representation.
    """
    features = _feature_matrix(features, "features")
    edit_vector = torch.as_tensor(
        edit_vector, device=features.device, dtype=features.dtype
    ).reshape(-1)
    centroid = torch.as_tensor(
        target_centroid, device=features.device, dtype=features.dtype
    ).reshape(-1)
    if not torch.isfinite(edit_vector).all() or not torch.isfinite(centroid).all():
        raise ValueError("edit vector and target centroid must be finite")
    if edit_vector.shape != features.shape[1:] or centroid.shape != features.shape[1:]:
        raise ValueError("edit vector and target centroid must match the feature dimension")
    if not -1.0 <= float(threshold) <= 1.0:
        raise ValueError("cosine threshold must be in [-1, 1]")
    similarities = F.cosine_similarity(features, centroid.unsqueeze(0), dim=-1)
    gate = similarities > float(threshold)
    edited = features + gate.to(features.dtype).unsqueeze(-1) * edit_vector.unsqueeze(0)
    return edited, gate, similarities


class LatentSteeringClassifier(torch.nn.Module):
    """Classifier with cosine-gated feature editing at inference time."""

    def __init__(
        self,
        encoder,
        head,
        edit_vector,
        target_centroid,
        threshold=0.5,
        feature_fn=None,
    ):
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.register_buffer("edit_vector", torch.as_tensor(edit_vector).detach().clone())
        self.register_buffer("target_centroid", torch.as_tensor(target_centroid).detach().clone())
        self.threshold = float(threshold)
        self.feature_fn = feature_fn

    def forward(self, inputs, return_details=False):
        if self.feature_fn is not None:
            features = self.feature_fn(self.encoder, inputs)
        elif hasattr(self.encoder, "encode_image"):
            features = self.encoder.encode_image(inputs)
        else:
            features = self.encoder(inputs)
        edited, gate, similarities = latent_steer(
            features, self.edit_vector, self.target_centroid, self.threshold
        )
        logits = self.head(edited)
        if return_details:
            return logits, edited, gate, similarities
        return logits


def save_class_vector_bank(bank: ClassVectorBank, path):
    """Save tensor-only state on CPU for portable, safe loading."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_bank = bank.cpu()
    torch.save(
        {
            "vectors": cpu_bank.vectors,
            "pretrained_centroids": cpu_bank.pretrained_centroids,
            "finetuned_centroids": cpu_bank.finetuned_centroids,
            "counts": cpu_bank.counts,
        },
        str(path),
    )


def load_class_vector_bank(path, map_location="cpu"):
    """Load a tensor-only bank without executing arbitrary pickle payloads."""
    payload = torch.load(str(Path(path).expanduser()), map_location=map_location, weights_only=True)
    required = {"vectors", "pretrained_centroids", "finetuned_centroids", "counts"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("invalid class-vector bank file")
    return ClassVectorBank(**payload)


def _default_batch_to_xy(batch):
    if isinstance(batch, dict):
        y_key = next(
            (key for key in ("labels", "label", "targets", "target") if key in batch),
            None,
        )
        if y_key is None:
            raise KeyError("batch dict must contain image/input and label/target fields")
        input_keys = {"labels", "label", "targets", "target"}
        inputs = {key: value for key, value in batch.items() if key not in input_keys}
        if not inputs:
            raise KeyError("batch dict must contain model inputs")
        if len(inputs) == 1 and next(iter(inputs)) in (
            "images",
            "image",
            "inputs",
            "input",
            "pixel_values",
        ):
            inputs = next(iter(inputs.values()))
        return inputs, batch[y_key]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("batch must be a pair or a dict containing inputs and labels")


def _move_to_device(value, device):
    """Move nested tensor inputs (including Hugging Face batches) to a device."""
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    return value


def extract_labeled_features(encoder, data_loader, device="cpu", feature_fn=None, batch_to_xy=None):
    """Extract aligned features/labels without gradients; works on CPU or GPU."""
    batch_to_xy = batch_to_xy or _default_batch_to_xy
    if feature_fn is None:
        feature_fn = lambda model, images: (
            model.encode_image(images) if hasattr(model, "encode_image") else model(images)
        )
    encoder.to(device).eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch_to_xy(batch)
            inputs = _move_to_device(inputs, device)
            features = feature_fn(encoder, inputs)
            if isinstance(features, (tuple, list)):
                features = features[-1]
            features = _feature_matrix(features, "extracted features")
            labels = torch.as_tensor(labels).reshape(-1)
            if features.shape[0] != labels.numel():
                raise ValueError("feature rows and labels differ in a batch")
            all_features.append(features.detach().cpu())
            all_labels.append(labels.detach().cpu())
    if not all_features:
        raise ValueError("data loader produced no batches")
    return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)


def extract_class_vector_bank(
    pretrained_encoder,
    finetuned_encoder,
    data_loader,
    feature_fn,
    *,
    device="cpu",
    class_ids=None,
    samples_per_class=None,
):
    """Accumulate paired class centroids with memory proportional to class count."""
    if samples_per_class is not None and samples_per_class < 1:
        raise ValueError("samples_per_class must be positive")
    selected = set(class_ids) if class_ids is not None else None
    pretrained_encoder.to(device).eval()
    finetuned_encoder.to(device).eval()
    sums_pre, sums_ft, counts = {}, {}, {}
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = _default_batch_to_xy(batch)
            inputs = _move_to_device(inputs, device)
            pre = _feature_matrix(
                feature_fn(pretrained_encoder, inputs), "pretrained features"
            ).cpu()
            ft = _feature_matrix(feature_fn(finetuned_encoder, inputs), "fine-tuned features").cpu()
            labels = torch.as_tensor(labels).cpu()
            if labels.ndim != 1 or pre.shape != ft.shape or len(labels) != len(pre):
                raise ValueError("paired features and single-label batches must have matching rows")
            if not torch.equal(labels, labels.long().to(labels.dtype)) or (labels < 0).any():
                raise ValueError("labels must be nonnegative integer class IDs")
            for key in labels.unique().tolist():
                if selected is not None and key not in selected:
                    continue
                mask = labels == key
                rows_pre, rows_ft = pre[mask], ft[mask]
                if samples_per_class is not None:
                    remaining = max(0, samples_per_class - counts.get(key, 0))
                    rows_pre, rows_ft = rows_pre[:remaining], rows_ft[:remaining]
                if not len(rows_pre):
                    continue
                sum_pre, sum_ft = rows_pre.double().sum(0), rows_ft.double().sum(0)
                sums_pre[key] = sums_pre.get(key, torch.zeros_like(sum_pre)) + sum_pre
                sums_ft[key] = sums_ft.get(key, torch.zeros_like(sum_ft)) + sum_ft
                counts[key] = counts.get(key, 0) + len(rows_pre)
            if (
                selected
                and samples_per_class
                and all(counts.get(key, 0) >= samples_per_class for key in selected)
            ):
                break
    if selected is not None and selected != set(counts):
        raise ValueError("support data is missing configured class IDs")
    means_pre = {key: (value / counts[key]).float() for key, value in sums_pre.items()}
    means_ft = {key: (value / counts[key]).float() for key, value in sums_ft.items()}
    vectors = {key: means_ft[key] - means_pre[key] for key in counts}
    return ClassVectorBank(vectors, means_pre, means_ft, counts)


def extract_paired_features(
    pretrained_encoder,
    finetuned_encoder,
    data_loader,
    device="cpu",
    pretrained_feature_fn=None,
    finetuned_feature_fn=None,
    batch_to_xy=None,
    pretrained_input_fn=None,
    finetuned_input_fn=None,
):
    """Extract both encoders on the same samples and in the same pass/order."""
    batch_to_xy = batch_to_xy or _default_batch_to_xy
    pretrained_feature_fn = pretrained_feature_fn or (
        lambda model, images: (
            model.encode_image(images) if hasattr(model, "encode_image") else model(images)
        )
    )
    finetuned_feature_fn = finetuned_feature_fn or (
        lambda model, images: (
            model.encode_image(images) if hasattr(model, "encode_image") else model(images)
        )
    )
    pretrained_encoder.to(device).eval()
    finetuned_encoder.to(device).eval()
    pre_batches, ft_batches, labels_out = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            inputs, labels = batch_to_xy(batch)
            inputs_pre = pretrained_input_fn(inputs) if pretrained_input_fn else inputs
            inputs_ft = finetuned_input_fn(inputs) if finetuned_input_fn else inputs
            pre = pretrained_feature_fn(pretrained_encoder, _move_to_device(inputs_pre, device))
            ft = finetuned_feature_fn(finetuned_encoder, _move_to_device(inputs_ft, device))
            if isinstance(pre, (tuple, list)):
                pre = pre[-1]
            if isinstance(ft, (tuple, list)):
                ft = ft[-1]
            pre = _feature_matrix(pre, "pretrained features")
            ft = _feature_matrix(ft, "fine-tuned features")
            labels = torch.as_tensor(labels).reshape(-1)
            if pre.shape != ft.shape or pre.shape[0] != labels.numel():
                raise ValueError(
                    "paired feature batches must have matching [samples, features] shapes"
                )
            pre_batches.append(pre.detach().cpu())
            ft_batches.append(ft.detach().cpu())
            labels_out.append(labels.detach().cpu())
    if not pre_batches:
        raise ValueError("data loader produced no batches")
    return torch.cat(pre_batches), torch.cat(ft_batches), torch.cat(labels_out)
