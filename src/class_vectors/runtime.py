"""Model, data-loader, and classification-head construction."""

import importlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from .architecture_adapters import (
    bert_classifier_head,
    bert_feature_fn,
    build_image_classifier,
    load_bert_classifier_pair,
    vision_feature_fn,
)
from .config import data_section, sampling_class
from .model_loader import _state_dict_from_checkpoint, load_encoder_pair


@dataclass
class ModelRuntime:
    pretrained: nn.Module
    finetuned: nn.Module
    feature_fn: object
    preprocess: object = None
    tokenizer: object = None
    head: object = None


class NormalizedHead(nn.Module):
    def __init__(self, weights, bias=None, normalize=True):
        super().__init__()
        self.register_buffer("weight", weights.float())
        self.register_buffer("bias", bias.float() if bias is not None else None)
        self.normalize = normalize

    def forward(self, features):
        if self.normalize:
            features = torch.nn.functional.normalize(features, dim=-1)
        return torch.nn.functional.linear(features.float(), self.weight, self.bias)


def _to_rgb(image):
    return image.convert("RGB")


def vision_preprocess():
    from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToTensor

    return Compose(
        [
            Resize((224, 224)),
            Lambda(_to_rgb),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def device_for(config):
    value = config.model.device
    return ("cuda" if torch.cuda.is_available() else "cpu") if value == "auto" else value


def load_runtime(config):
    model = config.model
    device = device_for(config)
    if model.family == "clip":
        pair = load_encoder_pair(model)
        runtime = ModelRuntime(
            pair.pretrained,
            pair.finetuned,
            lambda m, x: m.encode_image(x),
            pair.val_preprocess,
        )
        if model.head_checkpoint:
            state = _state_dict_from_checkpoint(model.head_checkpoint, "cpu")
            runtime.head = NormalizedHead(
                state["weight"], state.get("bias"), model.normalize_features
            )
    elif model.family == "vision":
        if model.num_classes is None:
            raise ValueError("model.num_classes is required for vision classifiers")
        pre = build_image_classifier(
            model.name,
            model.num_classes,
            bool(model.pretrained) and not model.pretrained_checkpoint,
            model.pretrained_checkpoint,
        )
        ft = build_image_classifier(
            model.name, model.num_classes, False, model.finetuned_checkpoint
        )
        if not model.finetuned_checkpoint:
            raise ValueError("vision classifiers require model.finetuned_checkpoint")
        runtime = ModelRuntime(pre, ft, vision_feature_fn, vision_preprocess(), head=ft.classifier)
    else:
        pair = load_bert_classifier_pair(
            model.name,
            model.finetuned_checkpoint or model.finetuned_pretrained,
            model.cache_dir,
            model.local_files_only,
        )
        runtime = ModelRuntime(
            pair.pretrained_encoder,
            pair.finetuned_classifier,
            bert_feature_fn,
            tokenizer=pair.tokenizer,
            head=bert_classifier_head(pair.finetuned_classifier),
        )
    runtime.pretrained.to(device).eval()
    runtime.finetuned.to(device).eval()
    if runtime.head is not None:
        runtime.head.to(device).eval()
    return runtime


class JsonlImages(Dataset):
    def __init__(self, path, root, preprocess, label_field="label"):
        self.records = [
            json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
        ]
        self.root = Path(root)
        self.preprocess = preprocess
        self.label_field = label_field

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        row = self.records[index]
        with Image.open(self.root / row["file_name"]) as image:
            value = self.preprocess(image.convert("RGB"))
        return value, int(row[self.label_field])


def _split_spec(config, purpose):
    return data_section(config, purpose)


def _dataset_labels(dataset):
    if isinstance(dataset, Subset):
        return _dataset_labels(dataset.dataset)[dataset.indices]
    if isinstance(dataset, TensorDataset):
        return torch.as_tensor(dataset.tensors[-1]).reshape(-1)
    if "labels" in getattr(dataset, "column_names", ()):
        return torch.as_tensor(dataset["labels"]).reshape(-1)
    for name in ("targets", "labels"):
        value = getattr(dataset, name, None)
        if value is not None:
            return torch.as_tensor(value).reshape(-1)
    for name in ("samples", "_samples"):
        value = getattr(dataset, name, None)
        if value is not None:
            return torch.tensor([row[1] for row in value])
    if isinstance(dataset, JsonlImages):
        return torch.tensor([row[dataset.label_field] for row in dataset.records])
    return torch.tensor([int(dataset[index][1]) for index in range(len(dataset))])


def _holdout_dataset(dataset, spec, backend):
    fraction = spec.get("holdout_fraction")
    if fraction is None:
        return dataset
    if not 0 < float(fraction) < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    partition = spec.get("holdout_partition", "train")
    if partition not in ("train", "validation"):
        raise ValueError("holdout_partition must be train or validation")
    seed = int(spec.get("holdout_seed", 42))
    if backend == "huggingface":
        partitions = dataset.train_test_split(test_size=float(fraction), seed=seed)
        return partitions["train" if partition == "train" else "test"]
    count = int(len(dataset) * float(fraction))
    if not 0 < count < len(dataset):
        raise ValueError("holdout split requires non-empty train and validation partitions")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))
    indices = indices[:-count] if partition == "train" else indices[-count:]
    return Subset(dataset, indices.tolist())


def make_loader(config, runtime, purpose="support"):
    spec = _split_spec(config, purpose)
    backend = spec.get("backend", "registry")
    split = spec["split"]
    class_names = spec.get("class_names")
    if backend == "tensor":
        payload = torch.load(spec["path"], map_location="cpu", weights_only=True)
        dataset = TensorDataset(payload["inputs"], payload["labels"])
    elif backend == "imagefolder":
        from torchvision.datasets import ImageFolder

        dataset = ImageFolder(spec["root"], transform=runtime.preprocess)
        class_names = class_names or dataset.classes
    elif backend == "jsonl":
        dataset = JsonlImages(
            spec["path"],
            spec["root"],
            runtime.preprocess,
            spec.get("label_field", "label"),
        )
    elif backend == "huggingface":
        from datasets import load_dataset, load_from_disk

        if spec.get("root"):
            Path(spec["root"]).mkdir(parents=True, exist_ok=True)
        if runtime.tokenizer is None:
            raise ValueError("huggingface data requires a tokenizer")
        if spec.get("path"):
            data = load_from_disk(spec["path"])
            dataset = data[split] if hasattr(data, "keys") else data
        else:
            dataset = load_dataset(spec["name"], split=split, cache_dir=spec.get("root"))
        dataset = _holdout_dataset(dataset, spec, backend)
        text_field, label_field = (
            spec.get("text_field", "text"),
            spec.get("label_field", "label"),
        )
        tokenizer = runtime.tokenizer

        def tokenize(batch):
            encoded = tokenizer(
                batch[text_field],
                truncation=True,
                padding="max_length",
                max_length=int(spec.get("max_length", 128)),
            )
            encoded["labels"] = batch[label_field]
            return encoded

        dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
        dataset.set_format(type="torch")
    else:
        if backend != "registry":
            raise ValueError("unsupported data.backend: {}".format(backend))
        modules = {
            "Cars": "cars",
            "MNIST": "mnist",
            "SVHN": "svhn",
            "EuroSAT": "eurosat",
            "GTSRB": "gtsrb",
            "DTD": "dtd",
            "RESISC45": "resisc45",
            "CIFAR10": "cifar10",
            "CIFAR100": "cifar100",
            "SUN397": "sun397",
        }
        name = spec["name"]
        if name not in modules:
            raise ValueError("use imagefolder or jsonl for dataset {}".format(name))
        cls = getattr(importlib.import_module("src.datasets." + modules[name]), name)
        parameters = inspect.signature(cls).parameters
        kwargs = {
            key: spec[key]
            for key in ("batch_size", "num_workers")
            if key in parameters and key in spec
        }
        root_key = next(
            (key for key in ("location", "root_dir", "root") if key in parameters), None
        )
        if root_key:
            kwargs[root_key] = spec["root"]
        wrapper = cls(runtime.preprocess, **kwargs)
        class_names = class_names or wrapper.classnames
        if split == "val":
            raise ValueError(
                "configure an explicit validation split with imagefolder, jsonl, or huggingface"
            )
        if split not in ("train", "test"):
            raise ValueError("registry split must be train or test")
        dataset = getattr(wrapper, split + "_dataset")
    if backend != "huggingface":
        dataset = _holdout_dataset(dataset, spec, backend)
    fraction = float(spec.get("fraction", 1.0))
    if not 0 < fraction <= 1:
        raise ValueError("data.fraction must be in (0, 1]")
    partition = spec.get("partition", "selected")
    if partition not in ("selected", "remainder"):
        raise ValueError("data.partition must be selected or remainder")
    sample_class, sample_count = sampling_class(config, spec), spec.get("sample_count")
    for name, value, minimum in (
        ("sample_class", sample_class, 0),
        ("sample_count", sample_count, 1),
    ):
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < minimum
        ):
            raise ValueError("data.{} must be an integer >= {}".format(name, minimum))
    if (
        fraction < 1
        or partition == "remainder"
        or sample_count is not None
        or sample_class is not None
    ):
        generator = torch.Generator().manual_seed(int(config.raw.get("seed", 0)))
        eligible = torch.arange(len(dataset))
        if sample_class is not None:
            eligible = eligible[_dataset_labels(dataset) == sample_class]
        if not len(eligible):
            raise ValueError("dataset contains no eligible samples")
        count = sample_count if sample_count is not None else max(1, int(len(eligible) * fraction))
        if count > len(eligible):
            raise ValueError("dataset has fewer eligible samples than sample_count")
        if backend == "huggingface":
            import numpy as np

            order = torch.from_numpy(
                np.random.default_rng(int(config.raw.get("seed", 0))).permutation(len(eligible))
            )
        else:
            order = torch.randperm(len(eligible), generator=generator)
        chosen = eligible[order[:count]]
        if partition == "selected":
            indices = chosen
        else:
            remaining = torch.ones(len(dataset), dtype=torch.bool)
            remaining[chosen] = False
            indices = torch.arange(len(dataset))[remaining]
        if not len(indices):
            raise ValueError("configured partition contains no samples")
        dataset = Subset(dataset, indices.tolist())
    loader = DataLoader(
        dataset,
        batch_size=int(spec.get("batch_size", 64)),
        shuffle=bool(spec.get("shuffle", False)),
        num_workers=int(spec.get("num_workers", 0)),
        generator=torch.Generator().manual_seed(int(config.raw.get("seed", 0))),
    )
    return loader, class_names


def ensure_head(config, runtime, class_names):
    if runtime.head is not None:
        return runtime.head
    if config.model.family != "clip" or not class_names:
        raise ValueError("set model.head_checkpoint or data.class_names")
    if not getattr(runtime.finetuned, "_class_vector_text_available", True):
        raise ValueError("visual-only checkpoints require model.head_checkpoint")
    import open_clip

    templates = config.data.get("templates", ["a photo of a {}."])
    tokenizer = open_clip.get_tokenizer(config.model.name)
    device = device_for(config)
    weights = []
    with torch.no_grad():
        for name in class_names:
            tokens = tokenizer([template.format(name) for template in templates]).to(device)
            embeddings = torch.nn.functional.normalize(
                runtime.finetuned.encode_text(tokens).float(), dim=-1
            )
            weights.append(torch.nn.functional.normalize(embeddings.mean(dim=0), dim=-1))
        weight = torch.stack(weights) * runtime.finetuned.logit_scale.exp()
    runtime.head = (
        NormalizedHead(weight, normalize=config.model.normalize_features).to(device).eval()
    )
    return runtime.head
