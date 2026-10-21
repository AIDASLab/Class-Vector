"""Download and prepare task-specific CLIP checkpoints."""

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

import torch
import yaml
from filelock import FileLock

from .checkpoints import convert_checkpoint
from .config import data_section, vector_source_config
from .model_loader import _build_encoder, _state_dict_from_checkpoint
from .runtime import ModelRuntime, device_for, ensure_head, make_loader

CATALOG = Path(__file__).resolve().parents[2] / "configs/class_vectors/checkpoints.yaml"


def _source(model, dataset):
    catalog = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    try:
        identifier = catalog["task_encoders"][model][dataset]
    except KeyError as error:
        raise ValueError(
            "no automatic checkpoint source for {} / {}".format(model, dataset)
        ) from error
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier):
        raise ValueError("invalid Google Drive checkpoint identifier")
    return identifier


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish(staged, destination):
    """Publish a complete staged file without replacing an existing destination."""
    os.link(staged, destination)


def _receipt(path):
    return path.with_suffix(path.suffix + ".sha256.json")


def _verify_receipt(path, model, dataset, head_spec=None):
    receipt = _receipt(path)
    if receipt.exists():
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if data.get("model") != model or data.get("dataset") != dataset:
            raise ValueError("checkpoint source metadata does not match this configuration")
        if data.get("sha256") != _sha256(path):
            raise ValueError("checkpoint checksum mismatch: {}".format(path))
        if head_spec is not None and data.get("head_spec") != head_spec:
            raise ValueError("head settings changed; select a new model.head_checkpoint path")


def _write_receipt(path, model, dataset, source, raw_sha256=None, head_spec=None):
    data = {"model": model, "dataset": dataset, "source": source, "sha256": _sha256(path)}
    if raw_sha256:
        data["download_sha256"] = raw_sha256
    if head_spec is not None:
        data["head_spec"] = head_spec
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".receipt-") as directory:
        staged = Path(directory) / "receipt.json"
        staged.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        _publish(staged, _receipt(path))


def _check_archive(path):
    try:
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError("download is not a PyTorch checkpoint archive")
            if archive.testzip() is not None:
                raise ValueError("downloaded checkpoint failed its archive checksum")
    except zipfile.BadZipFile as error:
        raise ValueError("downloaded checkpoint is incomplete or is not a ZIP archive") from error


def _download_checkpoint(identifier, cache):
    import gdown

    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (identifier + ".pt")
    with FileLock(str(destination) + ".lock", timeout=60):
        if destination.exists():
            _check_archive(destination)
            return destination
        partial = destination.with_suffix(".part")
        result = gdown.download(
            id=identifier, output=str(partial), quiet=False, use_cookies=False, resume=True
        )
        if result is None or not partial.is_file():
            raise RuntimeError("checkpoint download failed; rerun prepare-clip to retry")
        try:
            _check_archive(partial)
        except ValueError:
            partial.unlink()
            raise
        _publish(partial, destination)
        partial.unlink()
    return destination


def _check_encoder(config, path):
    state = _state_dict_from_checkpoint(path, "cpu")
    if not all(torch.isfinite(value).all() for value in state.values()):
        raise ValueError("encoder checkpoint contains non-finite values")
    del state
    model, _, _ = _build_encoder(
        config.model.name,
        config.model.finetuned_pretrained or config.model.pretrained,
        path,
        config.model.cache_dir,
        "cpu",
    )
    del model


def _prepare_encoder(config, dataset, identifier, destination, trust_pickle):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(destination) + ".lock", timeout=60):
        if destination.exists():
            _verify_receipt(destination, config.model.name, dataset)
            _check_encoder(config, destination)
            return True
        if not trust_pickle:
            raise ValueError("prepare-clip requires --trust-pickle for Task Arithmetic model files")
        if _receipt(destination).exists():
            raise FileExistsError(
                "checkpoint metadata already exists: {}".format(_receipt(destination))
            )
        cache = Path(config.model.cache_dir or destination.parent) / "task-arithmetic"
        raw = _download_checkpoint(identifier, cache)
        with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".encoder-") as directory:
            staged = Path(directory) / "encoder.pt"
            convert_checkpoint(raw, staged, trust_pickle=True, component="encoder")
            _check_encoder(config, staged)
            _publish(staged, destination)
        _write_receipt(destination, config.model.name, dataset, identifier, _sha256(raw))
    return False


def prepare_clip(config, trust_pickle=False):
    """Prepare the task encoder, pretrained encoder cache, dataset, and linear head."""
    if config.model.family != "clip":
        raise ValueError("prepare-clip requires model.family: clip")
    for purpose in ("support", "mapping", "evaluation"):
        spec = data_section(config, purpose)
        if spec.get("backend", "registry") != "registry":
            continue
        name, root = spec.get("name"), Path(spec.get("root", "."))
        required = []
        if name == "EuroSAT":
            required = [root / "EuroSAT_splits" / part for part in ("train", "test")]
        elif name == "DTD":
            split = "test" if (root / "dtd/test").is_dir() else "val"
            required = [root / "dtd" / part for part in ("train", split)]
        elif name == "RESISC45":
            required = [root / "resisc45/NWPU-RESISC45"] + [
                root / "resisc45" / ("resisc45-" + part + ".txt") for part in ("train", "test")
            ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "prepare local {} data before downloading weights: {}".format(
                    name, ", ".join(missing)
                )
            )
    source = vector_source_config(config)
    if source is not config:
        prepare_clip(source, trust_pickle=trust_pickle)
    if not config.model.finetuned_checkpoint or not config.model.head_checkpoint:
        raise ValueError("set model.finetuned_checkpoint and model.head_checkpoint")
    dataset = config.raw.get("preparation", {}).get("dataset", config.data.get("name"))
    identifier = _source(config.model.name, dataset)
    destination, head_path = (
        Path(config.model.finetuned_checkpoint),
        Path(config.model.head_checkpoint),
    )
    if destination.resolve() == head_path.resolve():
        raise ValueError("encoder and head destinations must differ")
    reused = _prepare_encoder(config, dataset, identifier, destination, trust_pickle)
    model, _, preprocess = _build_encoder(
        config.model.name,
        config.model.pretrained,
        config.model.pretrained_checkpoint,
        config.model.cache_dir,
        device_for(config),
    )
    runtime = ModelRuntime(
        model, model, lambda encoder, images: encoder.encode_image(images), preprocess
    )
    _, class_names = make_loader(config, runtime, "support")
    make_loader(config, runtime, "evaluation")
    if not class_names:
        raise ValueError("prepare-clip requires dataset class names")
    import open_clip

    width = open_clip.get_model_config(config.model.name)["embed_dim"]
    head_spec = {
        "class_names": list(class_names),
        "templates": config.data.get("templates", ["a photo of a {}."]),
        "pretrained": config.model.pretrained,
        "pretrained_checkpoint_sha256": (
            _sha256(config.model.pretrained_checkpoint)
            if config.model.pretrained_checkpoint
            else None
        ),
    }
    head_path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(head_path) + ".lock", timeout=60):
        head_reused = head_path.exists()
        if head_reused:
            _verify_receipt(head_path, config.model.name, dataset, head_spec)
            state = _state_dict_from_checkpoint(head_path, "cpu")
            if "weight" not in state or tuple(state["weight"].shape) != (len(class_names), width):
                raise ValueError("existing head dimensions do not match the model and dataset")
            if "bias" in state and tuple(state["bias"].shape) != (len(class_names),):
                raise ValueError("existing head bias dimensions do not match the dataset")
            if not all(torch.isfinite(value).all() for value in state.values()):
                raise ValueError("classification head contains non-finite values")
        else:
            if _receipt(head_path).exists():
                raise FileExistsError(
                    "head metadata already exists: {}".format(_receipt(head_path))
                )
            head = ensure_head(config, runtime, class_names)
            with tempfile.TemporaryDirectory(dir=head_path.parent, prefix=".head-") as directory:
                staged = Path(directory) / "head.pt"
                torch.save(
                    {key: value.detach().cpu() for key, value in head.state_dict().items()}, staged
                )
                _publish(staged, head_path)
            _write_receipt(
                head_path,
                config.model.name,
                dataset,
                "pretrained-text-encoder",
                head_spec=head_spec,
            )
    return {
        "encoder_checkpoint": str(destination),
        "head_checkpoint": str(head_path),
        "encoder_reused": reused,
        "head_reused": head_reused,
        "dataset": dataset,
    }
