"""YAML configuration for classifier editing."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    pretrained: Optional[str] = "openai"
    pretrained_checkpoint: Optional[str] = None
    finetuned_checkpoint: Optional[str] = None
    cache_dir: Optional[str] = None
    device: str = "auto"
    family: str = "clip"
    finetuned_pretrained: Optional[str] = None
    num_classes: Optional[int] = None
    local_files_only: bool = True
    head_checkpoint: Optional[str] = None
    normalize_features: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    data: Dict[str, Any]
    editing: Dict[str, Any]
    output_dir: Optional[str]
    raw: Dict[str, Any]


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_environment(value):
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):

        def replace(match):
            name = match.group(1)
            if name not in os.environ:
                raise ValueError("environment variable {} is required by the config".format(name))
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    return value


def _resolve_path(value, config_dir):
    if value is None:
        return None
    expanded = os.path.expanduser(str(value))

    def replace(match):
        name = match.group(1)
        if name not in os.environ:
            raise ValueError("environment variable {} is required by the config".format(name))
        return os.environ[name]

    expanded = _ENV_PATTERN.sub(replace, expanded)
    path = Path(expanded)
    return str(path if path.is_absolute() else (config_dir / path).resolve())


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def data_section(config, purpose):
    values = {
        key: value
        for key, value in config.data.items()
        if key not in ("support", "mapping", "evaluation")
    }
    if purpose == "mapping":
        values.update(config.data.get("support", {}))
    values.update(config.data.get(purpose, {}))
    values.setdefault("split", "test" if purpose == "evaluation" else "train")
    return values


def sampling_class(config, spec):
    value = spec.get("sample_class")
    if value != "target":
        return value
    if config.editing.get("target_classes"):
        raise ValueError("sample_class: target requires a single editing.target_class")
    for key in ("mapping_class", "target_class", "source_class"):
        if config.editing.get(key) is not None:
            return int(config.editing[key])
    raise ValueError("sample_class: target requires a mapping, target, or source class")


def _load_yaml(path, ancestors=()):
    if path in ancestors:
        raise ValueError("configuration inheritance cycle: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    parent = raw.pop("extends", None)
    if parent:
        parent_path = Path(_resolve_path(parent, path.parent))
        raw = _merge(_load_yaml(parent_path, (*ancestors, path)), raw)
    return raw


def load_config(path) -> ExperimentConfig:
    """Resolve paths against the selected YAML file, including inherited values."""
    path = Path(path).expanduser().resolve()
    raw = _expand_environment(_load_yaml(path))
    model_data = dict(raw.get("model") or {})
    if not model_data.get("name"):
        raise ValueError("config must set model.name (e.g. ViT-B-16)")
    for key in (
        "pretrained_checkpoint",
        "finetuned_checkpoint",
        "cache_dir",
        "head_checkpoint",
    ):
        if key in model_data:
            model_data[key] = _resolve_path(model_data[key], path.parent)
    model_config = ModelConfig(**model_data)
    if model_config.family not in ("clip", "vision", "bert"):
        raise ValueError("model.family must be clip, vision, or bert")
    data = dict(raw.get("data") or {})
    data_paths = ("root", "dataset_root", "path", "support_path", "evaluation_path")
    for key in data_paths:
        if key in data:
            data[key] = _resolve_path(data[key], path.parent)
    for section in ("support", "mapping", "evaluation"):
        if section in data:
            data[section] = dict(data[section])
            for key in data_paths:
                if key in data[section]:
                    data[section][key] = _resolve_path(data[section][key], path.parent)
    editing = dict(raw.get("editing") or {})
    for key in ("bank_path", "mask_path"):
        if key in editing:
            editing[key] = _resolve_path(editing[key], path.parent)
    output_dir = _resolve_path(raw.get("output_dir"), path.parent)
    raw = dict(raw, model=model_data, data=data, editing=editing, output_dir=output_dir)
    if raw.get("vector_source"):
        raw["vector_source"] = _resolve_path(raw["vector_source"], path.parent)
    return ExperimentConfig(
        model=model_config,
        data=data,
        editing=editing,
        output_dir=output_dir,
        raw=raw,
    )


def vector_source_config(config):
    """Resolve an optional encoder/data configuration for cross-task vectors."""
    path = config.raw.get("vector_source")
    if not path:
        return config
    source = load_config(path)
    if source.raw.get("vector_source"):
        raise ValueError("vector_source must not reference another vector_source")
    if (source.model.family, source.model.name) != (config.model.family, config.model.name):
        raise ValueError("vector_source and destination must use the same encoder architecture")
    return source
