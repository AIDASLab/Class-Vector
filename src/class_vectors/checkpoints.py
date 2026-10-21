"""Checkpoint conversion to tensor-only state dictionaries."""

import pickle
import types
from pathlib import Path

import torch


class _ModuleState(torch.nn.Module):
    """Restore registered parameters and submodules from serialized model wrappers."""


def _checkpoint_rgb(image):
    return image.convert("RGB")


class _WrapperUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "open_clip.model" and name in (
            "CLIP",
            "VisualTransformer",
            "VisionTransformer",
            "LayerNorm",
            "QuickGELU",
            "ResidualAttentionBlock",
            "Transformer",
        ):
            return _ModuleState
        if module == "open_clip.transform" and name == "_convert_to_rgb":
            return _checkpoint_rgb
        if module in ("modeling", "src.modeling") and name in (
            "ImageEncoder",
            "ImageClassifier",
            "ClassificationHead",
        ):
            return _ModuleState
        return super().find_class(module, name)


def _trusted_pickle_module():
    module = types.ModuleType("class_vector_pickle")
    module.Unpickler = _WrapperUnpickler
    module.load = pickle.load
    return module


def _select_component(value, component):
    if component == "all":
        return value
    names = (
        ("image_encoder", "encoder")
        if component == "encoder"
        else ("classification_head", "classifier")
    )
    for name in names:
        candidate = getattr(value, name, None)
        if isinstance(candidate, torch.nn.Module):
            return candidate
    if component == "head" and isinstance(value, torch.nn.Module):
        state = value.state_dict()
        if "weight" not in state or state["weight"].ndim != 2:
            raise ValueError("checkpoint has no standalone linear classification head")
    return value


def _strip_prefixes(state):
    while state:
        prefix = next(
            (item for item in ("module.", "model.") if all(key.startswith(item) for key in state)),
            None,
        )
        if prefix is None:
            break
        state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def convert_checkpoint(source, destination, trust_pickle=False, component="all"):
    if component not in ("all", "encoder", "head"):
        raise ValueError("component must be all, encoder, or head")
    source, destination = Path(source).expanduser(), Path(destination).expanduser()
    if source.resolve() == destination.resolve():
        raise ValueError("checkpoint destination must differ from source")
    if destination.exists():
        raise FileExistsError("checkpoint destination already exists: {}".format(destination))
    options = {"pickle_module": _trusted_pickle_module()} if trust_pickle else {}
    value = torch.load(source, map_location="cpu", weights_only=not trust_pickle, **options)
    value = _select_component(value, component)
    if isinstance(value, torch.nn.Module):
        value = value.state_dict()
    if isinstance(value, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in value:
                candidate = value[key]
                if isinstance(candidate, torch.nn.Module):
                    candidate = candidate.state_dict()
                if isinstance(candidate, dict):
                    value = candidate
                    break
    if (
        not isinstance(value, dict)
        or not value
        or not all(torch.is_tensor(item) for item in value.values())
    ):
        raise ValueError("checkpoint must contain a model or tensor state dictionary")
    value = _strip_prefixes(value)
    if component != "all":
        wrappers = (
            ("image_encoder.", "encoder.")
            if component == "encoder"
            else ("classification_head.", "classifier.")
        )
        for wrapper in wrappers:
            selected = {
                key[len(wrapper) :]: item for key, item in value.items() if key.startswith(wrapper)
            }
            if selected:
                value = selected
                break
    value = _strip_prefixes(value)
    if component == "head" and ("weight" not in value or value["weight"].ndim != 2):
        raise ValueError("checkpoint has no standalone linear classification head")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({key: item.detach().cpu() for key, item in value.items()}, destination)
    return str(destination)
