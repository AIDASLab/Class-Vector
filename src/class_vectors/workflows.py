"""Application workflows for class-vector extraction, editing, and evaluation."""

import copy
import json
import tempfile
from pathlib import Path

import torch
from filelock import FileLock

from .config import data_section, vector_source_config
from .core import (
    extract_class_vector_bank,
    load_class_vector_bank,
    save_class_vector_bank,
)
from .evaluation import evaluate_encoder
from .model_loader import _state_dict_from_checkpoint
from .recipes import class_transfer_edit, concept_removal_edit, unlearning_edit
from .runtime import device_for, ensure_head, load_runtime, make_loader
from .weight_mapping import train_weight_mapping

TASKS = (
    "unlearning",
    "environment_adaptation",
    "typography_defense",
    "trigger_optimization",
)


def compose(editing, bank):
    recipe = editing.get("recipe", "unlearning")
    if recipe == "unlearning":
        return unlearning_edit(
            bank.vectors,
            int(editing["target_class"]),
            float(editing.get("scale", -1.5)),
        )
    if recipe == "class_transfer":
        source, destination = (
            int(editing["source_class"]),
            int(editing["destination_class"]),
        )
        if source == destination:
            raise ValueError("source_class and destination_class must differ")
        return class_transfer_edit(
            bank.vectors, source, destination, float(editing.get("scale", 1))
        )
    if recipe == "concept_removal":
        affected, clean = int(editing["affected_class"]), int(editing["clean_class"])
        return concept_removal_edit(
            bank.vectors[affected], bank.vectors[clean], float(editing.get("scale", -1))
        )
    if recipe == "coefficients":
        from .core import compose_edit_vector

        return compose_edit_vector(
            bank.vectors,
            {int(key): float(value) for key, value in editing["coefficients"].items()},
        )
    raise ValueError("unsupported editing.recipe: {}".format(recipe))


def target_class(editing):
    for key in ("target_class", "affected_class", "source_class", "mapping_class", "gate_class"):
        if editing.get(key) is not None:
            return int(editing[key])
    raise ValueError("editing requires a target, source, mapping, or gate class")


def summarize_unlearning(results):
    summary = {}
    for key in ("target_accuracy", "retain_accuracy"):
        values = [item["edited"][key] for item in results if item["edited"][key] is not None]
        if values:
            values = torch.tensor(values, dtype=torch.float64)
            summary[key] = {"mean": values.mean().item(), "std": values.std(unbiased=False).item()}
    return {"results": results, "summary": summary}


class Experiment:
    def __init__(self, config):
        self.config = config
        if not config.output_dir:
            raise ValueError("output_dir is required")
        self.output = Path(config.output_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self._runtime = None
        self._bank = None
        self._mapped = None

    @property
    def runtime(self):
        if self._runtime is None:
            self._runtime = load_runtime(self.config)
        return self._runtime

    @property
    def bank(self):
        if self._bank is None:
            path = self.config.editing.get("bank_path", str(self.output / "class_vectors.pt"))
            self._bank = load_class_vector_bank(path)
        return self._bank

    def save_json(self, name, payload):
        path = self.output / name
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return str(path)

    def extract_bank(self):
        source = vector_source_config(self.config)
        runtime = self.runtime if source is self.config else load_runtime(source)
        loader, _ = make_loader(source, runtime, "support")
        self._bank = extract_class_vector_bank(
            runtime.pretrained,
            runtime.finetuned,
            loader,
            runtime.feature_fn,
            device=device_for(self.config),
            class_ids=self.config.editing.get("class_ids"),
            samples_per_class=data_section(source, "support").get("samples_per_class"),
        )
        path = self.output / "class_vectors.pt"
        save_class_vector_bank(self._bank, path)
        return str(path)

    def compose_edit(self, editing=None):
        path = self.output / "edit_vector.pt"
        torch.save(compose(editing or self.config.editing, self.bank).cpu(), path)
        return str(path)

    def map_weights(self, editing=None, save=True):
        editing = editing or self.config.editing
        runtime = self.runtime
        loader, _ = make_loader(self.config, runtime, "mapping")
        patterns = editing.get("editable_parameter_patterns")
        if not patterns:
            raise ValueError("editing.editable_parameter_patterns is required for weight mapping")
        model, target, history, names = train_weight_mapping(
            copy.deepcopy(runtime.finetuned),
            loader,
            runtime.feature_fn,
            int(editing.get("mapping_class", target_class(editing))),
            compose(editing, self.bank),
            patterns,
            n_references=int(editing.get("reference_count", 1)),
            epochs=int(editing.get("epochs", 15)),
            learning_rate=float(editing.get("learning_rate", 1e-4)),
            device=device_for(self.config),
            target_mode=editing.get("target_mode", "fixed_mean"),
            optimizer_name=editing.get("optimizer", "adam"),
            weight_decay=float(editing.get("weight_decay", 0)),
            train_mode=editing.get("train_mode", False),
            reference_model=runtime.finetuned,
        )
        if save:
            self._mapped = model
            path = self.output / "mapped_weights.pt"
            torch.save(
                {key: value.detach().cpu() for key, value in model.state_dict().items()},
                path,
            )
            self.save_json(
                "mapping.json",
                {
                    "loss": history,
                    "trainable_parameters": list(names),
                    "target": target.cpu().tolist() if target is not None else None,
                    "target_mode": editing.get("target_mode", "fixed_mean"),
                },
            )
            return str(path)
        return model

    def latent_edits(self, editing):
        edits = []
        for entry in editing.get("edits", [editing]):
            class_id = int(entry["gate_class"] if "gate_class" in entry else target_class(entry))
            edits.append(
                (
                    compose(entry, self.bank),
                    self.bank.finetuned_centroids[class_id],
                    float(entry.get("cosine_threshold", editing.get("cosine_threshold", 0.5)))
                    if entry.get("gated", editing.get("gated", True))
                    else None,
                )
            )
        return edits

    def evaluate(self, editing=None, model=None, steering=True, edits=None):
        editing = editing or self.config.editing
        runtime = self.runtime
        loader, names = make_loader(self.config, runtime, "evaluation")
        head = ensure_head(self.config, runtime, names)
        if edits is None:
            edits = self.latent_edits(editing) if steering else []
        return evaluate_encoder(
            model if model is not None else runtime.finetuned,
            head,
            loader,
            runtime.feature_fn,
            edits=edits,
            device=device_for(self.config),
            target_class=target_class(editing),
            destination_class=editing.get("destination_class"),
        )

    def evaluate_mapping(self, editing, model):
        original = self.evaluate(editing, model=self.runtime.finetuned, steering=False)
        mapped = self.evaluate(editing, model=model, steering=False)
        return {"clean": original["clean"], "edited": mapped["edited"], "gate_coverage": None}

    def save_latent_edit(self, editing, edits):
        payload = {
            "version": 1,
            "target_class": target_class(editing),
            "destination_class": editing.get("destination_class"),
            "edits": [
                {
                    "vector": vector.detach().cpu(),
                    "centroid": centroid.detach().cpu(),
                    "threshold": threshold,
                }
                for vector, centroid, threshold in edits
            ],
        }
        torch.save(payload, self.output / "latent_edit.pt")

    def evaluate_latent_payload(self, payload):
        if (
            not isinstance(payload, dict)
            or payload.get("version") != 1
            or not isinstance(payload.get("edits"), list)
        ):
            raise ValueError("unsupported latent edit file")
        editing = {
            "target_class": int(payload["target_class"]),
            "destination_class": payload["destination_class"],
        }
        edits = [
            (entry["vector"], entry["centroid"], entry["threshold"]) for entry in payload["edits"]
        ]
        return self.evaluate(editing, edits=edits)

    def evaluate_saved(self, checkpoint=None):
        """Evaluate saved edits without retraining or replacing training metrics."""
        mode = self.config.raw.get("mode", "latent")
        editing = self.config.editing
        saved_config = self.output / "config.json"
        snapshot = None
        if saved_config.is_file() and not checkpoint:
            snapshot = json.loads(saved_config.read_text(encoding="utf-8"))
            if snapshot.get("mode", "latent") != mode or snapshot.get(
                "task", "unlearning"
            ) != self.config.raw.get("task", "unlearning"):
                raise ValueError("task and mode must match the saved run configuration")
            editing = snapshot["editing"]
        if self.config.raw.get("task") == "trigger_optimization":
            path = Path(checkpoint) if checkpoint else self.output / "trigger_pattern.pt"
            payload = torch.load(path, map_location="cpu", weights_only=True)
            metrics = self.evaluate_trigger_pattern(payload)
        else:
            targets = editing.get("target_classes")
            if checkpoint and mode != "weights":
                raise ValueError("--checkpoint requires weights mode or a trigger configuration")
            if checkpoint and targets and len(targets) > 1:
                raise ValueError("an explicit checkpoint requires a single target class")

            def evaluate_one(editing, directory):
                if mode == "weights":
                    path = Path(checkpoint) if checkpoint else directory / "mapped_weights.pt"
                    if not path.is_file():
                        raise FileNotFoundError("edited checkpoint does not exist: {}".format(path))
                    model = copy.deepcopy(self.runtime.finetuned)
                    model.load_state_dict(_state_dict_from_checkpoint(path, "cpu"), strict=True)
                    result = self.evaluate_mapping(editing, model)
                else:
                    path = directory / "latent_edit.pt"
                    if path.is_file():
                        payload = torch.load(path, map_location="cpu", weights_only=True)
                        if int(payload["target_class"]) != target_class(editing):
                            raise ValueError("saved latent edit target does not match the run")
                        result = self.evaluate_latent_payload(payload)
                    else:
                        if snapshot is None:
                            raise FileNotFoundError(
                                "saved latent edit does not exist: {}".format(path)
                            )
                        bank_path = (
                            snapshot["editing"].get("bank_path") or self.output / "class_vectors.pt"
                        )
                        self._bank = load_class_vector_bank(bank_path)
                        result = self.evaluate(editing)
                return dict(result, mode=mode)

            if targets:
                results = []
                for class_id in targets:
                    selected_editing = dict(editing, target_class=int(class_id))
                    directory = self.output / "class_{}".format(int(class_id))
                    results.append(
                        dict(evaluate_one(selected_editing, directory), target_class=int(class_id))
                    )
                metrics = summarize_unlearning(results)
            else:
                metrics = evaluate_one(editing, self.output)
        return self.save_json("evaluation.json", metrics)

    def run_edit(self):
        if self.config.editing.get("bank_path"):
            _ = self.bank
        else:
            self.extract_bank()
        targets = self.config.editing.get("target_classes")
        if targets:
            if self.config.editing.get("recipe", "unlearning") != "unlearning":
                raise ValueError("target_classes requires the unlearning recipe")
            root = self.output
            results = []
            try:
                for class_id in targets:
                    self.output = root / ("class_{}".format(int(class_id)))
                    self.output.mkdir(parents=True, exist_ok=True)
                    editing = dict(self.config.editing, target_class=int(class_id))
                    results.append({"target_class": int(class_id), **self.run_single_edit(editing)})
            finally:
                self.output = root
            return self.save_json("metrics.json", summarize_unlearning(results))
        self.run_single_edit(self.config.editing)
        return str(self.output / "metrics.json")

    def run_single_edit(self, editing):
        self.compose_edit(editing)
        mode = self.config.raw.get("mode", "latent")
        if mode not in ("latent", "weights"):
            raise ValueError("mode must be latent or weights")
        if mode == "weights":
            self.map_weights(editing)
            metrics = self.evaluate_mapping(editing, self._mapped)
        else:
            edits = self.latent_edits(editing)
            self.save_latent_edit(editing, edits)
            metrics = self.evaluate(editing, edits=edits)
        metrics["mode"] = mode
        self.save_json("metrics.json", metrics)
        return metrics

    def optimize_trigger(self):
        from .core import _default_batch_to_xy
        from .trigger import apply_trigger, optimize_trigger_loader
        from .weight_mapping import collect_reference_features

        if not self.config.editing.get("bank_path"):
            self.extract_bank()
        runtime, editing = self.runtime, self.config.editing
        source = int(editing["source_class"])
        edit = compose(dict(editing, recipe="class_transfer"), self.bank)
        loader, _ = make_loader(self.config, runtime, "mapping")
        device = device_for(self.config)
        first = next(iter(loader), None)
        if first is None:
            raise ValueError("trigger data loader produced no images")
        images, _ = _default_batch_to_xy(first)
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("trigger optimization requires image tensors with shape [N, C, H, W]")
        if editing.get("mask_path"):
            mask = torch.load(editing["mask_path"], map_location="cpu", weights_only=True)
        else:
            mask = torch.ones_like(images[0])
            if editing.get("trigger_mode", "patch") == "patch":
                size = int(editing.get("patch_size", 20))
                if size < 1 or size > min(images.shape[-2:]):
                    raise ValueError("editing.patch_size must fit inside the image")
                mask.zero_()
                mask[:, -size:, -size:] = 1
        initial = torch.zeros_like(images[:1], device=device)
        mask = mask.to(device=device, dtype=initial.dtype)
        if tuple(torch.broadcast_shapes(initial.shape, mask.shape)) != tuple(initial.shape):
            raise ValueError("trigger mask must broadcast to one image")
        if not torch.isfinite(mask).all() or not ((mask >= 0) & (mask <= 1)).all():
            raise ValueError("trigger mask values must be in [0, 1]")
        mode, transparency = (
            editing.get("trigger_mode", "patch"),
            float(editing.get("transparency", 1)),
        )
        references = collect_reference_features(
            runtime.finetuned,
            loader,
            lambda model, x: runtime.feature_fn(
                model, apply_trigger(x, initial, mask, mode, transparency)
            ),
            source,
            int(editing.get("reference_count", 1)),
            device=device,
        )
        target = references.mean(0) + edit.to(device)
        trigger, losses = optimize_trigger_loader(
            runtime.finetuned,
            loader,
            target,
            initial,
            mask,
            runtime.feature_fn,
            source_class=source,
            epochs=int(editing.get("epochs", 1)),
            max_steps=editing.get("steps"),
            learning_rate=float(editing["learning_rate"]),
            lower_bound=editing.get("lower_bound"),
            upper_bound=editing.get("upper_bound"),
            mode=editing.get("trigger_mode", "patch"),
            transparency=float(editing.get("transparency", 1)),
            device=device,
        )
        path = self.output / "trigger_pattern.pt"
        payload = {
            "trigger": trigger,
            "pattern_format": "raw",
            "mask": mask.cpu(),
            "losses": losses,
            "target": target.detach().cpu(),
            "mode": editing.get("trigger_mode", "patch"),
            "transparency": float(editing.get("transparency", 1)),
            "source_class": source,
            "destination_class": int(editing["destination_class"]),
        }
        torch.save(payload, path)
        self.save_json("metrics.json", self.evaluate_trigger_pattern(payload))
        return str(path)

    def evaluate_trigger_pattern(self, payload):
        from .evaluation import evaluate_trigger

        runtime, editing = self.runtime, self.config.editing
        evaluation_loader, names = make_loader(self.config, runtime, "evaluation")
        pattern, mask = payload["trigger"], payload["mask"]
        pattern_format = payload.get("pattern_format", "masked")
        if pattern_format == "masked":
            pattern = pattern / torch.where(mask > 0, mask, torch.ones_like(mask))
        elif pattern_format != "raw":
            raise ValueError("unsupported trigger pattern format")
        return evaluate_trigger(
            runtime.finetuned,
            ensure_head(self.config, runtime, names),
            evaluation_loader,
            runtime.feature_fn,
            pattern,
            mask,
            source_class=int(payload.get("source_class", editing["source_class"])),
            destination_class=int(payload.get("destination_class", editing["destination_class"])),
            mode=payload["mode"],
            transparency=float(payload["transparency"]),
            device=device_for(self.config),
        )

    def run(self):
        task = self.config.raw.get("task", "unlearning")
        if task not in TASKS:
            raise ValueError("unsupported task: {}".format(task))
        handlers = {
            "unlearning": self.run_edit,
            "environment_adaptation": self.run_edit,
            "typography_defense": self.run_edit,
            "trigger_optimization": self.optimize_trigger,
        }
        destination = self.output
        with FileLock(str(destination) + ".lock", timeout=0):
            if any(destination.iterdir()):
                raise FileExistsError(
                    "output_dir is not empty; choose a new output_dir: {}".format(destination)
                )
            with tempfile.TemporaryDirectory(
                dir=destination.parent, prefix=".{}-run-".format(destination.name)
            ) as directory:
                self.output = Path(directory)
                try:
                    torch.manual_seed(int(self.config.raw.get("seed", 0)))
                    self.save_json("config.json", self.config.raw)
                    result = Path(handlers[task]()).relative_to(self.output)
                    destination.rmdir()
                    try:
                        self.output.rename(destination)
                    except OSError:
                        destination.mkdir(exist_ok=True)
                        raise
                except BaseException:
                    self._bank = self._mapped = None
                    raise
                finally:
                    self.output = destination
        return str(destination / result)
