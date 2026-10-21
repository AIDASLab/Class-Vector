"""Command-line interface for classifier editing applications."""

import argparse
import json
import math
from pathlib import Path

from .config import data_section, load_config, sampling_class, vector_source_config

COMMANDS = (
    "doctor",
    "validate",
    "plan",
    "run",
    "run-suite",
    "convert-checkpoint",
    "prepare-cnn",
    "prepare-head",
    "prepare-clip",
    "extract-bank",
    "compose-edit",
    "map-weights",
    "evaluate",
    "optimize-trigger",
)


def validate(config, check_files=True, evaluation_only=False):
    if not evaluation_only:
        source = vector_source_config(config)
        if source is not config:
            validate(source, check_files=check_files)
    if not config.output_dir:
        raise ValueError("output_dir is required")
    if not config.data:
        raise ValueError("data is required")
    if not config.model.finetuned_checkpoint and not config.model.finetuned_pretrained:
        raise ValueError("set model.finetuned_checkpoint or model.finetuned_pretrained")
    task = config.raw.get("task", "unlearning")
    mode = config.raw.get("mode", "pixels" if task == "trigger_optimization" else "latent")
    if task == "trigger_optimization":
        if mode != "pixels":
            raise ValueError("trigger optimization requires mode: pixels")
    elif mode not in ("latent", "weights"):
        raise ValueError("mode must be latent or weights")
    tasks = (
        "unlearning",
        "environment_adaptation",
        "typography_defense",
        "trigger_optimization",
    )
    if task not in tasks:
        raise ValueError("unsupported task: {}".format(task))
    editing = config.editing
    recipe = editing.get("recipe", "unlearning")
    required = {
        "unlearning": ("target_class",),
        "class_transfer": ("source_class", "destination_class"),
        "concept_removal": ("affected_class", "clean_class"),
        "coefficients": ("coefficients",),
    }
    if recipe not in required:
        raise ValueError("unsupported editing.recipe: {}".format(recipe))
    for key in required[recipe]:
        if editing.get(key) is None:
            raise ValueError("editing.{} is required for {}".format(key, recipe))
    if recipe == "coefficients":
        if not editing["coefficients"]:
            raise ValueError("editing.coefficients must not be empty")
        if not any(
            editing.get(key) is not None for key in ("target_class", "mapping_class", "gate_class")
        ):
            raise ValueError("coefficient edits require target_class, mapping_class, or gate_class")
    if config.raw.get("mode") == "weights" and not editing.get("editable_parameter_patterns"):
        raise ValueError("editing.editable_parameter_patterns is required for weights mode")
    for options in (editing, config.raw.get("preparation", {})):
        if options.get("optimizer", "adam") not in ("adam", "adamw"):
            raise ValueError("optimizer must be adam or adamw")
        rate = float(options.get("learning_rate", 1e-4))
        decay = float(options.get("weight_decay", 0))
        if not math.isfinite(rate) or rate <= 0 or not math.isfinite(decay) or decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if int(options.get("epochs", 1)) < 1:
            raise ValueError("epochs must be positive")
    if editing.get("target_mode", "fixed_mean") not in ("fixed_mean", "per_input"):
        raise ValueError("target_mode must be fixed_mean or per_input")
    for key in ("gated", "train_mode"):
        if not isinstance(editing.get(key, True if key == "gated" else False), bool):
            raise ValueError("editing.{} must be boolean".format(key))
    if task == "trigger_optimization":
        for key in ("source_class", "destination_class", "learning_rate"):
            if editing.get(key) is None:
                raise ValueError("editing.{} is required for trigger optimization".format(key))
    if editing.get("target_classes") and len(set(editing["target_classes"])) != len(
        editing["target_classes"]
    ):
        raise ValueError("editing.target_classes must not contain duplicates")
    for purpose in ("support", "mapping", "evaluation"):
        data = data_section(config, purpose)
        backend = data.get("backend", "registry")
        if backend not in ("registry", "tensor", "imagefolder", "jsonl", "huggingface"):
            raise ValueError("unsupported data.backend: {}".format(backend))
        required = {
            "registry": ("name", "root"),
            "tensor": ("path",),
            "imagefolder": ("root",),
            "jsonl": ("path", "root"),
            "huggingface": (),
        }[backend]
        for key in required:
            if not data.get(key):
                raise ValueError("data.{}.{} is required".format(purpose, key))
        if backend == "huggingface" and not data.get("name") and not data.get("path"):
            raise ValueError("huggingface data requires a name or path")
        if not 0 < float(data.get("fraction", 1.0)) <= 1:
            raise ValueError("data.fraction must be in (0, 1]")
        if data.get("partition", "selected") not in ("selected", "remainder"):
            raise ValueError("data.partition must be selected or remainder")
        if int(data.get("batch_size", 64)) < 1 or int(data.get("num_workers", 0)) < 0:
            raise ValueError("batch_size must be positive and num_workers nonnegative")
        if data.get("holdout_fraction") is not None:
            if not 0 < float(data["holdout_fraction"]) < 1:
                raise ValueError("holdout_fraction must be in (0, 1)")
            if data.get("holdout_partition", "train") not in ("train", "validation"):
                raise ValueError("holdout_partition must be train or validation")
        if not isinstance(data.get("shuffle", False), bool):
            raise ValueError("data.shuffle must be boolean")
        for name, minimum in (("sample_class", 0), ("sample_count", 1), ("samples_per_class", 1)):
            value = sampling_class(config, data) if name == "sample_class" else data.get(name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < minimum
            ):
                raise ValueError("data.{} must be an integer >= {}".format(name, minimum))
    if check_files:
        paths = [
            config.model.pretrained_checkpoint,
            config.model.finetuned_checkpoint,
            config.model.head_checkpoint,
        ]
        if not evaluation_only:
            paths += [config.editing.get("bank_path"), config.editing.get("mask_path")]
        downloadable = {"MNIST", "SVHN", "GTSRB", "CIFAR10", "CIFAR100", "SUN397"}
        purposes = ("evaluation",) if evaluation_only else ("support", "mapping", "evaluation")
        for purpose in purposes:
            section = data_section(config, purpose)
            backend = section.get("backend", "registry")
            paths.append(section.get("path"))
            root_is_cache = backend == "huggingface" or (
                backend == "registry" and section.get("name") in downloadable
            )
            if not root_is_cache:
                paths.append(section.get("root"))
        for value in paths:
            if value and not Path(value).exists():
                raise FileNotFoundError("configured path does not exist: {}".format(value))


def plan(config):
    task = config.raw.get("task", "unlearning")
    steps = [] if config.editing.get("bank_path") else ["extract-bank"]
    if task == "trigger_optimization":
        steps.append("optimize-trigger")
    else:
        steps.append("compose-edit")
        if config.raw.get("mode", "latent") == "weights":
            steps.append("map-weights")
        steps.append("evaluate")
    return {
        "task": task,
        "model": config.model.name,
        "family": config.model.family,
        "mode": config.raw.get("mode", "pixels" if task == "trigger_optimization" else "latent"),
        "steps": steps,
        "output_dir": config.output_dir,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Class Vector classifier editing")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--component", choices=("all", "encoder", "head"), default="all")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--trust-pickle",
        action="store_true",
        help="allow loading a trusted serialized Python model",
    )
    parser.add_argument("--schema-only", action="store_true", help="skip filesystem checks")
    parser.add_argument("--prepare", action="store_true", help="prepare assets before run-suite")
    args = parser.parse_args(argv)
    if args.checkpoint and args.command not in ("convert-checkpoint", "evaluate"):
        parser.error("--checkpoint is only supported by convert-checkpoint and evaluate")
    if args.max_steps is not None and args.command != "prepare-cnn":
        parser.error("--max-steps is only supported by prepare-cnn")
    if args.prepare and (args.command != "run-suite" or args.schema_only):
        parser.error("--prepare requires run-suite without --schema-only")
    if args.command == "doctor":
        from .environment import check_environment

        result = check_environment()
        print(json.dumps(result, indent=2))
        if not result["ready"]:
            raise SystemExit(1)
        return
    if args.command == "convert-checkpoint":
        if not args.checkpoint or not args.output:
            parser.error("convert-checkpoint requires --checkpoint and --output")
        from .checkpoints import convert_checkpoint

        print(convert_checkpoint(args.checkpoint, args.output, args.trust_pickle, args.component))
        return
    if not args.config:
        parser.error("--config is required")
    if args.command == "run-suite":
        import yaml

        path = Path(args.config).expanduser().resolve()
        suite = yaml.safe_load(path.read_text(encoding="utf-8"))
        paths = suite.get("experiments", [])
        if not paths:
            raise ValueError("suite.experiments must list YAML config paths")
        configs = [load_config(path.parent / value) for value in paths]
        for config in configs:
            validate(config, check_files=False)
        if args.prepare:
            from .downloads import prepare_clip
            from .preparation import prepare_cnn

            for config in configs:
                if config.model.family == "clip":
                    prepare_clip(config, trust_pickle=args.trust_pickle)
                elif config.model.family == "vision":
                    checkpoints = (
                        config.model.pretrained_checkpoint,
                        config.model.finetuned_checkpoint,
                    )
                    if not all(value and Path(value).is_file() for value in checkpoints):
                        prepare_cnn(config)
        for config in configs:
            validate(config, check_files=not args.schema_only)
        if args.schema_only:
            print(json.dumps([plan(config) for config in configs], indent=2))
        else:
            from .workflows import Experiment

            for config in configs:
                print(Experiment(config).run())
        return
    config = load_config(args.config)
    if args.command in ("prepare-cnn", "prepare-head", "prepare-clip"):
        validate(config, check_files=False)
        if args.command == "prepare-clip":
            from .downloads import prepare_clip

            print(json.dumps(prepare_clip(config, args.trust_pickle), indent=2))
            return
        from .preparation import prepare_cnn, prepare_head

        result = (
            prepare_cnn(config, args.max_steps)
            if args.command == "prepare-cnn"
            else prepare_head(config)
        )
        print(json.dumps(result, indent=2))
        return
    if args.command in ("validate", "plan"):
        validate(config, check_files=not args.schema_only and args.command == "validate")
        print(json.dumps(plan(config) if args.command == "plan" else {"valid": True}, indent=2))
        return
    validate(config, evaluation_only=args.command == "evaluate")
    from .workflows import Experiment

    experiment = Experiment(config)
    handlers = {
        "run": experiment.run,
        "extract-bank": experiment.extract_bank,
        "compose-edit": experiment.compose_edit,
        "map-weights": experiment.map_weights,
        "evaluate": lambda: experiment.evaluate_saved(args.checkpoint),
        "optimize-trigger": experiment.optimize_trigger,
    }
    print(handlers[args.command]())


if __name__ == "__main__":
    main()
