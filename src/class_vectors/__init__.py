"""Class Vector extraction, classifier editing, and evaluation."""

from importlib import import_module

_MODULES = {
    "core": (
        "ClassVectorBank",
        "compute_class_vector_bank",
        "extract_class_vector_bank",
        "compose_edit_vector",
        "extract_paired_features",
        "extract_labeled_features",
        "latent_steer",
        "save_class_vector_bank",
        "load_class_vector_bank",
        "LatentSteeringClassifier",
    ),
    "config": ("ExperimentConfig", "ModelConfig", "load_config"),
    "model_loader": ("EncoderPair", "load_encoder_pair"),
    "weight_mapping": (
        "build_mapping_target",
        "configure_trainable_parameters",
        "train_weight_mapping",
    ),
    "recipes": (
        "class_transfer_edit",
        "concept_removal_edit",
        "trigger_target_edit",
        "unlearning_edit",
    ),
    "trigger": ("optimize_shared_trigger",),
    "architecture_adapters": (
        "BertClassifierPair",
        "ImageClassifier",
        "VisionFeatureExtractor",
        "bert_classifier_head",
        "bert_feature_fn",
        "build_image_classifier",
        "forget_preserve_accuracy",
        "load_bert_classifier_pair",
        "vision_feature_fn",
    ),
    "unlearning": (
        "evaluate_latent_unlearning",
        "extract_architecture_class_vectors",
    ),
    "evaluation": ("evaluate_encoder", "classification_metrics"),
}
_EXPORTS = {name: module for module, names in _MODULES.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
