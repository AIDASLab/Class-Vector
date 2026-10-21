"""Editing-vector arithmetic."""

from .core import compose_edit_vector


def unlearning_edit(class_vectors, target_class, scale=-1.5):
    return compose_edit_vector(class_vectors, {target_class: scale})


def class_transfer_edit(class_vectors, source_class, destination_class, scale=1.0):
    return compose_edit_vector(class_vectors, {source_class: -scale, destination_class: scale})


def concept_removal_edit(affected_vector, clean_vector, scale=-1.0):
    """Remove a nuisance concept: lambda * (kappa_affected - kappa_clean)."""
    return float(scale) * (affected_vector - clean_vector)


def trigger_target_edit(class_vectors, source_class, destination_class, scale=1.5):
    """Latent target for trigger optimization: lambda * (kappa_dst-kappa_src)."""
    return class_transfer_edit(class_vectors, source_class, destination_class, scale)
