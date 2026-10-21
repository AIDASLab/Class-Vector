"""Optimizer construction for classifier preparation and editing."""

import math

import torch


def make_optimizer(parameters, name, learning_rate, weight_decay=0.0):
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    choices = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
    if name not in choices:
        raise ValueError("optimizer must be adam or adamw")
    return choices[name](parameters, lr=learning_rate, weight_decay=weight_decay)
