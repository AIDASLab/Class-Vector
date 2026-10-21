# Class Vectors

[NeurIPS 2025] Official implementation.

[Exploring and Leveraging Class Vectors for Classifier Editing](https://arxiv.org/abs/2510.11268) · [Project](https://aidaslab.github.io/Class-Vector/)

## Install

Python 3.10. Run commands from the repository root.

```bash
git clone https://github.com/AIDASLab/Class-Vector.git
cd Class-Vector
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-class-vectors.lock
python -m src.class_vectors.cli doctor

export CV_CHECKPOINT_ROOT="$PWD/assets/checkpoints"
export CV_DATA_ROOT="$PWD/assets/datasets"
export CV_OUTPUT_ROOT="$PWD/results"
export OPENCLIP_CACHE_DIR="$PWD/assets/openclip-cache"
export HF_HOME="$PWD/assets/huggingface-cache"
export TORCH_HOME="$PWD/assets/torch-cache"
mkdir -p "$CV_CHECKPOINT_ROOT" "$CV_DATA_ROOT" "$CV_OUTPUT_ROOT" "$OPENCLIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"
```

Repeat the exports in each new shell. Set `model.device: cuda` in YAML for GPU execution; CPU is the default.

## Applications

All configuration paths below are relative to `configs/class_vectors/`.

| Application | Configuration example |
|---|---|
| CLIP class unlearning | `unlearning/vit_b32_mnist_latent.yaml` |
| Permanent weight-space unlearning | `unlearning/vit_b32_mnist_weights.yaml` |
| MNIST vectors → SVHN unlearning | `cross_task/vit_b32_mnist_to_svhn_weights.yaml` |
| Small-patch trigger | `triggers/vit_b32_gtsrb_patch.yaml` |
| Blended full-image trigger | `triggers/vit_b32_gtsrb_blend.yaml` |

Choose a CLIP configuration and run:

```bash
CONFIG=configs/class_vectors/unlearning/vit_b32_mnist_latent.yaml
python -m src.class_vectors.cli prepare-clip --config "$CONFIG" --trust-pickle
python -m src.class_vectors.cli run --config "$CONFIG"
python -m src.class_vectors.cli evaluate --config "$CONFIG"
```

Preparation downloads and converts CLIP weights, prepares downloadable datasets, and builds classification heads. Cross-task preparation also prepares the MNIST source model; vectors are extracted from MNIST and applied to the SVHN model. Existing assets are reused. `--trust-pickle` permits Python deserialization of the official [Task Arithmetic model files](https://github.com/mlfoundations/task_vectors#checkpoints); use only with a trusted source.

Replace `vit_b32` with `vit_b16` or `vit_l14`. Unlearning configurations cover `mnist`, `eurosat`, `gtsrb`, `resisc45`, and `dtd`, in `latent` and `weights` modes. Trigger configurations cover `gtsrb`, `svhn`, and `resisc45`, with `patch` and `blend` modes.

CNN and BERT:

```bash
python -m src.class_vectors.cli prepare-cnn --config configs/class_vectors/cnn_unlearning.yaml
python -m src.class_vectors.cli run --config configs/class_vectors/cnn_unlearning.yaml
python -m src.class_vectors.cli run --config configs/class_vectors/bert_unlearning.yaml
```

CNN alternatives: `cnn_resnet50.yaml`, `cnn_convnext_tiny.yaml`. BERT alternatives: `bert_dbpedia.yaml`, `bert_20newsgroups.yaml`. Append `_weights` before `.yaml` for permanent weight-space edits. Prepare each CNN architecture once; BERT downloads its models and datasets on first use.

CNN/BERT latent edits shift all features without cosine gating. Weight edits fit frozen per-input features plus the edit using AdamW: CNN uses 500 target validation images for 10 epochs; BERT uses 10 target validation texts for 5 epochs. Both evaluate on the full test split.

## Data and settings

Under `CV_DATA_ROOT`, supply local Task Arithmetic-compatible splits for EuroSAT (`EuroSAT_splits/{train,test}/CLASS/`), DTD (`dtd/{train,test}/CLASS/`; `val` is also accepted), and RESISC45 (`resisc45/NWPU-RESISC45/CLASS/` plus `resisc45-{train,test}.txt` in `resisc45/`). MNIST, GTSRB, and SVHN download automatically.

Unlearning runs independently for each `editing.target_classes` entry; CNN/BERT use `editing.target_class`. Keep `editing.class_ids` consistent with selected targets. Triggers use `source_class` and `destination_class`, train on 10% of the test set, and evaluate on its disjoint remainder. Model paths, learning rates, and sample counts are configurable in YAML.

`sample_class` selects a class (`target` follows `editing.target_class`); `fraction` samples within it, and `sample_count` overrides the fraction. `partition: remainder` keeps all unselected examples, including other classes. `samples_per_class` limits vector extraction. `holdout_fraction` and `holdout_partition: train|validation` split a training dataset before sampling.

## Suites and outputs

```bash
python -m src.class_vectors.cli run-suite --config configs/class_vectors/clip_unlearning.yaml --prepare --trust-pickle
```

Other suites: `clip_triggers.yaml`, `cross_task_unlearning.yaml`, `architecture_generalization.yaml`. `--schema-only` checks a suite without preparation or execution.

Results are saved under `CV_OUTPUT_ROOT`. Each `run` requires an empty `output_dir`; choose a new directory to rerun an experiment. `metrics.json` contains run results; `evaluation.json` reevaluates saved edits without retraining. `latent_edit.pt` stores vectors, centroids, and gating settings; evaluation reuses saved editing settings with the configured evaluation data. Independent class runs save weights in `class_ID/mapped_weights.pt`. An explicit single-target checkpoint can be selected with `evaluate --checkpoint PATH`. [Python API](src/class_vectors/README.md).
