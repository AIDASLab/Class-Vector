# Class Vector API

The [repository README](../../README.md) describes configuration, commands, checkpoints, and datasets.

| Module | API |
|---|---|
| `core.py` | Class centroids, vector banks, paired feature extraction, cosine-gated steering, bank serialization |
| `config.py` | YAML inheritance, environment expansion, path resolution |
| `model_loader.py` | OpenCLIP encoder pairs and state dictionaries |
| `runtime.py` | CLIP/CNN/BERT runtimes, data loaders, classification heads |
| `recipes.py` | Unlearning, class transfer, concept removal, trigger targets |
| `weight_mapping.py` | Fixed-reference or per-input MSE mapping and editable parameter selection |
| `evaluation.py` | Clean/edited accuracy, per-class metrics, retained-class accuracy, gate coverage |
| `trigger.py` | Patch, blended-pattern, and additive-noise optimization |
| `architecture_adapters.py` | ResNet18/50, ConvNeXt-Tiny, BERT pooled CLS adapters |
| `unlearning.py` | Shared feature extraction and latent unlearning metrics |
| `workflows.py` | Task orchestration and artifact output |
| `checkpoints.py` | Conversion to tensor-only state dictionaries |
| `preparation.py` | CNN fine-tuning and pretrained CLIP classification heads |
| `downloads.py` | Automatic CLIP task checkpoint preparation and checksum receipts |
| `cli.py` | Command-line entry point |

```python
from src.class_vectors import load_config
from src.class_vectors.workflows import Experiment

config = load_config("configs/class_vectors/vit_b32_example.yaml")
metrics_path = Experiment(config).run()
```
