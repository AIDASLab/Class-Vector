"""Dependency import and version checks."""

import importlib
import importlib.metadata
import sys


def check_environment():
    dependencies = {
        "torch": "torch",
        "torchvision": "torchvision",
        "open_clip_torch": "open_clip",
        "PyYAML": "yaml",
        "Pillow": "PIL",
        "numpy": "numpy",
        "scipy": "scipy",
        "transformers": "transformers",
        "datasets": "datasets",
        "gdown": "gdown",
        "filelock": "filelock",
    }
    result = {"python": sys.version.split()[0], "dependencies": {}, "ready": True}
    for distribution, module in dependencies.items():
        try:
            importlib.import_module(module)
            result["dependencies"][distribution] = {
                "version": importlib.metadata.version(distribution),
                "importable": True,
            }
        except Exception as error:
            result["ready"] = False
            result["dependencies"][distribution] = {"importable": False, "error": str(error)}
    return result
