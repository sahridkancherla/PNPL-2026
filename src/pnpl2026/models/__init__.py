"""Model registry: config `model:` string -> nn.Module class."""

from __future__ import annotations

from .cnn_baseline import SimpleWordCNN

REGISTRY = {
    "cnn_baseline": SimpleWordCNN,
}


def build_model(name: str, **kwargs):
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)
