"""Model registry for trial 19: config `model:` string -> nn.Module class.

One family only: `trial19_continuous` (`continuous.py`). Earlier trials'
per-word networks are not loadable here; their checkpoints are predicted by
their own trials' code.
"""

from __future__ import annotations

from .continuous import N_PHONEME_CLASSES, T5_WIDTH, ContinuousNet, sample_step, step_label_indices
from .encoder import AttentionPool
from .merger import ChannelMerger, FourierEmbedding, LinearSpatial, SubjectLayer
from .tower import WordTower
from .transformer import SentenceTransformer, attention_mask

REGISTRY = {
    "trial19_continuous": ContinuousNet,
}


def build_model(name, **kwargs):
    """Construct a registered model by its config name.

    ARGS:
        name (str): the `model:` string from the run config
        **kwargs: forwarded to the model's __init__

    RETURNS:
        an nn.Module
    """
    #name the known keys; a typo otherwise surfaces as a bare KeyError
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "AttentionPool",
    "ChannelMerger",
    "ContinuousNet",
    "FourierEmbedding",
    "LinearSpatial",
    "N_PHONEME_CLASSES",
    "REGISTRY",
    "SentenceTransformer",
    "SubjectLayer",
    "T5_WIDTH",
    "WordTower",
    "attention_mask",
    "build_model",
    "sample_step",
    "step_label_indices",
]
