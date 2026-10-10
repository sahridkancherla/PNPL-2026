"""Model registry for trial 20: config `model:` string -> nn.Module class.

One family: `trial20_continuous` (`continuous.py`), also under trial 19's
name, so trial 19's checkpoints load here (with `word_depth` 0 the two are
the same network). Earlier per-word networks are predicted by their own
trials' code.
"""

from __future__ import annotations

from .continuous import N_PHONEME_CLASSES, T5_WIDTH, ContinuousNet, sample_step, step_label_indices, word_grid
from .encoder import AttentionPool
from .merger import ChannelMerger, FourierEmbedding, LinearSpatial, SubjectLayer
from .tower import WordTower
from .transformer import SentenceTransformer, attention_mask

REGISTRY = {
    "trial20_continuous": ContinuousNet,
    #trial 19's name: the same class, word transformer off
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
    "word_grid",
]
