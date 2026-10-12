"""Model registry for trial 22: config `model:` string -> nn.Module class.

The continuous family, `trial22_continuous` (`continuous.py`), also under
trials 19, 20 and 21's names so their checkpoints load here. The per-word
family (trials 10-18) lives in `..perword.models`, with its own registry;
prediction and the embedding dump dispatch on the checkpoint's model name
(`..families`).
"""

from __future__ import annotations

from .continuous import N_PHONEME_CLASSES, STEM_KINDS, T5_WIDTH, ContinuousNet, sample_step, step_label_indices, word_grid
from .encoder import AttentionPool, ConvSequence, MaskedBatchNorm1d
from .merger import ChannelMerger, FourierEmbedding, LinearSpatial, SubjectLayer
from .tower import WordTower
from .transformer import SentenceTransformer, attention_mask

REGISTRY = {
    "trial22_continuous": ContinuousNet,
    #trial 21's name: the same class, either stem, so the hybrid B and the
    #distilled students load here
    "trial21_continuous": ContinuousNet,
    #trials 19 and 20's names: the same class with the megstate stem, so
    #their checkpoints load here
    "trial20_continuous": ContinuousNet,
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
    "ConvSequence",
    "MaskedBatchNorm1d",
    "STEM_KINDS",
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
