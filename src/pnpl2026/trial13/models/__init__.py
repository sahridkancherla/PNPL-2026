"""Model registry for trial 13: config `model:` string -> nn.Module class.

Separate from every other trial's registry: a trial owns its architectures.
`trial10_reference` is the same class under trial 10's name, so trial 10 and
12 checkpoints -- whose saved config names it -- load here for scoring.
"""

from __future__ import annotations

from .encoder import SPATIAL_KINDS, AttentionPool, ConvSequence, MaskedBatchNorm1d, WordEncoder
from .merger import ChannelMerger, FixedPCA, FourierEmbedding, LinearSpatial, SubjectLayer
from .net import T5_WIDTH, ReferenceNet
from .transformer import SentenceTransformer, attention_mask

REGISTRY = {
    "trial13_reference": ReferenceNet,
    #an alias, for checkpoints saved by trial 10 and 12
    "trial10_reference": ReferenceNet,
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
    "ConvSequence",
    "FixedPCA",
    "FourierEmbedding",
    "LinearSpatial",
    "MaskedBatchNorm1d",
    "REGISTRY",
    "ReferenceNet",
    "SPATIAL_KINDS",
    "SentenceTransformer",
    "SubjectLayer",
    "T5_WIDTH",
    "WordEncoder",
    "attention_mask",
    "build_model",
]
