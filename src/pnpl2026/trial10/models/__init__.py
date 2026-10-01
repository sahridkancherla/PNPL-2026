"""Model registry for trial 10: config `model:` string -> nn.Module class.

Separate from every other trial's registry: a trial owns its architectures.
"""

from __future__ import annotations

from .encoder import SPATIAL_KINDS, AttentionPool, ConvSequence, WordEncoder
from .merger import ChannelMerger, FixedPCA, FourierEmbedding, LinearSpatial, SubjectLayer
from .net import T5_WIDTH, ReferenceNet
from .transformer import SentenceTransformer, attention_mask

REGISTRY = {
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
