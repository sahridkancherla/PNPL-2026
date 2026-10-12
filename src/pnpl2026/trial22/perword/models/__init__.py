"""Model registry for trial 18: config `model:` string -> nn.Module class.

Separate from every other trial's registry: a trial owns its architectures.
Earlier trials' names (trial 10 to 17) are the same class, so their
checkpoints -- whose saved configs name them -- load here, for scoring and as
the frozen base of Variant M's refinement. Their model_kwargs carry no
`input_norm`, so they run with "none", exactly as they were trained.
"""

from __future__ import annotations

from .encoder import INPUT_NORMS, SPATIAL_KINDS, AttentionPool, ConvSequence, MaskedBatchNorm1d, WordEncoder
from .merger import ChannelMerger, FixedPCA, FourierEmbedding, LinearSpatial, SubjectLayer
from .net import T5_WIDTH, ReferenceNet
from .refine import PyramidRefiner
from .transformer import SentenceTransformer, attention_mask

REGISTRY = {
    "trial18_reference": ReferenceNet,
    #aliases, for checkpoints saved by trials 10, 12, 13, 15, 16 and 17
    "trial17_reference": ReferenceNet,
    "trial16_reference": ReferenceNet,
    "trial15_reference": ReferenceNet,
    "trial13_reference": ReferenceNet,
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
    "INPUT_NORMS",
    "LinearSpatial",
    "MaskedBatchNorm1d",
    "PyramidRefiner",
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
