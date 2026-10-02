"""Model registry for trial 11: config `model:` string -> nn.Module class.

Deliberately separate from every other trial's registry: a trial owns its
architectures, so changing one here cannot alter a finished trial's results.
"""

from __future__ import annotations

from .frontend import (
    BaselineCorrect,
    ComponentDropout,
    FixedLowpassDecimate,
    GaussianNoise,
    RobustInstanceNorm,
)
from .heads import AttentionPool, ClosedSetHead, MeanPool, SemanticHead
from .net import StreamZipformerNet
from .transformer import FlatTransformer, attention_mask
from .trunk import DilatedConvTrunk, ReceptiveFieldError, dilation_schedule, receptive_field
from .zipformer import Bypass, Downsample, ZipformerEncoder, ZipformerStack, upsample

REGISTRY = {
    "trial11_net": StreamZipformerNet,
}


def build_model(name, **kwargs):
    """Construct a registered model by its config name.

    ARGS:
        name (str): the `model:` string from the run config
        **kwargs: forwarded to the model's __init__

    RETURNS:
        an nn.Module
    """
    #name the known keys, since a typo otherwise surfaces as a bare KeyError
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "AttentionPool",
    "Bypass",
    "BaselineCorrect",
    "ClosedSetHead",
    "ComponentDropout",
    "DilatedConvTrunk",
    "Downsample",
    "FixedLowpassDecimate",
    "FlatTransformer",
    "GaussianNoise",
    "MeanPool",
    "REGISTRY",
    "ReceptiveFieldError",
    "RobustInstanceNorm",
    "SemanticHead",
    "StreamZipformerNet",
    "ZipformerEncoder",
    "ZipformerStack",
    "attention_mask",
    "build_model",
    "dilation_schedule",
    "receptive_field",
    "upsample",
]
