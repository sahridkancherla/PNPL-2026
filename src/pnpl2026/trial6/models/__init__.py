"""Model registry for trial 6: config `model:` string -> nn.Module class.

Deliberately separate from every other trial's registry: a trial owns its
architectures, so changing one here cannot alter a finished trial's results.
"""

from __future__ import annotations

from .envelope import (
    BandEnvelope,
    EnvelopeStream,
    FittedProjection,
    fit_envelope_projections,
)
from .frontend import (
    BaselineCorrect,
    ComponentDropout,
    FixedLowpassDecimate,
    GaussianNoise,
    RobustInstanceNorm,
    TimeMask,
)
from .heads import AttentionPool, ClosedSetHead, SemanticHead, SessionAdversary
from .net import TrialSixNet
from .norms import BiasNorm, DropPath, SwooshL, SwooshR
from .transformer import (
    FeedForward,
    FlatTransformer,
    MultiHeadSelfAttention,
    TransformerBlock,
)
from .trunk import (
    DilatedConvTrunk,
    GatedLinearUnit,
    ReceptiveFieldError,
    TrunkLayer,
    dilation_schedule,
    receptive_field,
    resolve_groups,
)

REGISTRY = {
    "trial6_net": TrialSixNet,
}


def build_model(name, **kwargs):
    """Construct a registered model by its config name.

    ARGS:
        name (str): the `model:` string from the run config
        **kwargs: forwarded to the model's __init__

    RETURNS:
        an nn.Module
    """
    #name the known keys, since a typo here otherwise surfaces as a KeyError
    #with no indication of what was expected
    if name not in REGISTRY:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](**kwargs)


__all__ = [
    "AttentionPool",
    "BandEnvelope",
    "BaselineCorrect",
    "BiasNorm",
    "ComponentDropout",
    "ClosedSetHead",
    "DilatedConvTrunk",
    "DropPath",
    "EnvelopeStream",
    "FeedForward",
    "FittedProjection",
    "FixedLowpassDecimate",
    "FlatTransformer",
    "GatedLinearUnit",
    "GaussianNoise",
    "MultiHeadSelfAttention",
    "REGISTRY",
    "ReceptiveFieldError",
    "RobustInstanceNorm",
    "SemanticHead",
    "SessionAdversary",
    "SwooshL",
    "SwooshR",
    "TimeMask",
    "TransformerBlock",
    "TrialSixNet",
    "TrunkLayer",
    "build_model",
    "fit_envelope_projections",
    "dilation_schedule",
    "receptive_field",
    "resolve_groups",
]
