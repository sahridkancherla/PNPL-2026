"""Model registry for trial 4: config `model:` string -> nn.Module class.

Deliberately separate from `pnpl2026.models` and from every other trial's
registry: a trial owns its architectures, so changing one here cannot alter a
finished trial's results.
"""

from __future__ import annotations

from .conformer import ConformerBlock, ConvolutionModule, MultiHeadSelfAttention
from .encoder import ConformerStack, MultiRateEncoder
from .frontend import FixedLowpassDecimate, RobustInstanceNorm
from .heads import AttentionPool, ClosedSetHead, SemanticHead, SessionAdversary
from .net import TrialFourNet
from .norms import BiasNorm, DropPath, SwooshL, SwooshR

REGISTRY = {
    "trial4_net": TrialFourNet,
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
    "BiasNorm",
    "ClosedSetHead",
    "ConformerBlock",
    "ConformerStack",
    "ConvolutionModule",
    "DropPath",
    "FixedLowpassDecimate",
    "MultiHeadSelfAttention",
    "MultiRateEncoder",
    "REGISTRY",
    "RobustInstanceNorm",
    "SemanticHead",
    "SessionAdversary",
    "SwooshL",
    "SwooshR",
    "TrialFourNet",
    "build_model",
]
