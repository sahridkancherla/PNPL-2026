"""Model registry for trial 1: config `model:` string -> nn.Module class.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.

Deliberately separate from `pnpl2026.models`: a trial owns its architectures so
that changing one here cannot alter a finished trial's results.
"""

from __future__ import annotations

from .block import EncoderBlock, QuadratureBranch
from .net import TrialOneNet

REGISTRY = {
    "trial1_net": TrialOneNet,
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


__all__ = ["EncoderBlock", "QuadratureBranch", "TrialOneNet", "REGISTRY", "build_model"]
