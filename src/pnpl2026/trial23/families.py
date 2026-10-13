"""Which family a checkpoint belongs to, and one loader for both.

Trial 21 drives two networks with one distillation loop and one predictor:

    perword      trials 10-18's `ReferenceNet`: one 3 s window per word, a
                 conv encoder, a transformer across the words of a sentence.
                 Its code is a copy of trial 18's under `perword/`
    continuous   trials 19-21's `ContinuousNet`: the item read once, every
                 word read out at its onset. This trial's own code

A checkpoint names its network in `hyper_parameters["model"]`; the two
registries do not overlap, so that name decides everything downstream.
"""

from __future__ import annotations

import torch

from . import lit as continuous_lit
from .models import REGISTRY as CONTINUOUS_MODELS
from .perword import lit as perword_lit
from .perword.models import REGISTRY as PERWORD_MODELS

#the two families, by the name every tool uses
FAMILIES = ("perword", "continuous")


def checkpoint_model_name(checkpoint_path):
    """The `model:` string a checkpoint was trained under.

    ARGS:
        checkpoint_path (Path or str): a Lightning best.ckpt or last.ckpt

    RETURNS:
        str
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return str(checkpoint["hyper_parameters"]["model"])


def family_of(model_name):
    """Which family a model name belongs to.

    ARGS:
        model_name (str): e.g. "trial13_reference" or "trial19_continuous"

    RETURNS:
        str: "perword" or "continuous"
    """
    if model_name in PERWORD_MODELS:
        return "perword"
    if model_name in CONTINUOUS_MODELS:
        return "continuous"
    raise KeyError(
        f"Unknown model {model_name!r}; per-word names are {sorted(PERWORD_MODELS)}, "
        f"continuous names {sorted(CONTINUOUS_MODELS)}."
    )


def load_any(checkpoint_path, device="cpu", use_ema=True):
    """Rebuild a trained module of either family from its checkpoint alone.

    ARGS:
        checkpoint_path (Path or str)
        device (str or torch.device)
        use_ema (bool): swap in averaged weights when the checkpoint has them

    RETURNS:
        module : LitTrialEighteen or LitTrialTwentyThree, in eval mode, on `device`
        cfg : the run's config, from the checkpoint
        info : dict with the checkpoint's epoch and global_step
        family : "perword" or "continuous"
    """
    family = family_of(checkpoint_model_name(checkpoint_path))
    loader = perword_lit.load_trained if family == "perword" else continuous_lit.load_trained
    module, cfg, info = loader(checkpoint_path, device, use_ema=use_ema)
    return module, cfg, info, family


def lit_module_for(family):
    """The family's training module file, for `adapt_batch_norm` and friends.

    ARGS:
        family (str): "perword" or "continuous"

    RETURNS:
        module: `perword.lit` or `lit`
    """
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    return perword_lit if family == "perword" else continuous_lit
