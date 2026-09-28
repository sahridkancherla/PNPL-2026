"""Trial-6 run config: the shared fields, plus what only this trial has.

Subclasses the shared `Config` so `lr`, `batch_size` and the rest keep one
definition and one coercion path. Only the fields trial 6 introduces live here.

⚠️ Trial 6 reads trial 1's cache (73 whitened components, 400-sample windows),
exactly as trial 5 did. Every Variant C run is a change against trial 5's
network, so the input has to stay the same for the comparison to mean anything.

⚠️ `load_trial6_config` merges `model_kwargs` key by key across files. The
shared `load_config` replaces the whole dict, so an ablation file carrying only
`model_kwargs: {depth: 2}` would silently drop every other architecture setting
and fall back to the constructor's defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_type_hints

import yaml

from ..config import REPO_ROOT, Config, _coerce


@dataclass
class Trial6Config(Config):
    """Everything trial 6 needs that the shared config does not carry.

    ATTRS:
        cache_dir (str): shard directory, RELATIVE to the repo root
        n_components (int): whitener rank the cache was built at
        window_bounds (list): cached window in seconds, wider than the scored one
        clip_sigma (float): clip threshold the builder applied before fp16
        jitter_seconds (float): one-sided time jitter, training only
        class_weight_power (float): 0 unweighted, 0.5 sqrt-inverse, 1 inverse
        word_embeddings (str): .npz of target word embeddings, repo-relative
        centre_embeddings (bool): subtract the mean target row before training
        semantic_weight (float): weight on the D-SigLIP loss. 0 disables it
        closed_set_weight (float): weight on the 50-way cross-entropy
        session_adv_weight (float): weight on the session-adversarial loss
        session_adv_warmup_epochs (int): epochs to ramp the adversary in over
        inference_semantic_weight (float): how much the semantic head
            contributes to the scored logits, 0 = closed-set only, 1 = semantic
        label_smoothing (float): on the closed-set head only
        siglip_sum_over_candidates (bool): False is trial 5's mean over every
            kept pair; True is SigLIP's own sum over candidates, mean over rows
        warmup_fraction (float): fraction of total steps spent warming up
        min_lr (float): floor the cosine schedule decays to
        ema_decay (float): 0 disables the weight EMA
        time_mask_spans (int): how many spans of time to zero per window
        time_mask_width (int): maximum width of one span, in input samples
        component_dropout (float): probability of zeroing a whole component
        gaussian_noise (float): std of additive noise after normalisation,
            training only. 0 disables it
        shuffle_labels (bool): permute the TRAINING labels -- the chance line
        envelope_pca_windows (int): training windows the envelope PCA is
            fitted on. Unused when no envelope band is configured
        accelerator (str): "auto" picks MPS on Apple Silicon; "cpu" forces CPU
        torch_threads (int): 0 leaves torch's default
        duty_cycle (float): fraction of wall-clock spent computing
        cooldown_every_minutes (float): 0 disables the periodic long pause
        cooldown_seconds (float): length of that pause
    """

    cache_dir: str = "data/trial1_cache"
    n_components: int = 73

    #wider than the scored (0, 1) s window; the margin is what jitter uses
    window_bounds: list = field(default_factory=lambda: [-0.32, 1.28])
    clip_sigma: float = 20.0
    jitter_seconds: float = 0.1

    #full inverse frequency spans the vocabulary's 121x and destabilises the
    #batch gradient; the square root keeps the direction, not the magnitude
    class_weight_power: float = 0.5

    # --- the three heads ---
    word_embeddings: str = "data/word_embeddings_t5.npz"

    #⚠️ on by default: t5's middle layer puts these 50 words at a mean pairwise
    #cosine of 0.709, and centring is what makes the contrastive loss separable
    centre_embeddings: bool = True
    semantic_weight: float = 1.0
    closed_set_weight: float = 0.3
    session_adv_weight: float = 0.1
    session_adv_warmup_epochs: int = 5

    #chosen on validation, never assumed -- the two heads are not equally good
    inference_semantic_weight: float = 0.5
    label_smoothing: float = 0.1

    #⚠️ False reproduces trial 5, whose semantic term came out near 0.015 of
    #the loss because the mean over kept pairs divides by ~batch^2. C7 flips it
    siglip_sum_over_candidates: bool = False

    # --- schedule ---
    warmup_fraction: float = 0.05
    min_lr: float = 1.0e-5
    ema_decay: float = 0.999

    # --- augmentation, applied inside the model after the normalisation ---
    time_mask_spans: int = 2
    time_mask_width: int = 40
    component_dropout: float = 0.1
    gaussian_noise: float = 0.0

    # --- controls and fitted stages ---
    shuffle_labels: bool = False
    envelope_pca_windows: int = 16384

    # --- laptop thermal budget. See ../throttle.py ---
    accelerator: str = "auto"
    torch_threads: int = 0
    duty_cycle: float = 1.0
    cooldown_every_minutes: float = 0.0
    cooldown_seconds: float = 0.0

    def __post_init__(self):
        """Catch the config mistakes that would otherwise train to chance."""
        #the model's input width IS the whitener rank; if these disagree the
        #stem silently reads the wrong number of components
        if self.n_channels != self.n_components:
            raise ValueError(
                f"n_channels ({self.n_channels}) must equal n_components "
                f"({self.n_components}) -- trial 6 reads whitened components, "
                f"not the 306 sensors. Set n_channels: {self.n_components}."
            )

        #a scored window outside the cached bounds has nothing to crop from
        low, high = self.window_bounds
        if self.tmin < low or self.tmax > high:
            raise ValueError(
                f"Scored window ({self.tmin}, {self.tmax}) s is not inside the "
                f"cached ({low}, {high}) s."
            )

        #⚠️ the holdout serves exactly [0, 1] s from word onset, so a window
        #reaching past it trains a model that can never be scored
        if self.tmin < 0.0 or self.tmax > 1.0:
            raise ValueError(
                f"Scored window ({self.tmin}, {self.tmax}) s reaches outside "
                f"[0, 1] s, which is all the holdout serves."
            )

        #every loss weight at zero trains nothing at all, and the run would
        #otherwise look healthy for an hour before anyone noticed
        if self.semantic_weight <= 0 and self.closed_set_weight <= 0:
            raise ValueError(
                "Both semantic_weight and closed_set_weight are 0 -- there is "
                "no classification loss left to train on."
            )

        #a semantic-only run cannot read its score off the closed-set head
        if self.semantic_weight <= 0 and self.inference_semantic_weight > 0:
            raise ValueError(
                "semantic_weight is 0 but inference_semantic_weight is "
                f"{self.inference_semantic_weight} -- the scored logits would "
                "come partly from an untrained head."
            )
        if self.closed_set_weight <= 0 and self.inference_semantic_weight < 1:
            raise ValueError(
                "closed_set_weight is 0 but inference_semantic_weight is "
                f"{self.inference_semantic_weight} -- the scored logits would "
                "come partly from an untrained head."
            )

        #negative noise is a typo, and the model would take its absolute value
        if self.gaussian_noise < 0:
            raise ValueError(f"gaussian_noise must be >= 0, got {self.gaussian_noise}")
        if self.envelope_pca_windows <= 0:
            raise ValueError(
                f"envelope_pca_windows must be > 0, got {self.envelope_pca_windows}"
            )

    def resolved_cache_dir(self):
        """Absolute path to the cache, derived from the repo root.

        RETURNS:
            Path to the shard directory
        """
        #one machine's absolute path is broken on the next, so it is built here
        return REPO_ROOT / self.cache_dir

    def resolved_word_embeddings(self):
        """Absolute path to the target word-embedding table.

        RETURNS:
            Path to the .npz written by `cli.build_word_embeddings`
        """
        return REPO_ROOT / self.word_embeddings


def load_trial6_config(*paths):
    """Load YAML run configs into a `Trial6Config`, merging model_kwargs.

    Files merge left to right, later ones winning, exactly as the shared
    loader does -- except `model_kwargs`, which merges KEY BY KEY. Each Variant
    C run changes one architecture knob, and its file should carry only that.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial6Config
    """
    merged = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            layer = yaml.safe_load(handle) or {}

        #every key but model_kwargs replaces what an earlier file said
        model_kwargs = layer.pop("model_kwargs", None)
        merged.update(layer)

        #⚠️ the one deep merge. `or {}` covers a file that writes the key bare
        if model_kwargs is not None:
            merged.setdefault("model_kwargs", {}).update(model_kwargs or {})

    #the same unknown-key check and coercion as the shared loader, so a typo
    #still raises and a YAML 1.1 "1e-4" string still becomes a float
    types = get_type_hints(Trial6Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {name: _coerce(name, value, types[name]) for name, value in merged.items()}
    return Trial6Config(**coerced)
