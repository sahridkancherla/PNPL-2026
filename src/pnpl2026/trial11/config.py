"""Trial-11 run config: the shared fields, plus what only this trial has.

Trial 11 serves each sentence as ONE stream -- its MEG from the first scored
onset to its end -- and runs a Zipformer-style multi-rate encoder over it,
reading every word off the shared timeline at the end. Everything that shapes
a training sequence is set here from what the holdout actually serves,
measured 2026-09-30:

    sentence source  156 sentences, 868 scored words, 1-9 per sentence
    word source      92 isolated words, 1 s each, no context
    MEG after a sentence's last word: 1.13-1.68 s for 90% of sentences
    only scored words have onsets, so only in-vocabulary words are context

`tmax` is the padded STREAM length here, not a per-word window.

⚠️ `load_trial11_config` merges `model_kwargs` key by key across files, as
trial 6's loader does, so a run file carries only the knob it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_type_hints

import yaml

from ..config import REPO_ROOT, Config, _coerce


@dataclass
class Trial11Config(Config):
    """Everything trial 11 needs that the shared config does not carry.

    ATTRS:
        continuous_dir (str): trial 7's continuous cache, repo-relative --
            trial 11 reads it unchanged and has no builder of its own
        n_components (int): whitener rank, the cache's component count
        max_tokens (int): longest sequence, the holdout's longest sentence
        tail_seconds (list): [low, high] of the MEG kept after a sequence's
            last onset, drawn uniformly in training, fixed in validation
        isolated_fraction (float): share of words served alone with a 1 s
            window, like the holdout's word source
        isolated_seconds (float): the isolated words' window
        max_onset_span_seconds (float): a training sentence is split wherever
            its first and last onsets would be further apart than this, so
            every sequence fits one stream of `tmax` seconds
        jitter_seconds (float): one-sided jitter of a whole stream's start,
            training only
        class_weight_power (float): 0 unweighted, 0.5 sqrt-inverse, 1 inverse
        word_embeddings (str): .npz of target word embeddings, repo-relative
        centre_embeddings (bool): subtract the mean target row before training
        semantic_weight (float): weight on the SigLIP loss. 0 disables it
        closed_set_weight (float): weight on the 50-way cross-entropy
        inference_semantic_weight (float): the semantic head's share of the
            scored logits, 0 = closed-set only
        label_smoothing (float): on the closed-set head only
        siglip_dedup (bool): keep one occurrence per word in the SigLIP loss.
            False is D4, plain SigLIP
        word_mask_p (float): must be 0 -- trial 11 has no word tokens to mask
        mask_target (bool): must be False, for the same reason
        warmup_fraction (float): fraction of total steps spent warming up
        min_lr (float): floor the cosine schedule decays to
        ema_decay (float): 0 disables the parameter EMA
        component_dropout (float): probability of zeroing a whole component
        gaussian_noise (float): std of additive noise after normalisation
        shuffle_labels (bool): permute the TRAINING labels -- the chance line
        grad_probe_every (int): steps between gradient-balance measurements.
            0 disables them
        accelerator (str): "auto", "cuda" or "cpu"
        torch_threads (int): 0 leaves torch's default
        duty_cycle (float): fraction of wall-clock spent computing
        cooldown_every_minutes (float): 0 disables the periodic long pause
        cooldown_seconds (float): length of that pause
    """

    continuous_dir: str = "data/trial7_continuous"
    n_components: int = 73

    # --- holdout matching ---
    max_tokens: int = 9
    tail_seconds: list = field(default_factory=lambda: [1.1, 1.7])
    isolated_fraction: float = 0.096
    isolated_seconds: float = 1.0
    max_onset_span_seconds: float = 3.0
    jitter_seconds: float = 0.03

    #full inverse frequency spans the vocabulary's 121x and destabilises the
    #batch gradient; the square root keeps the direction, not the magnitude
    class_weight_power: float = 0.5

    # --- heads and losses ---
    word_embeddings: str = "data/word_embeddings_t5.npz"
    centre_embeddings: bool = True
    semantic_weight: float = 1.0
    closed_set_weight: float = 1.0
    inference_semantic_weight: float = 0.5
    label_smoothing: float = 0.1
    siglip_dedup: bool = True

    # --- the sentence transformer's training-time masking ---
    word_mask_p: float = 0.0
    mask_target: bool = False

    # --- schedule ---
    warmup_fraction: float = 0.05
    min_lr: float = 1.0e-5
    ema_decay: float = 0.999

    # --- augmentation, inside the model after the normalisation ---
    component_dropout: float = 0.1
    gaussian_noise: float = 0.1

    # --- controls and diagnostics ---
    shuffle_labels: bool = False
    grad_probe_every: int = 50

    # --- laptop thermal budget. See ../throttle.py ---
    accelerator: str = "auto"
    torch_threads: int = 0
    duty_cycle: float = 1.0
    cooldown_every_minutes: float = 0.0
    cooldown_seconds: float = 0.0

    def __post_init__(self):
        """Catch the config mistakes that would otherwise train to chance."""
        #the model's input width IS the whitener rank
        if self.n_channels != self.n_components:
            raise ValueError(
                f"n_channels ({self.n_channels}) must equal n_components "
                f"({self.n_components}) -- trial 11 reads whitened components."
            )

        #every window starts at word onset: the holdout has no pre-onset
        #samples for a sentence's first word or for any isolated word
        if self.tmin != 0.0:
            raise ValueError(f"tmin must be 0.0 -- windows start at onset, got {self.tmin}")

        #the isolated window must fit inside the longest one
        if not 0 < self.isolated_seconds <= self.tmax:
            raise ValueError(
                f"isolated_seconds {self.isolated_seconds} must be in (0, tmax={self.tmax}]."
            )

        #the holdout guarantees at least 1 s after every scored onset, so a
        #tail below the isolated window would train on windows it never serves
        low, high = self.tail_seconds
        if not self.isolated_seconds <= low <= high:
            raise ValueError(
                f"tail_seconds {self.tail_seconds} must satisfy "
                f"{self.isolated_seconds} <= low <= high."
            )

        if not 0.0 <= self.isolated_fraction < 1.0:
            raise ValueError(f"isolated_fraction must be in [0, 1), got {self.isolated_fraction}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

        #every loss weight at zero trains nothing at all
        if self.semantic_weight <= 0 and self.closed_set_weight <= 0:
            raise ValueError("Both semantic_weight and closed_set_weight are 0.")

        #the scored logits must not come partly from an untrained head
        if self.semantic_weight <= 0 and self.inference_semantic_weight > 0:
            raise ValueError(
                "semantic_weight is 0 but inference_semantic_weight is "
                f"{self.inference_semantic_weight}."
            )
        if self.closed_set_weight <= 0 and self.inference_semantic_weight < 1:
            raise ValueError(
                "closed_set_weight is 0 but inference_semantic_weight is "
                f"{self.inference_semantic_weight}."
            )

        #words are not tokens until the readout, so there is nothing to mask
        if self.word_mask_p > 0 or self.mask_target:
            raise ValueError("Trial 11 has no word tokens to mask: word_mask_p 0, mask_target false.")

        if self.gaussian_noise < 0:
            raise ValueError(f"gaussian_noise must be >= 0, got {self.gaussian_noise}")

        #the longest sequence must fit one stream: span plus tail plus jitter
        longest = self.max_onset_span_seconds + max(self.tail_seconds) + self.jitter_seconds
        if longest > self.window_seconds:
            raise ValueError(
                f"A sequence can need {longest:.2f} s of stream but tmax is "
                f"{self.window_seconds:.2f} s; raise tmax or lower max_onset_span_seconds."
            )

        #a stack layout that does not fit fails here, in the PBS preflight,
        #rather than after the job has waited in the queue
        factors = self.model_kwargs.get("stack_factors", [1, 2, 4, 8, 4, 2])
        depths = self.model_kwargs.get("stack_depths", [1, 1, 1, 1, 1, 1])
        if len(factors) != len(depths):
            raise ValueError(f"stack_factors {factors} and stack_depths {depths} differ in length")
        #the decimator's frame count: one frame per `factor` samples, rounded up
        n_times = int(round(self.window_seconds * self.sfreq))
        n_frames = -(-n_times // self.model_kwargs.get("decimate_factor", 5))
        uneven = [factor for factor in factors if n_frames % factor != 0]
        if uneven:
            raise ValueError(f"{n_frames} frames per stream do not divide by stack factors {uneven}")

    @property
    def window_seconds(self):
        """The padded stream length in seconds: tmax, since tmin is 0."""
        return self.tmax - self.tmin

    def resolved_continuous_dir(self):
        """Absolute path to the continuous cache, derived from the repo root.

        RETURNS:
            Path to the directory
        """
        #one machine's absolute path is broken on the next, so it is built here
        return REPO_ROOT / self.continuous_dir

    def resolved_word_embeddings(self):
        """Absolute path to the target word-embedding table.

        RETURNS:
            Path to the .npz written by `cli.build_word_embeddings`
        """
        return REPO_ROOT / self.word_embeddings


def load_trial11_config(*paths):
    """Load YAML run configs into a `Trial11Config`, merging model_kwargs.

    Files merge left to right, later ones winning, except `model_kwargs`,
    which merges KEY BY KEY so each run file carries only its change.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial11Config
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

    #the same unknown-key check and coercion as the shared loader
    types = get_type_hints(Trial11Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {name: _coerce(name, value, types[name]) for name, value in merged.items()}
    return Trial11Config(**coerced)
