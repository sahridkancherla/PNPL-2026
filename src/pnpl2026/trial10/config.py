"""Trial-10 run config: the shared fields, plus what only this trial has.

Every default is d'Ascoli et al.'s, or brainmagick's where the paper is silent,
so `trial10.yaml` with no run file on top IS the reference (F0). The queue's
three other runs each move one thing back towards trials 7-8.

⚠️ `load_trial10_config` merges `model_kwargs` key by key across files, as
trials 6-8 do, so a run file carries only the knob it changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_type_hints

import yaml

from ..config import REPO_ROOT, Config, _coerce

#how a word becomes a token: every word (the reference) or only scored ones
TOKEN_SETS = ("all", "invocab")


@dataclass
class Trial10Config(Config):
    """Everything trial 10 needs that the shared config does not carry.

    ATTRS:
        cache_dir (str): the reference cache, repo-relative
        word_embeddings (str): the all-word t5 table, repo-relative
        centre_embeddings (bool): subtract the mean row. The paper does not
        clamp (float): applied per window after the per-recording scaler
        baseline_seconds (float): mean of this much of the window's start is
            subtracted per channel. 0 disables it
        token_set (str): "all", every word a token (the reference), or
            "invocab", scored words only (F2)
        max_tokens (int): longer sentences are cut into near-equal runs
        siglip_dedup (bool): drop repeated words from the SigLIP loss
        holdout_max_tokens (int): the holdout protocol's longest sentence
        tail_seconds (list): [low, high] MEG after a holdout-protocol
            sequence's last word, as the holdout serves it
        isolated_fraction (float): holdout-protocol words served alone
        isolated_seconds (float): their window
        jitter_seconds (float): one-sided onset jitter, training only. 0 is
            the reference
        warmup_fraction (float): fraction of steps spent warming up
        min_lr (float): floor of the cosine decay
        ema_decay (float): 0 disables the parameter EMA, as the reference
        matmul_precision (str): torch.set_float32_matmul_precision. "high"
            allows TF32 on Ampere and later; no effect on an RTX6000
        precision (str): Lightning's precision, "32" or "bf16-mixed"
        accelerator (str): "auto", "cuda" or "cpu"
        torch_threads (int): 0 leaves torch's default
        duty_cycle (float): laptop thermal pacing; 1.0 is off
        cooldown_every_minutes (float): 0 disables the periodic pause
        cooldown_seconds (float): length of that pause
    """

    cache_dir: str = "data/trial10_reference"
    word_embeddings: str = "data/word_embeddings_t5_all.npz"
    centre_embeddings: bool = False

    # --- the reference's preprocessing, per window ---
    clamp: float = 5.0
    baseline_seconds: float = 0.5

    # --- sequences ---
    token_set: str = "all"
    max_tokens: int = 64
    siglip_dedup: bool = True

    # --- the holdout protocol, scored by cli.evaluate ---
    holdout_max_tokens: int = 9
    tail_seconds: list = field(default_factory=lambda: [1.1, 1.7])
    isolated_fraction: float = 0.096
    isolated_seconds: float = 1.0

    # --- training ---
    jitter_seconds: float = 0.0
    warmup_fraction: float = 0.01
    min_lr: float = 0.0
    ema_decay: float = 0.0
    matmul_precision: str = "high"
    precision: str = "32"

    # --- machine ---
    accelerator: str = "auto"
    torch_threads: int = 0
    duty_cycle: float = 1.0
    cooldown_every_minutes: float = 0.0
    cooldown_seconds: float = 0.0

    def __post_init__(self):
        """Catch the config mistakes that would otherwise train to chance."""
        #every window starts at word onset, as the reference cuts them
        if self.tmin != 0.0:
            raise ValueError(f"tmin must be 0.0 -- windows start at onset, got {self.tmin}")

        if self.token_set not in TOKEN_SETS:
            raise ValueError(f"token_set must be one of {TOKEN_SETS}, got {self.token_set!r}")

        #the baseline slice must sit inside the shortest window the holdout
        #protocol serves, or an isolated word would have no baseline
        if not 0.0 <= self.baseline_seconds < self.isolated_seconds <= self.tmax:
            raise ValueError(
                f"Need 0 <= baseline_seconds ({self.baseline_seconds}) < "
                f"isolated_seconds ({self.isolated_seconds}) <= tmax ({self.tmax})."
            )

        low, high = self.tail_seconds
        if not self.isolated_seconds <= low <= high:
            raise ValueError(f"tail_seconds {self.tail_seconds} must start at isolated_seconds or later.")

        if self.clamp <= 0:
            raise ValueError(f"clamp must be > 0, got {self.clamp}")
        if self.precision not in ("32", "bf16-mixed", "16-mixed"):
            raise ValueError(f"precision must be '32', 'bf16-mixed' or '16-mixed', got {self.precision!r}")

    @property
    def window_seconds(self):
        """The full window after onset: tmax, since tmin is 0."""
        return self.tmax - self.tmin

    def samples(self, seconds):
        """Seconds to samples at the cache's rate.

        ARGS:
            seconds (float)

        RETURNS:
            int
        """
        return int(round(seconds * self.sfreq))

    def resolved_cache_dir(self):
        """Absolute path to the reference cache, derived from the repo root."""
        #one machine's absolute path is broken on the next, so it is built here
        return REPO_ROOT / self.cache_dir

    def resolved_word_embeddings(self):
        """Absolute path to the all-word t5 table."""
        return REPO_ROOT / self.word_embeddings


def load_trial10_config(*paths):
    """Load YAML run configs into a `Trial10Config`, merging model_kwargs.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial10Config
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
    types = get_type_hints(Trial10Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {name: _coerce(name, value, types[name]) for name, value in merged.items()}
    return Trial10Config(**coerced)
