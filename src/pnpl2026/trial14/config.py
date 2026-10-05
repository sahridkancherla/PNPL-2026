"""Trial-14 run config: trial 13's fields, plus the 50-word head and the chunked protocol.

A superset of trial 13's config (itself a superset of trial 10's), field for
field, so any trial 10, 12 or 13 checkpoint's saved hyperparameters load here
unchanged -- which is how trial 14 starts from G2's weights, and how
`cli.calibrate` scores the older runs under trial 14's protocols.

⚠️ `load_trial14_config` merges `model_kwargs` key by key across files, so a
run file carries only the knob it changes.

⚠️ The protocol fields (`PROTOCOL_FIELDS`) define how runs are SCORED. They
are reset to this file's defaults whenever a checkpoint is evaluated, so every
run -- trial 10, 12, 13 or 14 -- is judged on identical sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import get_type_hints

import yaml

from ..config import REPO_ROOT, Config, _coerce

#how a word becomes a token: every word (the reference) or only scored ones
TOKEN_SETS = ("all", "invocab")

#how sequences are cut for scoring; see data/sequences.py
PROTOCOLS = ("reference", "chunked", "fragment", "sparse")

#the fields that define a protocol, reset to the defaults below at evaluation
PROTOCOL_FIELDS = (
    "max_tokens",
    "holdout_max_tokens",
    "tail_seconds",
    "isolated_fraction",
    "isolated_seconds",
    "fragment_min_words",
    "fragment_max_gap_seconds",
    "chunk_lengths",
)


@dataclass
class Trial14Config(Config):
    """Everything trial 14 needs that the shared config does not carry.

    ATTRS:
        cache_dir (str): trial 10's reference cache, repo-relative
        word_embeddings (str): trial 10's all-word t5 table, repo-relative
        centre_embeddings (bool): subtract the mean row. The paper does not
        clamp (float): applied per window after the per-recording scaler
        baseline_seconds (float): mean of this much of the window's start is
            subtracted per channel. 0 disables it
        token_set (str): "all", every word a token, or "invocab"
        max_tokens (int): longer sentences are cut into near-equal runs
        siglip_dedup (bool): drop repeated words from the SigLIP loss
        holdout_max_tokens (int): the longest fragment or chunk, 9
        tail_seconds (list): [low, high] MEG kept after a fragment's or
            chunk's last onset; the real holdout's 1.13-1.68 s
        isolated_fraction (float): scored words served alone, 92 of 960
        isolated_seconds (float): their window, exactly 1 s
        fragment_min_words (int): the fragment protocol's shortest run
        fragment_max_gap_seconds (float): onsets further apart end a run
        chunk_lengths (list): weights of chunk lengths 1..9 in the chunked
            protocol: the real holdout's fragment-length histogram
        fragment_p (float): training items cropped to a holdout-like fragment
        isolated_p (float): training items reduced to one word with 1 s
        fragment_lengths (list): weights of training fragment lengths 1..9
        selection_protocol (str): the validation set's protocol, which picks
            best.ckpt and stops the run
        init_checkpoint (str): weights to start from, repo-relative; "{seed}"
            is replaced by the run's seed. Empty starts from scratch
        freeze_backbone (bool): train only the 50-word head; everything the
            checkpoint brought stays fixed, BatchNorm statistics included
        semantic_weight (float): weight of the D-SigLIP loss; 0 drops it
        closed_weight (float): weight of the 50-word head's loss; 0 drops it
        closed_dropout (float): dropout on the vector the head reads
        logit_adjust_tau (float): the head trains on logits + tau * log(prior),
            which makes its plain logits balanced-error optimal (Menon et al.
            2021). 0 is plain cross-entropy
        label_smoothing (float): on the head's cross-entropy
        head_lr (float): the head's learning rate; it starts from random
        inference_closed_weight (float): the blend at scoring time, 0 the
            retrieval scores alone, 1 the head alone
        jitter_seconds (float): one-sided onset jitter, training only
        warmup_fraction (float): fraction of steps spent warming up
        min_lr (float): floor of the cosine decay
        ema_decay (float): 0 disables the parameter EMA
        matmul_precision (str): torch.set_float32_matmul_precision
        precision (str): Lightning's precision, "32", "bf16-mixed", "16-mixed"
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

    # --- the holdout's shape, measured on subject 0's holdout files 2026-10-02 ---
    holdout_max_tokens: int = 9
    tail_seconds: list = field(default_factory=lambda: [1.13, 1.68])
    isolated_fraction: float = 0.096
    isolated_seconds: float = 1.0
    fragment_min_words: int = 4
    fragment_max_gap_seconds: float = 0.8
    chunk_lengths: list = field(default_factory=lambda: [2, 0, 2, 24, 55, 37, 23, 11, 2])

    # --- training under holdout conditions ---
    fragment_p: float = 0.0
    isolated_p: float = 0.0
    fragment_lengths: list = field(default_factory=lambda: [3, 1, 3, 25, 56, 38, 24, 12, 3])
    selection_protocol: str = "chunked"

    # --- the 50-word head, fine-tuned from earlier weights ---
    init_checkpoint: str = ""
    freeze_backbone: bool = False
    semantic_weight: float = 1.0
    closed_weight: float = 0.0
    closed_dropout: float = 0.1
    logit_adjust_tau: float = 1.0
    label_smoothing: float = 0.0
    head_lr: float = 1.0e-3
    inference_closed_weight: float = 0.0

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
        if self.selection_protocol not in PROTOCOLS:
            raise ValueError(f"selection_protocol must be one of {PROTOCOLS}, got {self.selection_protocol!r}")

        #the baseline slice must sit inside the shortest window ever served,
        #or an isolated word would have no baseline
        if not 0.0 <= self.baseline_seconds < self.isolated_seconds <= self.tmax:
            raise ValueError(
                f"Need 0 <= baseline_seconds ({self.baseline_seconds}) < "
                f"isolated_seconds ({self.isolated_seconds}) <= tmax ({self.tmax})."
            )

        low, high = self.tail_seconds
        if not self.isolated_seconds <= low <= high:
            raise ValueError(f"tail_seconds {self.tail_seconds} must start at isolated_seconds or later.")

        #the shaping probabilities split each item three ways
        if not (0.0 <= self.fragment_p and 0.0 <= self.isolated_p and self.fragment_p + self.isolated_p <= 1.0):
            raise ValueError(f"Need fragment_p, isolated_p >= 0 summing to <= 1; got {self.fragment_p}, {self.isolated_p}")

        #one weight per length 1..holdout_max_tokens, for training and for chunks
        for name in ("fragment_lengths", "chunk_lengths"):
            weights = list(getattr(self, name))
            if len(weights) != self.holdout_max_tokens or min(weights) < 0 or sum(weights) <= 0:
                raise ValueError(
                    f"{name} needs {self.holdout_max_tokens} non-negative weights with a positive sum, got {weights}"
                )
        if not 1 <= self.fragment_min_words <= self.holdout_max_tokens:
            raise ValueError(f"fragment_min_words must be in 1..{self.holdout_max_tokens}, got {self.fragment_min_words}")

        #a run must learn something, and a frozen run must have a head to train
        if self.semantic_weight < 0 or self.closed_weight < 0 or self.semantic_weight + self.closed_weight <= 0:
            raise ValueError(f"Need non-negative loss weights, not both 0; got {self.semantic_weight}, {self.closed_weight}")
        if self.freeze_backbone and (self.closed_weight <= 0 or not self.init_checkpoint):
            raise ValueError("freeze_backbone trains the head alone: it needs closed_weight > 0 and an init_checkpoint.")
        if not 0.0 <= self.inference_closed_weight <= 1.0:
            raise ValueError(f"inference_closed_weight must be in [0, 1], got {self.inference_closed_weight}")
        if self.inference_closed_weight > 0 and self.closed_weight <= 0:
            raise ValueError("inference_closed_weight > 0 would blend in a head this run never trains.")

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

    def resolved_init_checkpoint(self):
        """The checkpoint to start from, with this run's seed filled in, or None.

        RETURNS:
            Path or None
        """
        if not self.init_checkpoint:
            return None
        #"{seed}" lets one config start each seed from its own parent seed
        return REPO_ROOT / self.init_checkpoint.replace("{seed}", str(self.seed))


def load_trial14_config(*paths):
    """Load YAML run configs into a `Trial14Config`, merging model_kwargs.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial14Config
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
    types = get_type_hints(Trial14Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {name: _coerce(name, value, types[name]) for name, value in merged.items()}
    return Trial14Config(**coerced)


def with_protocol_defaults(cfg):
    """A copy of `cfg` whose protocol fields are this trial's defaults.

    An older checkpoint saved its own trial's protocol settings; scoring it
    under those would compare runs on different sequences.

    ARGS:
        cfg (Trial14Config): typically rebuilt from a checkpoint

    RETURNS:
        Trial14Config
    """
    defaults = Trial14Config()
    return replace(cfg, **{name: getattr(defaults, name) for name in PROTOCOL_FIELDS})
