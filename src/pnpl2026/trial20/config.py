"""Trial-20 run config: trial 19's, unchanged (the word transformer is a model_kwargs switch).

The model is `trial20_continuous` (`models/continuous.py`); its sizes live
in `model_kwargs`, merged key by key across files. New top-level fields, all
with holdout-matched defaults:

    whole_tail, lead_max_seconds, marker_drop_p, time_mask_p,
    time_mask_seconds     how a training span is cut around its words
    phoneme_dir, phoneme_weight, phoneme_lag_seconds
                          the per-step phoneme head ("" = off)
    word_descriptions, tower_dim, tower_hidden, tower_anchor_weight,
    tower_score_mode      the word tower ("" = off)

`second_embeddings` (Variant K's head) and the all-data switches are kept.
Variant M's refiner is gone.

⚠️ The protocol fields (`PROTOCOL_FIELDS`) define how runs are SCORED. They
are reset to this file's defaults whenever a checkpoint is evaluated, so every
run is judged on identical sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import get_type_hints

import yaml

from ..config import REPO_ROOT, Config, _coerce

#how a word becomes a token: every word (the reference) or only scored ones
TOKEN_SETS = ("all", "invocab")

#how sequences are cut for scoring; see data/sequences.py
PROTOCOLS = ("reference", "fragment", "sparse", "isolated")

#what the training set is cut by: whole sentences (shaped or not), or every
#word alone for the lone-word specialist
TRAINING_PROTOCOLS = ("reference", "isolated")

#the fields that define a protocol, reset to the defaults below at evaluation
PROTOCOL_FIELDS = (
    "max_tokens",
    "holdout_max_tokens",
    "tail_seconds",
    "isolated_fraction",
    "isolated_seconds",
    "fragment_min_words",
    "fragment_max_gap_seconds",
)


@dataclass
class Trial20Config(Config):
    """Everything trial 13 needs that the shared config does not carry.

    ATTRS:
        cache_dir (str): trial 10's reference cache, repo-relative
        word_embeddings (str): trial 10's all-word t5 table, repo-relative
        centre_embeddings (bool): subtract the mean row. The paper does not
        clamp (float): applied per window after the per-recording scaler
        baseline_seconds (float): mean of this much of the window's start is
            subtracted per channel. 0 disables it
        second_embeddings (str): Variant K's second target table,
            repo-relative; "" means one head, as every earlier trial
        second_loss_weight (float): the second head's share of the loss
        token_set (str): "all", every word a token, or "invocab"
        max_tokens (int): longer sentences are cut into near-equal runs
        siglip_dedup (bool): drop repeated words from the SigLIP loss
        holdout_max_tokens (int): the longest fragment, 9 in the holdout
        tail_seconds (list): [low, high] MEG kept after a fragment's last
            onset. Measured on the real holdout: 1.13-1.68 s (5th-95th)
        isolated_fraction (float): scored words served alone, 92 of 960
        isolated_seconds (float): their window, exactly 1 s
        fragment_min_words (int): shortest run of consecutive vocabulary words
            the fragment protocol keeps; the holdout's are almost all 4-8
        fragment_max_gap_seconds (float): onsets further apart than this end a
            run; 99.4% of the holdout's gaps are under 0.8 s
        fragment_p (float): training items cropped to a holdout-like fragment
        isolated_p (float): training items reduced to one word with 1 s
        run_p (float): trial 17 -- training items served as holdout runs:
            scored words only, neighbours at most `run_max_skipped` unscored
            words and `fragment_max_gap_seconds` apart, lengths as
            `fragment_lengths` from `fragment_min_words` (4) up, the MEG cut a
            tail after the last onset; a sentence with no such run stays whole
        run_max_skipped (int): unscored words allowed between neighbours in
            a holdout run; about a third of the holdout's pairs skip one
        fragment_lengths (list): relative weights of fragment lengths 1..9 in
            training: the holdout's histogram plus one
        selection_protocol (str): the validation set's protocol, which picks
            best.ckpt and stops the run
        training_protocol (str): "reference", whole sentences as trial 13
            trained (with any shaping), or "isolated", every training word
            alone with 1 s -- the lone-word specialist; no shaping then
        jitter_seconds (float): one-sided onset jitter, training only
        warmup_fraction (float): fraction of steps spent warming up
        min_lr (float): floor of the cosine decay
        ema_decay (float): 0 disables the parameter EMA
        matmul_precision (str): torch.set_float32_matmul_precision
        precision (str): Lightning's precision, "32", "bf16-mixed", "16-mixed"
        accelerator (str): "auto", "cuda" or "cpu"
        whole_tail (str): "holdout" draws a whole training sentence's tail
            after its last onset from `tail_seconds`; "full" keeps tmax
        lead_max_seconds (float): random lead before a span's first onset,
            training only; 0 is the holdout's shape
        marker_drop_p (float): chance a training item tells the network
            only its scored words' onsets
        time_mask_p (float): chance a training span gets 1-2 blanked stretches
        time_mask_seconds (list): [low, high] length of each stretch
        phoneme_dir (str): per-sample phoneme labels, repo-relative
            (cli.build_phoneme_labels); "" means no phoneme head
        phoneme_weight (float): its cross-entropy's share of the loss
        phoneme_lag_seconds (float): the MEG trails the sound by this much
        word_descriptions (str): the tower's description table,
            repo-relative (cli.build_word_descriptions); "" means no tower
        tower_dim (int): the shared space's width
        tower_hidden (int): the tower's hidden width
        tower_anchor_weight (float): the frozen t5 head's loss share with a
            tower; 0 drops it
        tower_score_mode (str): "combine" averages the heads' z-scored
            scores; "tower" ranks by the tower alone
        train_partitions (list): subject 0's pnpl partitions trained on;
            ["train", "validation", "test"] is all of subject 0
        others_cache_dir (str): the other subjects' cache, repo-relative; ""
            trains on subject 0 alone
        n_subjects (int): subject layers; written by cli.train from the
            others' cache, so a checkpoint rebuilds without it
        epochs_from (str): a glob of earlier runs' evaluation.json files; with
            unclean selection, max_epochs becomes their mean best epoch count
        torch_threads (int): 0 leaves torch's default
        duty_cycle (float): laptop thermal pacing; 1.0 is off
        cooldown_every_minutes (float): 0 disables the periodic pause
        cooldown_seconds (float): length of that pause
    """

    cache_dir: str = "data/trial10_reference"
    word_embeddings: str = "data/word_embeddings_t5_all.npz"
    centre_embeddings: bool = False

    # --- Variant K: a second head against a second target table ---
    second_embeddings: str = ""
    second_loss_weight: float = 1.0

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

    # --- training under holdout conditions ---
    fragment_p: float = 0.0
    isolated_p: float = 0.0
    run_p: float = 0.0
    run_max_skipped: int = 1
    fragment_lengths: list = field(default_factory=lambda: [3, 1, 3, 25, 56, 38, 24, 12, 3])
    selection_protocol: str = "fragment"
    training_protocol: str = "reference"

    # --- training ---
    jitter_seconds: float = 0.0
    warmup_fraction: float = 0.01
    min_lr: float = 0.0
    ema_decay: float = 0.0
    matmul_precision: str = "high"
    precision: str = "32"

    # --- trial 19: how a training span is cut ---
    whole_tail: str = "holdout"
    lead_max_seconds: float = 0.0
    marker_drop_p: float = 0.5
    time_mask_p: float = 0.0
    time_mask_seconds: list = field(default_factory=lambda: [0.1, 0.3])

    # --- trial 19: the phoneme head ---
    phoneme_dir: str = ""
    phoneme_weight: float = 0.5
    phoneme_lag_seconds: float = 0.15

    # --- trial 19: the word tower ---
    word_descriptions: str = ""
    tower_dim: int = 256
    tower_hidden: int = 512
    tower_anchor_weight: float = 1.0
    tower_score_mode: str = "combine"

    # --- all-data training: subject 0's every partition, the other subjects ---
    train_partitions: list = field(default_factory=lambda: ["train"])
    others_cache_dir: str = ""
    n_subjects: int = 1
    epochs_from: str = ""

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
        if self.training_protocol not in TRAINING_PROTOCOLS:
            raise ValueError(f"training_protocol must be one of {TRAINING_PROTOCOLS}, got {self.training_protocol!r}")

        #shaping cuts whole sentences; an all-isolated training set has none
        if self.training_protocol == "isolated" and (self.fragment_p > 0 or self.isolated_p > 0 or self.run_p > 0):
            raise ValueError(
                "training_protocol isolated serves every word alone already; set fragment_p, isolated_p and run_p to 0."
            )

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

        #the shaping probabilities split each item four ways; the rest stay whole
        shares = (self.fragment_p, self.isolated_p, self.run_p)
        if min(shares) < 0.0 or sum(shares) > 1.0:
            raise ValueError(f"Need fragment_p, isolated_p, run_p >= 0 summing to <= 1; got {shares}")
        if self.run_max_skipped < 0:
            raise ValueError(f"run_max_skipped must be >= 0, got {self.run_max_skipped}")

        #one weight per fragment length, 1..holdout_max_tokens
        weights = list(self.fragment_lengths)
        if len(weights) != self.holdout_max_tokens or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(
                f"fragment_lengths needs {self.holdout_max_tokens} non-negative weights with a positive sum, "
                f"got {weights}"
            )
        if not 1 <= self.fragment_min_words <= self.holdout_max_tokens:
            raise ValueError(f"fragment_min_words must be in 1..{self.holdout_max_tokens}, got {self.fragment_min_words}")

        if self.clamp <= 0:
            raise ValueError(f"clamp must be > 0, got {self.clamp}")

        #a second table the same as the first would make the second head a copy
        if self.second_embeddings and self.second_embeddings == self.word_embeddings:
            raise ValueError("second_embeddings names the same table as word_embeddings.")
        if self.second_loss_weight < 0:
            raise ValueError(f"second_loss_weight must be >= 0, got {self.second_loss_weight}")
        if self.precision not in ("32", "bf16-mixed", "16-mixed"):
            raise ValueError(f"precision must be '32', 'bf16-mixed' or '16-mixed', got {self.precision!r}")

        #all-data: subject 0's own partitions, train always among them
        known = {"train", "validation", "test"}
        if "train" not in self.train_partitions or not set(self.train_partitions) <= known:
            raise ValueError(f"train_partitions must include 'train' and name only {sorted(known)}, got {self.train_partitions}")
        if self.n_subjects < 1:
            raise ValueError(f"n_subjects must be >= 1, got {self.n_subjects}")

        #trial 19: the span settings and the two optional heads
        if self.whole_tail not in ("holdout", "full"):
            raise ValueError(f"whole_tail must be 'holdout' or 'full', got {self.whole_tail!r}")
        if self.lead_max_seconds < 0 or not 0.0 <= self.marker_drop_p <= 1.0 or not 0.0 <= self.time_mask_p <= 1.0:
            raise ValueError("lead_max_seconds must be >= 0; marker_drop_p and time_mask_p in [0, 1].")
        low, high = self.time_mask_seconds
        if not 0.0 < low <= high:
            raise ValueError(f"time_mask_seconds must be 0 < low <= high, got {self.time_mask_seconds}")
        if self.phoneme_weight < 0 or self.phoneme_lag_seconds < 0:
            raise ValueError("phoneme_weight and phoneme_lag_seconds must be >= 0.")
        if self.tower_dim <= 0 or self.tower_hidden <= 0 or self.tower_anchor_weight < 0:
            raise ValueError("tower_dim and tower_hidden must be > 0, tower_anchor_weight >= 0.")
        if self.tower_score_mode not in ("combine", "tower"):
            raise ValueError(f"tower_score_mode must be 'combine' or 'tower', got {self.tower_score_mode!r}")

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

    @property
    def has_second_head(self):
        """Whether this run trains Variant K's second head."""
        return bool(self.second_embeddings)

    def resolved_second_embeddings(self):
        """Absolute path to the second target table; None with one head."""
        return REPO_ROOT / self.second_embeddings if self.second_embeddings else None

    @property
    def clean_selection(self):
        """Whether validation is untouched by training: no subject-0 validation
        recordings in it, and no other subject (they heard those same stories)."""
        return "validation" not in self.train_partitions and not self.others_cache_dir

    def resolved_others_cache_dir(self):
        """Absolute path to the other subjects' cache; None without one."""
        return REPO_ROOT / self.others_cache_dir if self.others_cache_dir else None

    @property
    def has_phoneme_head(self):
        """Whether this run trains the per-step phoneme head."""
        return bool(self.phoneme_dir)

    def resolved_phoneme_dir(self):
        """Absolute path to the phoneme labels; None without the head."""
        return REPO_ROOT / self.phoneme_dir if self.phoneme_dir else None

    @property
    def has_tower(self):
        """Whether this run trains the word tower."""
        return bool(self.word_descriptions)

    def resolved_word_descriptions(self):
        """Absolute path to the description table; None without a tower."""
        return REPO_ROOT / self.word_descriptions if self.word_descriptions else None




def load_trial20_config(*paths):
    """Load YAML run configs into a `Trial20Config`, merging model_kwargs.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial20Config
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
    types = get_type_hints(Trial20Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {name: _coerce(name, value, types[name]) for name, value in merged.items()}
    return Trial20Config(**coerced)


def with_protocol_defaults(cfg):
    """A copy of `cfg` whose protocol fields are this trial's defaults.

    A trial 10 checkpoint saved trial 10's protocol settings; scoring it under
    those would compare runs on different sequences.

    ARGS:
        cfg (Trial20Config): typically rebuilt from a checkpoint

    RETURNS:
        Trial20Config
    """
    defaults = Trial20Config()
    return replace(cfg, **{name: getattr(defaults, name) for name in PROTOCOL_FIELDS})
