"""Trial-18 run config: trial 17's fields, plus Variant M's refinement.

Trial 16's fields, off by default:

    second_embeddings    a second all-word target table (another t5 layer),
                         repo-relative; "" means no second head
    second_loss_weight   weight of the second head's SigLIP loss

Trial 18's, off by default: `refine_base` names a trained checkpoint
(`{seed}` filled from the run's seed); set, the run trains Variant M's
refiner on that checkpoint, frozen. Such a run takes every other setting from
the BASE (`refinement_config`): only `REFINE_RUN_FIELDS` come from its own
files, so the base's architecture, data and shaping can never drift.

All-data training, off by default: `train_partitions` adds subject 0's
validation and test recordings to training, `others_cache_dir` adds the 32
broad-track subjects (`cli/build_others.py`). Either one makes local
validation unclean, so such a run trains a fixed number of epochs
(`epochs_from` reads them off earlier runs) and keeps the last one.

Trial 23's final recipes (`cli/anneal.py`), off by default:

    schedule           "cosine" (every earlier run) or "one_minus_sqrt", the
                       anneal's 1 - sqrt(progress) decay
    freeze_batchnorm   every BatchNorm keeps its running statistics in training
    freeze_backbone    the network runs as at inference (dropout off); only
                       parameters left trainable by the caller learn
    sam_rho            > 0 wraps AdamW in sharpness-aware minimisation
                       (`sam.py`); needs precision "32" or "bf16-mixed"

`input_norm` is an encoder argument and lives in `model_kwargs`. The dataclass
is a superset of trial 10's, 13's and 15's configs, so their checkpoints'
saved hyperparameters load here unchanged and run with one head.

⚠️ `load_trial18_config` merges `model_kwargs` key by key across files, so a
run file carries only the knob it changes.

⚠️ The protocol fields (`PROTOCOL_FIELDS`) define how runs are SCORED. They
are reset to this file's defaults whenever a checkpoint is evaluated, so every
run -- trial 10, 12 or 13 -- is judged on identical sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import get_type_hints

import yaml

from ...config import REPO_ROOT, Config, _coerce

#how a word becomes a token: every word (the reference) or only scored ones
TOKEN_SETS = ("all", "invocab")

#how sequences are cut for scoring; see data/sequences.py
PROTOCOLS = ("reference", "fragment", "sparse", "isolated", "chains", "chains38")

#what the training set is cut by: whole sentences (shaped or not), or every
#word alone for the lone-word specialist
TRAINING_PROTOCOLS = ("reference", "isolated")

#the learning rate's decay after warmup; trial 23's anneal uses the second
SCHEDULES = ("cosine", "one_minus_sqrt")

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
class Trial18Config(Config):
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
        warmup_fraction (float): fraction of steps spent warming up; 0 is none
        min_lr (float): floor of the decay
        schedule (str): "cosine" or "one_minus_sqrt", the decay after warmup
        freeze_batchnorm (bool): BatchNorm in eval mode during training
        freeze_backbone (bool): the whole network in eval mode during training
        sam_rho (float): SAM's neighbourhood radius; 0 is plain AdamW
        ema_decay (float): 0 disables the parameter EMA
        matmul_precision (str): torch.set_float32_matmul_precision
        precision (str): Lightning's precision, "32", "bf16-mixed", "16-mixed"
        accelerator (str): "auto", "cuda" or "cpu"
        refine_base (str): Variant M -- a checkpoint, repo-relative, `{seed}`
            filled in; "" means no refiner
        init_encoder_from (str): trial 23's SSL -- a word encoder's state dict
            (cli.pretrain_encoder's encoder.pt), repo-relative, `{seed}`
            filled in; loaded before the fit, the pool left fresh. "" is none
        refine_layers (list): ConvSequence layers read, the gated outputs of
            HDND's stages 2 and 4
        refine_bins (list): time resolutions pooled
        refine_seconds (float): pooled from each window's start; 1 s, which
            every item has whole
        refine_width (int): the refiner's projected width
        refine_bound (float): c, the residual's norm limit
        refine_query_skip (bool): the paper's eta; off, so the correction
            comes from the word's own MEG
        refine_kl_weight (float): KL(base || refined) over the 50 words
        refine_pair_weight (float): the shuffled-feature pairing loss
        refine_strengths (list): the strengths compared on validation
        refine_calibration_batches (int): training batches the
            standardisation statistics come from
        refine_centre_momentum (float): the residual centre's update rate
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

    # --- trial 23: the anneal, head-only training, SAM ---
    schedule: str = "cosine"
    freeze_batchnorm: bool = False
    freeze_backbone: bool = False
    sam_rho: float = 0.0

    # --- trial 23's SSL: the encoder from masked-reconstruction pretraining ---
    init_encoder_from: str = ""

    # --- Variant M: HDND's refinement on a frozen checkpoint ---
    refine_base: str = ""
    refine_layers: list = field(default_factory=lambda: [3, 7])
    refine_bins: list = field(default_factory=lambda: [1, 2, 4])
    refine_seconds: float = 1.0
    refine_width: int = 128
    refine_bound: float = 0.25
    refine_query_skip: bool = False
    refine_kl_weight: float = 0.25
    refine_pair_weight: float = 0.3
    refine_strengths: list = field(default_factory=lambda: [0.0, 0.125, 0.25, 0.5, 1.0])
    refine_calibration_batches: int = 64
    refine_centre_momentum: float = 0.01

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

        #trial 23: the decay's shape, and SAM's two forward passes per step
        if self.schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.sam_rho < 0:
            raise ValueError(f"sam_rho must be >= 0, got {self.sam_rho}")
        #⚠️ Lightning's GradScaler steps without a closure, and SAM needs one
        if self.sam_rho > 0 and self.precision == "16-mixed":
            raise ValueError("sam_rho > 0 needs precision '32' or 'bf16-mixed'; 16-mixed steps without a closure.")

        #all-data: subject 0's own partitions, train always among them
        known = {"train", "validation", "test"}
        if "train" not in self.train_partitions or not set(self.train_partitions) <= known:
            raise ValueError(f"train_partitions must include 'train' and name only {sorted(known)}, got {self.train_partitions}")
        if self.n_subjects < 1:
            raise ValueError(f"n_subjects must be >= 1, got {self.n_subjects}")

        #Variant M: one head only, and a pooled span every item has whole
        if self.refine_base:
            if self.has_second_head:
                raise ValueError("refine_base with a second head: the refiner corrects one head only.")
            if not 0.0 < self.refine_seconds <= self.isolated_seconds:
                raise ValueError(
                    f"refine_seconds ({self.refine_seconds}) must be in (0, isolated_seconds "
                    f"({self.isolated_seconds})], the span every item has whole."
                )
            if self.refine_width <= 0 or self.refine_bound <= 0:
                raise ValueError(f"refine_width and refine_bound must be > 0, got {self.refine_width}, {self.refine_bound}")
            if min(self.refine_kl_weight, self.refine_pair_weight) < 0:
                raise ValueError("refine_kl_weight and refine_pair_weight must be >= 0.")
            if not self.refine_strengths or not all(0.0 <= value <= 1.0 for value in self.refine_strengths):
                raise ValueError(f"refine_strengths must be values in [0, 1], got {self.refine_strengths}")
            if not self.refine_layers or not self.refine_bins:
                raise ValueError("refine_layers and refine_bins must not be empty.")

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
    def has_refiner(self):
        """Whether this run carries Variant M's refiner."""
        return bool(self.refine_base)

    def resolved_init_encoder_from(self):
        """Absolute path to the pretrained encoder, this run's seed filled in; None without one."""
        return REPO_ROOT / self.init_encoder_from.format(seed=self.seed) if self.init_encoder_from else None

    def resolved_refine_base(self):
        """Absolute path to the frozen base checkpoint, this run's seed filled in."""
        return REPO_ROOT / self.refine_base.format(seed=self.seed)

    def refine_settings(self):
        """The refiner's construction arguments for ReferenceNet, or None.

        RETURNS:
            dict or None
        """
        if not self.has_refiner:
            return None
        return {
            "layers": list(self.refine_layers),
            "bins": list(self.refine_bins),
            "seconds": self.refine_seconds,
            "width": self.refine_width,
            "bound": self.refine_bound,
            "query_skip": self.refine_query_skip,
            "centre_momentum": self.refine_centre_momentum,
        }


#Variant M: the fields a refinement run sets itself; every other one is the base's
REFINE_RUN_FIELDS = (
    "run_name",
    "seed",
    "cache_dir",
    "refine_base",
    "refine_layers",
    "refine_bins",
    "refine_seconds",
    "refine_width",
    "refine_bound",
    "refine_query_skip",
    "refine_kl_weight",
    "refine_pair_weight",
    "refine_strengths",
    "refine_calibration_batches",
    "refine_centre_momentum",
    "lr",
    "min_lr",
    "weight_decay",
    "warmup_fraction",
    "max_epochs",
    "patience",
    "ema_decay",
    "batch_size",
    "num_workers",
    "precision",
    "matmul_precision",
    "accelerator",
    "torch_threads",
    "duty_cycle",
    "cooldown_every_minutes",
    "cooldown_seconds",
)


def refinement_config(run_cfg, base_cfg):
    """A refinement run's config: the base's settings, the run's own fields on top.

    ARGS:
        run_cfg (Trial18Config): from the run's YAML files
        base_cfg (Trial18Config): from the base checkpoint's hyperparameters

    RETURNS:
        Trial18Config - the base's architecture, data and shaping, unchanged
    """
    #a base that was itself refined would stack two refiners on one prediction
    if base_cfg.has_refiner:
        raise ValueError("The base checkpoint already carries a refiner.")
    merged = base_cfg.to_dict()
    for name in REFINE_RUN_FIELDS:
        merged[name] = getattr(run_cfg, name)
    return Trial18Config(**merged)


def load_trial18_config(*paths):
    """Load YAML run configs into a `Trial18Config`, merging model_kwargs.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial18Config
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
    types = get_type_hints(Trial18Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {name: _coerce(name, value, types[name]) for name, value in merged.items()}
    return Trial18Config(**coerced)


def with_protocol_defaults(cfg):
    """A copy of `cfg` whose protocol fields are this trial's defaults.

    A trial 10 checkpoint saved trial 10's protocol settings; scoring it under
    those would compare runs on different sequences.

    ARGS:
        cfg (Trial18Config): typically rebuilt from a checkpoint

    RETURNS:
        Trial18Config
    """
    defaults = Trial18Config()
    return replace(cfg, **{name: getattr(defaults, name) for name in PROTOCOL_FIELDS})
