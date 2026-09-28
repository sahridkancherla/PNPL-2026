"""Trial-5 run config: the shared fields, plus what only this trial has.

Subclasses the shared `Config` so `lr`, `batch_size` and the rest keep one
definition and one coercion path. Only the fields trial 5 introduces live here.

⚠️ Trial 5 reads trial 1's cache (73 whitened components, 400-sample windows).
That is a compromise forced by the external drive being unmounted: the design
document calls for raw 306 sensors and a learned spatial front end, which needs
the continuous recordings. See `models/net.py` for what variant A would
replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import REPO_ROOT, Config, load_config


@dataclass
class Trial5Config(Config):
    """Everything trial 5 needs that the shared config does not carry.

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
        warmup_fraction (float): fraction of total steps spent warming up
        min_lr (float): floor the cosine schedule decays to
        ema_decay (float): 0 disables the weight EMA
        time_mask_spans (int): how many spans of time to zero per window
        time_mask_width (int): maximum width of one span, in input samples
        component_dropout (float): probability of zeroing a whole component
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

    # --- schedule ---
    warmup_fraction: float = 0.05
    min_lr: float = 1.0e-5
    ema_decay: float = 0.999

    # --- augmentation, applied on the batch inside the LightningModule ---
    time_mask_spans: int = 2
    time_mask_width: int = 40
    component_dropout: float = 0.1

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
                f"({self.n_components}) -- trial 5 reads whitened components, "
                f"not the 306 sensors. Set n_channels: {self.n_components}."
            )

        #a scored window outside the cached bounds has nothing to crop from
        low, high = self.window_bounds
        if self.tmin < low or self.tmax > high:
            raise ValueError(
                f"Scored window ({self.tmin}, {self.tmax}) s is not inside the "
                f"cached ({low}, {high}) s."
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


def load_trial5_config(*paths):
    """Load YAML run configs into a `Trial5Config`.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial5Config
    """
    return load_config(*paths, config_class=Trial5Config)
