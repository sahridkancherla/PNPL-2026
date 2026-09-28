"""Trial-1 run config: the shared fields, plus what only this trial has.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.

Subclasses the shared `Config` rather than replacing it, so `lr`, `batch_size`
and the rest keep one definition and one coercion path. Only the fields this
trial actually introduces live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import REPO_ROOT, Config, load_config


@dataclass
class Trial1Config(Config):
    """Everything trial 1 needs that the shared config does not carry.

    ATTRS:
        cache_dir (str): shard directory, RELATIVE to the repo root. A relative
            path is the point: an absolute one is broken on the next machine
        n_components (int): whitener rank the cache was built at
        window_bounds (list): cached window in seconds, wider than the scored
            one so there is margin for jitter to slide into
        clip_sigma (float): clip threshold applied before the float16 cast
        jitter_seconds (float): one-sided time jitter, training only
        class_weight_power (float): 0 unweighted, 0.5 sqrt-inverse, 1 inverse
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

    # --- AI, 2026-09-24: laptop thermal budget. See ../throttle.py. ---
    #"auto" picks MPS on Apple Silicon; "cpu" forces the slower, cooler path
    accelerator: str = "auto"

    #0 leaves torch's default (all performance cores). On a 6+6 M-series chip,
    #4 keeps two performance cores free for everything else on the machine
    torch_threads: int = 0

    #1.0 runs flat out. 0.6 means 60% computing, 40% idling -- and 1.7x the
    #wall-clock. There is no way to cap GPU utilisation directly on Apple
    #Silicon, so duty cycle is the lever that actually bounds heat
    duty_cycle: float = 1.0

    #a longer pause lets the chassis shed heat, which short inter-batch gaps
    #do not. 0 disables it
    cooldown_every_minutes: float = 0.0
    cooldown_seconds: float = 0.0
    # --- end AI ---

    def __post_init__(self):
        """Catch the config mistakes that would otherwise train to chance."""
        #the model's input width IS the whitener rank; if these disagree the
        #first conv silently reads the wrong number of components
        if self.n_channels != self.n_components:
            raise ValueError(
                f"n_channels ({self.n_channels}) must equal n_components "
                f"({self.n_components}) -- trial 1 feeds the model whitened "
                f"components, not the 306 sensors. Set n_channels: "
                f"{self.n_components} in your config."
            )

        #a scored window outside the cached bounds has nothing to crop from
        low, high = self.window_bounds
        if self.tmin < low or self.tmax > high:
            raise ValueError(
                f"Scored window ({self.tmin}, {self.tmax}) s is not inside the "
                f"cached ({low}, {high}) s."
            )

    def resolved_cache_dir(self):
        """Absolute path to the cache, derived from the repo root.

        RETURNS:
            Path to the shard directory
        """
        #one machine's absolute path is broken on the next, so it is built here
        return REPO_ROOT / self.cache_dir


def load_trial1_config(*paths):
    """Load YAML run configs into a `Trial1Config`.

    ARGS:
        *paths: YAML files, merged left to right

    RETURNS:
        a validated Trial1Config
    """
    return load_config(*paths, config_class=Trial1Config)
