"""Run configuration and canonical project paths."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"
SUBMISSIONS_DIR = REPO_ROOT / "submissions"


@dataclass
class Config:
    # data
    track: str = "deep"
    tmin: float = 0.0
    tmax: float = 1.0
    sfreq: int = 250
    n_channels: int = 306
    n_classes: int = 50
    val_sessions: list[str] = field(default_factory=list)

    # model
    model: str = "cnn_baseline"
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    # optim
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 50
    patience: int = 5
    num_workers: int = 4
    seed: int = 0

    run_name: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(*paths: str | Path) -> Config:
    """Load YAML files left-to-right, later files overriding earlier ones."""
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            merged.update(yaml.safe_load(f) or {})
    return Config(**merged)
