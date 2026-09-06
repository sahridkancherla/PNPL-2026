"""Run configuration and canonical project paths."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, get_type_hints

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
    metric_k: int = 10
    val_sessions: list[str] = field(default_factory=list)
    val_fraction: float = 0.1

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


def _coerce(name: str, value: Any, target: Any) -> Any:
    """Cast a YAML scalar to the dataclass field's declared type.

    PyYAML implements YAML 1.1, where scientific notation needs a decimal point
    and a signed exponent: `1e-6` loads as the *string* "1e-6" while `1.0e-6`
    loads as a float. Without this, a learning rate written the obvious way
    reaches the optimiser as text and fails deep inside torch. Containers are
    left alone.
    """
    if target in (int, float, str, bool) and not isinstance(value, target):
        try:
            return target(value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                f"Config field {name!r} = {value!r} cannot be read as {target.__name__}"
            ) from e
    return value


def load_config(*paths: str | Path) -> Config:
    """Load YAML files left-to-right, later files overriding earlier ones."""
    merged: dict[str, Any] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            merged.update(yaml.safe_load(f) or {})

    # `from __future__ import annotations` makes field.type a string, so
    # resolve the real types rather than comparing against 'float'.
    types = get_type_hints(Config)
    unknown = set(merged) - set(types)
    if unknown:
        raise KeyError(f"Unknown config key(s): {sorted(unknown)}")

    coerced = {k: _coerce(k, v, types[k]) for k, v in merged.items()}
    return Config(**coerced)
