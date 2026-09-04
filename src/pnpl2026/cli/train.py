"""Train one run.

    uv run python -m pnpl2026.cli.train --config configs/base.yaml configs/deep.yaml

Writes everything for the run under runs/<run_name>/: a snapshot of the merged
config, the git SHA, checkpoints and metrics. The SHA is what makes a
leaderboard score reproducible later.
"""

from __future__ import annotations

import argparse

from ..config import RUNS_DIR, load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    run_dir = RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    # TODO: snapshot cfg + git SHA into run_dir, build datamodule + LitWordClassifier,
    # then L.Trainer(...).fit(), checkpointing every epoch so a walltime kill is survivable.
    raise NotImplementedError


if __name__ == "__main__":
    main()
