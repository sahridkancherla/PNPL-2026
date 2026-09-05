"""Train one run.

    PYTHONPATH=src uv run python -m pnpl2026.cli.train \
        --config configs/base.yaml configs/deep.yaml

Everything about the run lands in runs/<run_name>/: the merged config, the git
SHA, checkpoints and metrics. The SHA is what makes a leaderboard score
reproducible once you have forty runs and no memory of any of them.
"""

from __future__ import annotations

import argparse
import json
import subprocess

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..config import RUNS_DIR, load_config
from ..data.datamodule import LibriBrainDataModule
from ..lit import LitWordClassifier


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", nargs="+", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(*args.config)
    L.seed_everything(cfg.seed, workers=True)

    run_dir = RUNS_DIR / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"git_sha": git_sha(), "configs": list(args.config)}, indent=2),
        encoding="utf-8",
    )

    dm = LibriBrainDataModule(cfg)
    dm.setup()
    model = LitWordClassifier(cfg, class_weights=dm.class_weights)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        callbacks=[
            # Checkpoint every epoch: a walltime kill with no checkpoint loses
            # the whole job.
            ModelCheckpoint(
                dirpath=run_dir,
                filename="best",
                monitor="val/balanced_acc",
                mode="max",
                save_last=True,
            )
        ],
    )
    trainer.fit(model, datamodule=dm)
    print(f"run complete -> {run_dir}")


if __name__ == "__main__":
    main()
