"""Train one trial-1 run.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED: `lightning` was not installed on
the machine this was written on, so not one line of it has executed.

    PYTHONPATH=src python -m pnpl2026.trial1.cli.train \
        --config configs/base.yaml configs/trial1.yaml

Everything about the run lands in runs/<run_name>/: the merged config, the git
SHA, checkpoints and metrics. The SHA is what makes a leaderboard score
reproducible once there are forty runs and no memory of any of them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from ...config import RUNS_DIR
from ..config import load_trial1_config
from ..data import Trial1DataModule
from ..lit import LitTrialOne
from ..throttle import ThermalThrottle


def git_sha():
    """The current commit, or "unknown" outside a repo.

    RETURNS:
        str: the full SHA
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def parse_args():
    """Command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last.ckpt in the run directory (after a walltime kill).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a run directory that already holds results.",
    )
    return parser.parse_args()


def main():
    """Build the data, the model and the trainer, then fit."""
    args = parse_args()
    cfg = load_trial1_config(*args.config)
    L.seed_everything(cfg.seed, workers=True)

    # --- AI, 2026-09-24 ---
    #capping threads before any tensor work; torch reads this once and a later
    #change does not take effect
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
        print(f"[train] torch limited to {cfg.torch_threads} CPU threads")
    # --- end AI ---

    run_dir = RUNS_DIR / cfg.run_name

    #Lightning deletes prior logs and checkpoints in a reused directory, and
    #losing the record of an earlier score is worse than an early exit
    existing = list(run_dir.glob("*.ckpt")) + list(run_dir.glob("metrics.csv"))
    if existing and not args.force and not args.resume:
        raise SystemExit(
            f"{run_dir} already holds results ({len(existing)} file(s)). "
            f"Change run_name in your config, or pass --force to overwrite."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"git_sha": git_sha(), "configs": list(args.config)}, indent=2),
        encoding="utf-8",
    )

    #both early stopping and checkpoint selection follow the SCORED metric,
    #not the loss -- a weighted loss and a balanced metric do not move together
    monitor = f"val/bacc{cfg.metric_k}"

    datamodule = Trial1DataModule(cfg)
    datamodule.setup()
    model = LitTrialOne(cfg, class_weights=datamodule.class_weights)

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[train] {cfg.model} with {n_params:,} parameters")
    print(f"[train] monitoring {monitor}, chance is {cfg.metric_k / cfg.n_classes:.2f}")

    #pacing is a no-op at duty_cycle 1.0, so the callback is always attached
    #and the config alone decides whether it does anything
    throttle = ThermalThrottle(
        duty_cycle=cfg.duty_cycle,
        cooldown_every_minutes=cfg.cooldown_every_minutes,
        cooldown_seconds=cfg.cooldown_seconds,
    )

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        callbacks=[
            #checkpoint every epoch: a walltime kill with no checkpoint loses
            #the whole job, and last.ckpt is what --resume picks up
            ModelCheckpoint(
                dirpath=run_dir,
                filename="best",
                monitor=monitor,
                mode="max",
                save_last=True,
            ),
            EarlyStopping(
                monitor=monitor, mode="max", patience=cfg.patience, verbose=True
            ),
            #off a terminal tqdm cannot redraw in place, so every refresh is a
            #new line in the job log; throttle it when output is redirected
            TQDMProgressBar(refresh_rate=1 if sys.stdout.isatty() else 200),
            throttle,
        ],
    )

    last = run_dir / "last.ckpt"
    ckpt_path = str(last) if args.resume and last.exists() else None
    if args.resume and ckpt_path is None:
        print(f"[train] --resume given but {last} not found; starting fresh")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"[train] complete -> {run_dir}")


if __name__ == "__main__":
    main()
