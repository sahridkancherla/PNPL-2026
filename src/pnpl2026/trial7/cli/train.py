"""Train one trial-7 run: one entry in the Variant D queue.

    PYTHONPATH=src python -m pnpl2026.trial7.cli.train \\
        --config configs/base.yaml configs/trial7/trial7.yaml \\
                 configs/trial7/hpc.yaml configs/trial7/d1_full.yaml

The last file names the run and carries its one change; see
`configs/trial7/runs.txt` for the queue and `info/variant-d.md` for what each
run asks. Needs trial 7's continuous cache -- `cli.build_continuous`.

Everything about the run lands in runs/<run_name>/: the merged config, the git
SHA, checkpoints and metrics. The SHA is what makes a leaderboard score
reproducible once there are forty runs and no memory of any of them.

⚠️ A resume WIPES metrics.csv. Lightning's CSVLogger deletes previous log files
in a reused directory -- it warns, then does it. This script copies the file
aside first, because the run's own record of its history is otherwise the one
thing a resume destroys. That lesson cost trial 3 three epochs of history.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from ...config import RUNS_DIR
from ..config import load_trial7_config
from ..data import Trial7DataModule
from ..lit import LitTrialSeven
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
        help="Continue from last.ckpt in the run directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a run directory that already holds results.",
    )
    return parser.parse_args()


def preserve_metrics(run_dir):
    """Copy metrics.csv aside before Lightning deletes it.

    ARGS:
        run_dir (Path): the run directory being reused
    """
    metrics = run_dir / "metrics.csv"
    if not metrics.exists():
        return

    #stamped, so repeated resumes do not overwrite each other's copies
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kept = run_dir / f"metrics.{stamp}.csv"
    shutil.copy2(metrics, kept)
    print(f"[train] preserved previous metrics -> {kept.name}")


def main():
    """Build the data, the model and the trainer, then fit."""
    args = parse_args()
    cfg = load_trial7_config(*args.config)
    L.seed_everything(cfg.seed, workers=True)

    #capping threads before any tensor work; torch reads this once and a later
    #change does not take effect
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
        print(f"[train] torch limited to {cfg.torch_threads} CPU threads")

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
    if args.resume or args.force:
        preserve_metrics(run_dir)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8"
    )
    (run_dir / "provenance.json").write_text(
        json.dumps({"git_sha": git_sha(), "configs": list(args.config)}, indent=2),
        encoding="utf-8",
    )

    #both early stopping and checkpoint selection follow the SCORED metric, not
    #the loss -- three weighted losses and a balanced metric do not move together
    monitor = f"val/bacc{cfg.metric_k}"

    datamodule = Trial7DataModule(cfg)
    datamodule.setup()
    model = LitTrialSeven(cfg, class_weights=datamodule.class_weights)

    last = run_dir / "last.ckpt"
    resuming = args.resume and last.exists()

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[train] {cfg.model} with {n_params:,} parameters")
    print(f"[train] monitoring {monitor}, chance is {cfg.metric_k / cfg.n_classes:.2f}")
    print(
        f"[train] losses: semantic {cfg.semantic_weight} "
        f"({'deduplicated' if cfg.siglip_dedup else 'plain'} SigLIP) | closed-set "
        f"{cfg.closed_set_weight} | word masking {cfg.word_mask_p}"
        f"{' | D8: every word predicted with itself masked' if cfg.mask_target else ''}"
    )

    #pacing is a no-op at duty_cycle 1.0, so the callback is always attached and
    #the config alone decides whether it does anything
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
            #best.ckpt: the epoch with the highest SCORED metric, EMA weights
            ModelCheckpoint(
                dirpath=run_dir,
                filename="best",
                monitor=monitor,
                mode="max",
            ),
            #⚠️ last.ckpt from a callback of its own, refreshed EVERY epoch. On
            #Lightning 2.6, save_last=True on the callback above only writes when
            #the monitored metric improves -- so a run that plateaus stops
            #refreshing it, and --resume after a walltime kill silently goes back
            #to the last improvement. Trial 5 has that bug
            ModelCheckpoint(
                dirpath=run_dir,
                filename="last",
                monitor=None,
                save_top_k=1,
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

    ckpt_path = str(last) if resuming else None
    if args.resume and ckpt_path is None:
        print(f"[train] --resume given but {last} not found; starting fresh")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"[train] complete -> {run_dir}")


if __name__ == "__main__":
    main()
