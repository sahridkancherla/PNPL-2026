"""Train one trial-13 run, then score its best checkpoint on both partitions.

    PYTHONPATH=src python -m pnpl2026.trial13.cli.train \\
        --config configs/base.yaml configs/trial13/trial13.yaml \\
                 configs/trial13/hpc.yaml configs/trial13/g3_both.yaml

The last file names the run and carries its one change; `configs/trial13/runs.txt`
is the queue and `info/variant-g.md` says what each run asks. Needs trial 10's
reference cache and all-word t5 table (`pnpl2026.trial10.cli.build_reference`
and `build_word_embeddings`); trial 13 builds neither.

Trains on pnpl's train partition (whole sentences, some reshaped like the
holdout when the run asks), selects the checkpoint on its validation partition
under `selection_protocol` -- the fragment protocol, val/bacc10 -- and when the
run ends scores best.ckpt on validation AND test under all three protocols,
with and without BatchNorm re-estimation -> evaluation.json, plus per-word
dumps for `cli.calibrate`. The test partition is touched there and nowhere else.

⚠️ A resume WIPES metrics.csv: Lightning's CSVLogger deletes previous log files
in a reused directory. The file is copied aside first.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
from datetime import datetime

import lightning as L
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from ...config import RUNS_DIR
from ..config import load_trial13_config
from ..data import Trial13DataModule
from ..embeddings import TargetTable
from ..lit import LitTrialThirteen
from ..throttle import ThermalThrottle
from .evaluate import evaluate_checkpoint


def git_sha():
    """The current commit, or "unknown" outside a repo.

    RETURNS:
        str: the full SHA
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def parse_args():
    """Command-line arguments for a training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--resume", action="store_true", help="Continue from last.ckpt in the run directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite a run directory that holds results.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Stop after this many optimiser steps; smoke tests and timing only.",
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


def epoch_metrics(run_dir, epoch):
    """Every metric logged for one epoch, train and val side by side.

    ARGS:
        run_dir (Path): holds metrics.csv
        epoch (int or None)

    RETURNS:
        dict: metric name -> value, empty when nothing was logged
    """
    path = run_dir / "metrics.csv"
    if epoch is None or not path.exists():
        return {}

    #rows are sparse: each logging call fills its own columns. Take the last
    #non-empty value of each column within the epoch
    table = pd.read_csv(path)
    rows = table[table["epoch"] == epoch]
    found = {}
    for column in rows.columns:
        values = rows[column].dropna()
        if column not in ("epoch", "step") and len(values):
            found[column] = float(values.iloc[-1])
    return found


def main():
    """Build the data, the model and the trainer; fit; then score best.ckpt."""
    args = parse_args()
    cfg = load_trial13_config(*args.config)
    L.seed_everything(cfg.seed, workers=True)

    #threads capped before any tensor work; torch reads this once
    if cfg.torch_threads > 0:
        torch.set_num_threads(cfg.torch_threads)
        print(f"[train] torch limited to {cfg.torch_threads} CPU threads")
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    run_dir = RUNS_DIR / cfg.run_name

    #Lightning deletes prior logs and checkpoints in a reused directory
    existing = list(run_dir.glob("*.ckpt")) + list(run_dir.glob("metrics.csv"))
    if existing and not args.force and not args.resume:
        raise SystemExit(
            f"{run_dir} already holds results ({len(existing)} file(s)). "
            f"Change run_name in your config, or pass --force to overwrite."
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume or args.force:
        preserve_metrics(run_dir)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    (run_dir / "provenance.json").write_text(
        json.dumps({"git_sha": git_sha(), "configs": list(args.config)}, indent=2), encoding="utf-8"
    )

    #checkpoint selection and early stopping both follow the scored metric
    monitor = f"val/bacc{cfg.metric_k}"

    table = TargetTable(cfg.resolved_word_embeddings(), centre=cfg.centre_embeddings)
    print(f"[train] t5 table: {len(table.words):,} words x {table.embed_dim}")
    datamodule = Trial13DataModule(cfg, table)
    datamodule.setup()
    model = LitTrialThirteen(cfg, table)

    last = run_dir / "last.ckpt"
    resuming = args.resume and last.exists()

    n_params = sum(p.numel() for p in model.model.parameters())
    n_sentence = sum(p.numel() for p in model.model.sentence.parameters()) if model.model.sentence else 0
    print(
        f"[train] {cfg.model}: {n_params:,} parameters, {100 * n_sentence / n_params:.1f}% in the "
        f"sentence transformer (the paper's: 203.7M, 98.8%)"
    )
    print(
        f"[train] spatial {cfg.model_kwargs.get('spatial', 'merger')} | tokens {cfg.token_set} | "
        f"{'deduplicated' if cfg.siglip_dedup else 'plain'} SigLIP over every word | precision {cfg.precision}"
    )
    print(
        f"[train] shaping: {100 * cfg.fragment_p:.0f}% fragments, {100 * cfg.isolated_p:.0f}% isolated | "
        f"monitoring {monitor} on the {cfg.selection_protocol} protocol, chance is {cfg.metric_k / cfg.n_classes:.2f}"
    )

    #pacing is a no-op at duty_cycle 1.0; the config alone decides
    throttle = ThermalThrottle(
        duty_cycle=cfg.duty_cycle,
        cooldown_every_minutes=cfg.cooldown_every_minutes,
        cooldown_seconds=cfg.cooldown_seconds,
    )
    #weights only: at 204M parameters the optimiser state is another 1.6 GB per
    #file, and only last.ckpt -- the one --resume reads -- needs it
    best = ModelCheckpoint(dirpath=run_dir, filename="best", monitor=monitor, mode="max", save_weights_only=True)

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=args.max_steps,
        accelerator=cfg.accelerator,
        devices=1,
        precision=cfg.precision,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        callbacks=[
            best,
            #⚠️ last.ckpt from its own callback, refreshed every epoch. On
            #Lightning 2.6 save_last on the callback above writes only on
            #improvement, so a resume after a plateau would go back in time
            ModelCheckpoint(dirpath=run_dir, filename="last", monitor=None, save_top_k=1),
            EarlyStopping(monitor=monitor, mode="max", patience=cfg.patience, verbose=True),
            #off a terminal every refresh is a new log line; throttle it there
            TQDMProgressBar(refresh_rate=1 if sys.stdout.isatty() else 200),
            throttle,
        ],
    )

    ckpt_path = str(last) if resuming else None
    if args.resume and ckpt_path is None:
        print(f"[train] --resume given but {last} not found; starting fresh")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    print(f"[train] fit complete -> {run_dir}")

    #best.ckpt when validation ever ran; last.ckpt otherwise (a --max-steps smoke)
    chosen = best.best_model_path or str(last)
    del trainer, model, datamodule, table
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    report = evaluate_checkpoint(
        chosen,
        cache_dir=cfg.cache_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device="cuda" if cfg.accelerator != "cpu" and torch.cuda.is_available() else "cpu",
        dump_dir=run_dir / "dumps",
        bn_adapt=True,
    )

    #the train-val gap at the selected epoch, as the rules ask
    report["selected_epoch_metrics"] = epoch_metrics(run_dir, report["epoch"])
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[train] scored {chosen} -> {run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
