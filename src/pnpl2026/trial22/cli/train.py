"""Train one trial-21 run, then score its best checkpoint on both partitions.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.train \\
        --config configs/base.yaml configs/trial22/trial22.yaml \\
                 configs/trial22/hpc.yaml configs/trial22/b_hybrid.yaml

Trial 21: with `init_encoder_from` set, the dascoli stem is filled from that
per-word checkpoint's encoder before the fit (`lit.warm_start_encoder`).

All-data training (`train_partitions` beyond "train", or `others_cache_dir`):
validation is no longer clean, so the run trains a fixed number of epochs --
`epochs_from`'s mean best epoch count, else `max_epochs` -- with no early
stopping, and best.ckpt is the last epoch. Its local scores are flagged.

The last file names the run and carries its changes; `configs/trial22/runs.txt`
is the queue and `info/variant-k.md` says what each run asks. Needs trial 10's
reference cache and all-word t5 table (`pnpl2026.trial10.cli.build_reference`
and `build_word_embeddings`); trial 17 builds neither.

Trains on pnpl's train partition (whole sentences, some reshaped like the
holdout when the run asks), selects the checkpoint on its validation partition
under `selection_protocol` -- the fragment protocol, val/bacc10 -- and when the
run ends scores best.ckpt on validation AND test under all three protocols,
with and without BatchNorm re-estimation -> evaluation.json, plus per-word
dumps. The test partition is touched there and nowhere else.

⚠️ A resume WIPES metrics.csv: Lightning's CSVLogger deletes previous log files
in a reused directory. The file is copied aside first.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from datetime import datetime

import lightning as L
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from ...config import REPO_ROOT, RUNS_DIR
from ..config import load_trial22_config
from ..data import Trial22DataModule
from ..data.dataset import load_manifest
from ..embeddings import TargetTable
from ..lit import LitTrialTwentyTwo, warm_start_encoder
from ..models.tower import load_descriptions
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


def all_data_settings(cfg):
    """All-data training: the subject count, and the fixed epoch count.

    ARGS:
        cfg (Trial22Config)

    RETURNS:
        Trial22Config - n_subjects from the others' cache; max_epochs from
        `epochs_from` when validation is not clean
    """
    #one subject layer per other subject, after subject 0's
    if cfg.others_cache_dir:
        subjects = load_manifest(cfg.resolved_others_cache_dir())["subjects"]
        cfg = replace(cfg, n_subjects=1 + len(subjects))
        print(f"[train] all-data: {len(subjects)} other subjects' recordings join training")
    if cfg.clean_selection:
        return cfg

    #⚠️ validation is in training, or heard by the others: no selection on it
    if cfg.epochs_from:
        files = sorted(glob.glob(str(REPO_ROOT / cfg.epochs_from)))
        if not files:
            raise SystemExit(f"epochs_from {cfg.epochs_from} matches no file.")
        best = []
        for name in files:
            with open(name, encoding="utf-8") as handle:
                best.append(int(json.load(handle)["epoch"]) + 1)
        cfg = replace(cfg, max_epochs=max(1, round(sum(best) / len(best))))
        print(f"[train] all-data: {len(files)} earlier runs stopped best at {best} epochs -> {cfg.max_epochs}")
    print(
        f"[train] WARNING: validation is not clean (train_partitions {list(cfg.train_partitions)}, "
        f"others {'yes' if cfg.others_cache_dir else 'no'}): a fixed {cfg.max_epochs} epochs, no early "
        f"stopping, best.ckpt is the LAST epoch, and local scores are not evidence"
    )
    return cfg


def main():
    """Build the data, the model and the trainer; fit; then score best.ckpt."""
    args = parse_args()
    cfg = load_trial22_config(*args.config)

    cfg = all_data_settings(cfg)
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
    datamodule = Trial22DataModule(cfg, table)
    datamodule.setup()

    #Variant K: the second head's table, which must list the same words in order
    second_table = None
    if cfg.has_second_head:
        second_table = TargetTable(cfg.resolved_second_embeddings(), centre=cfg.centre_embeddings)
        print(f"[train] second table: {cfg.second_embeddings}, {len(second_table.words):,} words x "
              f"{second_table.embed_dim}, loss weight {cfg.second_loss_weight:g}")
    #trial 19: the word tower's descriptions, in the t5 table's word order
    descriptions = None
    if cfg.has_tower:
        descriptions = load_descriptions(cfg.resolved_word_descriptions(), table.words)
        print(f"[train] word tower: {descriptions['table'].shape[1]:,} description numbers per word in blocks "
              + ", ".join(f"{name} {width}" for name, width in zip(descriptions["blocks"], descriptions["widths"]))
              + f" -> {cfg.tower_dim}; t5 anchor weight {cfg.tower_anchor_weight:g}")
    if cfg.has_phoneme_head:
        print(f"[train] phoneme head: labels from {cfg.phoneme_dir}, weight {cfg.phoneme_weight:g}, "
              f"lag {cfg.phoneme_lag_seconds:g} s")
    model = LitTrialTwentyTwo(cfg, table, second_table=second_table, descriptions=descriptions)

    #trial 21: the hybrid stem starts from a trained per-word encoder
    if cfg.has_warm_start:
        source = cfg.resolved_init_encoder_from()
        if not source.exists():
            raise SystemExit(f"init_encoder_from {source} does not exist.")
        summary = warm_start_encoder(model, source, with_head=cfg.init_head)
        print(f"[train] warm start from {source.parent.name}/{source.name}: {len(summary['loaded'])} tensors loaded"
              + (f", {len(summary['skipped'])} skipped" if summary["skipped"] else "")
              + (f"; MISSING {summary['missing']}" if summary["missing"] else ""))
        for key, reason in list(summary["skipped"].items())[:6]:
            print(f"[train]   skipped {key}: {reason}")


    last = run_dir / "last.ckpt"
    resuming = args.resume and last.exists()

    n_params = sum(p.numel() for p in model.model.parameters())
    n_time = sum(p.numel() for p in model.model.time.parameters())
    print(
        f"[train] {cfg.model}, {model.model.stem_kind} stem: {n_params:,} parameters, {100 * n_time / n_params:.1f}% in the time "
        f"transformer; one step per {model.model.stride} samples ({cfg.sfreq / model.model.stride:g} steps/s), readout "
        f"{model.model.readout_before}+{model.model.readout_after} steps around each onset"
    )
    print(
        f"[train] spatial {cfg.model_kwargs.get('spatial', 'merger')} | tokens {cfg.token_set} | "
        f"{'deduplicated' if cfg.siglip_dedup else 'plain'} SigLIP over every word | precision {cfg.precision}"
    )
    print(
        f"[train] shaping: {100 * cfg.fragment_p:.0f}% fragments, {100 * cfg.isolated_p:.0f}% isolated, {100 * cfg.run_p:.0f}% holdout runs | "
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
    selection = [best, EarlyStopping(monitor=monitor, mode="max", patience=cfg.patience, verbose=True)]

    #all-data: no selection on an unclean validation set; last.ckpt becomes
    #best.ckpt after the fit (Lightning refuses two monitor-less checkpointers)
    if not cfg.clean_selection:
        selection = []

    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=args.max_steps,
        accelerator=cfg.accelerator,
        devices=1,
        precision=cfg.precision,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        callbacks=[
            *selection,
            #⚠️ last.ckpt from its own callback, refreshed every epoch. On
            #Lightning 2.6 save_last on the callback above writes only on
            #improvement, so a resume after a plateau would go back in time
            ModelCheckpoint(dirpath=run_dir, filename="last", monitor=None, save_top_k=1),
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

    #all-data: the last epoch IS the model; best.ckpt holds it for every reader
    if not cfg.clean_selection and last.exists():
        shutil.copy2(last, run_dir / "best.ckpt")
        chosen = str(run_dir / "best.ckpt")

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
        bn_adapt=False,
    )

    #the train-val gap at the selected epoch, as the rules ask
    report["selected_epoch_metrics"] = epoch_metrics(run_dir, report["epoch"])
    if not cfg.clean_selection:
        report["selection"] = "fixed epochs, last kept: local validation and test are NOT clean"
    (run_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[train] scored {chosen} -> {run_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
