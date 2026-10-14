"""Anneal a trained per-word member, or train a new head on its frozen backbone.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.anneal --parent runs/trial13_g2_s0/best.ckpt \\
        --run-name trial23_an_g2_s0 --mode anneal [--sam]
    PYTHONPATH=src python -m pnpl2026.trial23.cli.anneal --parent runs/trial13_g2_s0/best.ckpt \\
        --run-name trial23_hg_g2_s0 --mode head --head data/word_embeddings_gpt2medium_layer12.npz

The final-night note's Recipes 1, 2 and 4
(handoff-docs/final-night-note-2026-10-14.md; trial 23's
info/final-night-14oct.md says what was built and why).

--mode anneal (Recipe 1; Recipe 4 with --sam):

    start          the parent's weights, averaged (EMA) ones when it has them
    learning rate  the parent's x --lr-scale (0.1), no warmup, decaying as
                   1 - sqrt(progress) to 0 over --epochs (4)
    data           the parent's, holdout-shaped items -- G1's fragments plus
                   the parent's own isolated share -- --target-share (0.5) of
                   every epoch
    BatchNorm      frozen on the parent's running statistics
    weights        the last --average-last (3) epochs averaged into best.ckpt
                   (no EMA: the average replaces it)
    --sam          SAM (rho --sam-rho, 0.05) around AdamW; at precision "32",
                   since fp16's GradScaler steps without the closure SAM needs

--mode head (Recipe 2): Variant K's second head on the frozen parent, against
--head's table (`cli.build_lm_embeddings`: another language model's middle
layer). Only the new Linear and its SigLIP scale and bias train, lr --lr
(1e-3), cosine to 0 over --epochs (5), no warmup, the same holdout-shaped
mix; every other weight, BatchNorm and dropout as at inference. The member
then ranks by both heads combined, as K2 does.

Afterwards, scored with this trial's per-word evaluation (`perword.evaluation`):

    best.ckpt   reference and fragment protocols, validation and test, with
                dumps -> evaluation.json, dumps/
    the last    the final epoch, un-averaged, as a second member:
    epoch       runs/<fallback>/best.ckpt, scored the same way. <fallback> is
                the run name with "last" joined to its second word
                (trial23_an_g2_s0 -> trial23_anlast_g2_s0), so a prefix that
                picks the averaged members never picks it
    the parent  the fragment protocol, for the comparison

anneal.json holds the settings and the sanity verdict: a member within
--margin (1) BAcc@10 point of its parent on fragments (the mean of validation
and test) is kept, one further below is not. The epoch count is fixed, so
nothing is selected on validation; test labels reach only the reports.

`pbs/trial23_anneal.pbs` runs one line of configs/trial23/anneal.txt per array
task, then predicts both members.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import lightning as L
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger

from ...config import RUNS_DIR
from ..perword.data import Trial18DataModule
from ..perword.embeddings import TargetTable
from ..perword.evaluation import evaluate_checkpoint
from ..perword.lit import LitTrialEighteen, load_trained

#what the new head adds to a parent: its map, its SigLIP scale and bias
HEAD_PARAMETERS = ("model.project_second.weight", "model.project_second.bias", "log_scale_second", "logit_bias_second")

#the new head's candidate tables, filled from --head's table rather than the parent
HEAD_BUFFERS = ("primary_table_second", "moses_table_second")

#each mode's default epoch count: ~20% of the parents' 20-24 trained epochs,
#and a little more for a head that starts from nothing
DEFAULT_EPOCHS = {"anneal": 4, "head": 5}

#the per-epoch checkpoints' names, and the pattern that reads their epoch back
EPOCH_FILE = "epoch{epoch:02d}"
EPOCH_PATTERN = re.compile(r"epoch(\d+)\.ckpt$")


def parse_args():
    """Command-line arguments for the anneal and the head-only training."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True, help="the parent's best.ckpt (per-word family)")
    parser.add_argument("--run-name", required=True, help="runs/<run-name>/ receives best.ckpt")
    parser.add_argument("--mode", required=True, choices=("anneal", "head"))
    parser.add_argument("--epochs", type=int, default=None, help="default 4 (anneal) or 5 (head)")
    parser.add_argument("--lr-scale", type=float, default=0.1, help="anneal: the parent's learning rate times this")
    parser.add_argument("--lr", type=float, default=1e-3, help="head: the new head's learning rate")
    parser.add_argument("--target-share", type=float, default=0.5,
                        help="holdout-shaped items' share of every epoch: fragments fill what the parent's own shaping leaves")
    parser.add_argument("--average-last", type=int, default=3, help="the last this many epochs averaged into best.ckpt")
    parser.add_argument("--sam", action="store_true", help="anneal: SAM around AdamW, at precision 32")
    parser.add_argument("--sam-rho", type=float, default=0.05)
    parser.add_argument("--head", default=None, help="head: the new head's target table, repo-relative")
    parser.add_argument("--second-loss-weight", type=float, default=1.0, help="head: the new head's loss weight")
    parser.add_argument("--margin", type=float, default=1.0, help="BAcc@10 points below the parent a member may fall")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None, help="override the run's cache_dir (a staged copy)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--accelerator", default=None, help="override the parent's ('cuda' on CX3)")
    parser.add_argument("--runs-dir", default=None, help="default runs/ at the repo root; tests point elsewhere")
    parser.add_argument("--max-steps", type=int, default=-1, help="a smoke run's cap; -1 trains every epoch")
    parser.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    return parser.parse_args()


def fallback_name(run_name):
    """The un-averaged last epoch's run name: "last" joined to the second word.

    ARGS:
        run_name (str): e.g. "trial23_an_g2_s0"

    RETURNS:
        str: e.g. "trial23_anlast_g2_s0"; "<run_name>_last" for a short name
    """
    parts = run_name.split("_")
    if len(parts) < 3:
        return f"{run_name}_last"
    parts[1] += "last"
    return "_".join(parts)


def child_config(parent_cfg, args):
    """The parent's config with the anneal's or the head's changes.

    ARGS:
        parent_cfg (Trial18Config): from the parent checkpoint
        args (argparse.Namespace)

    RETURNS:
        Trial18Config
    """
    #fragments fill the target share beside the parent's own isolated and run items
    fragment_p = args.target_share - parent_cfg.isolated_p - parent_cfg.run_p
    if fragment_p < 0:
        raise SystemExit(f"The parent already shapes {100 * (parent_cfg.isolated_p + parent_cfg.run_p):.0f}% of items, "
                         f"above --target-share {args.target_share:g}.")
    changes = {
        "run_name": args.run_name,
        "seed": args.seed,
        "max_epochs": args.epochs or DEFAULT_EPOCHS[args.mode],
        "fragment_p": fragment_p,
        "warmup_fraction": 0.0,
        "min_lr": 0.0,
        "freeze_batchnorm": True,
        "ema_decay": 0.0,
        "refine_base": "",
    }
    if args.mode == "anneal":
        changes.update(lr=parent_cfg.lr * args.lr_scale, schedule="one_minus_sqrt",
                       sam_rho=args.sam_rho if args.sam else 0.0,
                       precision="32" if args.sam else parent_cfg.precision)
    else:
        changes.update(lr=args.lr, schedule="cosine", freeze_backbone=True, sam_rho=0.0, second_embeddings=args.head,
                       second_loss_weight=args.second_loss_weight)
    if args.cache_dir:
        changes["cache_dir"] = args.cache_dir
    if args.num_workers is not None:
        changes["num_workers"] = args.num_workers
    if args.accelerator:
        changes["accelerator"] = args.accelerator
    return replace(parent_cfg, **changes)


def check_arguments(args, parent_cfg):
    """Refuse the combinations that would train something else than asked.

    ARGS:
        args (argparse.Namespace)
        parent_cfg (Trial18Config)
    """
    if args.mode == "head":
        if not args.head:
            raise SystemExit("--mode head needs --head <table.npz>, from cli.build_lm_embeddings.")
        if args.sam:
            raise SystemExit("--sam applies to --mode anneal; a single Linear has no sharpness worth the second pass.")
        #K2-style parents already carry a second head; this adds one
        if parent_cfg.has_second_head:
            raise SystemExit(f"The parent already has a second head ({parent_cfg.second_embeddings}).")
    elif args.head:
        raise SystemExit("--head applies to --mode head; an anneal keeps the parent's heads.")
    if parent_cfg.has_refiner:
        raise SystemExit("The parent carries Variant M's refiner; anneal its base instead.")
    if args.average_last < 1:
        raise SystemExit(f"--average-last must be >= 1, got {args.average_last}")


def build_child(cfg, parent, mode):
    """The child module with the parent's weights in, and what trains switched on.

    ARGS:
        cfg (Trial18Config): from `child_config`
        parent (LitTrialEighteen): the loaded parent, averaged weights applied
        mode (str): "anneal" or "head"

    RETURNS:
        model : LitTrialEighteen
        table : TargetTable - the first head's, which the data module labels with
    """
    table = TargetTable(cfg.resolved_word_embeddings(), centre=cfg.centre_embeddings)
    second_table = TargetTable(cfg.resolved_second_embeddings(), centre=cfg.centre_embeddings) if cfg.has_second_head else None
    model = LitTrialEighteen(cfg, table, second_table=second_table)

    #the anneal is the parent's network exactly; the head mode adds the new head alone
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    allowed = set(HEAD_PARAMETERS + HEAD_BUFFERS) if mode == "head" else set()
    stray = [name for name in missing if name not in allowed]
    if unexpected or stray:
        raise SystemExit(f"The parent does not fit the child: unexpected {unexpected[:5]}, missing {stray[:5]}")

    #head mode: every parameter but the new head's stays the parent's
    if mode == "head":
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name in HEAD_PARAMETERS)
    n_trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[anneal] {n_trainable:,} trainable parameters of {sum(p.numel() for p in model.parameters()):,}")

    #⚠️ the frozen modes now: Lightning keeps whatever modes fit starts with
    model.train()
    frozen = [layer for layer in model.modules() if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)]
    if any(layer.training for layer in frozen):
        raise SystemExit("A BatchNorm layer is still in training mode; the anneal would move its statistics.")
    return model, table


def epoch_of(path):
    """The epoch number in a per-epoch checkpoint's name.

    ARGS:
        path (Path): e.g. runs/x/epoch03.ckpt

    RETURNS:
        int
    """
    return int(EPOCH_PATTERN.search(path.name).group(1))


def average_checkpoints(paths, parameter_names):
    """Several epochs' checkpoints averaged: parameters meaned, buffers the last one's.

    BatchNorm is frozen in every anneal, so the buffers are the parent's in
    every epoch; the last one's metadata (config, epoch) is kept.

    ARGS:
        paths (list): checkpoints, oldest first
        parameter_names (set): state-dict keys that are parameters

    RETURNS:
        dict: a checkpoint `load_trained` reads, plus "averaged_epochs"
    """
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    averaged = dict(checkpoints[-1])
    state = dict(checkpoints[-1]["state_dict"])
    for name in parameter_names:
        #float64 for the mean; dim=0 is the epoch axis
        stacked = torch.stack([checkpoint["state_dict"][name].double() for checkpoint in checkpoints], dim=0)
        state[name] = stacked.mean(dim=0).to(state[name].dtype)
    averaged["state_dict"] = state
    averaged["averaged_epochs"] = [int(checkpoint["epoch"]) for checkpoint in checkpoints]
    return averaged


def last_epoch_metrics(run_dir):
    """The last epoch's logged metrics, train and validation side by side.

    ARGS:
        run_dir (Path): holds metrics.csv

    RETURNS:
        dict: metric name -> value; empty when nothing was logged
    """
    path = run_dir / "metrics.csv"
    if not path.exists():
        return {}

    #rows are sparse: each column's last value within the last epoch
    table = pd.read_csv(path)
    rows = table[table["epoch"] == table["epoch"].max()]
    found = {}
    for column in rows.columns:
        values = rows[column].dropna()
        if column not in ("epoch", "step") and len(values):
            found[column] = float(values.iloc[-1])
    return found


def fragment_scores(report, k):
    """BAcc@k on the fragment protocol, validation, test and their mean.

    ARGS:
        report (dict): from `evaluate_checkpoint`, with the fragment protocol
        k (int)

    RETURNS:
        dict
    """
    results = report["results"]["fragment"]
    validation, test = results["validation"][f"bacc{k}"], results["test"][f"bacc{k}"]
    return {"validation": validation, "test": test, "mean": 0.5 * (validation + test)}


def train(cfg, model, table, run_dir, max_steps):
    """Fit for the fixed epoch count, one weights-only checkpoint per epoch.

    ARGS:
        cfg (Trial18Config)
        model (LitTrialEighteen)
        table (TargetTable)
        run_dir (Path)
        max_steps (int): -1 for every epoch

    RETURNS:
        list of Path: the epochs' checkpoints, oldest first
    """
    datamodule = Trial18DataModule(cfg, table)
    datamodule.setup()

    #every epoch kept, written at its end (a smoke run's cut epoch included)
    every_epoch = ModelCheckpoint(dirpath=run_dir, filename=EPOCH_FILE, auto_insert_metric_name=False, monitor=None,
                                  save_top_k=-1, every_n_epochs=1, save_weights_only=True, save_on_train_epoch_end=True)
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=max_steps,
        accelerator=cfg.accelerator,
        devices=1,
        precision=cfg.precision,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        #the rate in metrics.csv, so the decay can be read off the record
        callbacks=[every_epoch, LearningRateMonitor(logging_interval="step"),
                   TQDMProgressBar(refresh_rate=1 if sys.stdout.isatty() else 200)],
    )
    trainer.fit(model, datamodule=datamodule)
    return sorted(run_dir.glob("epoch*.ckpt"), key=epoch_of)


def main():
    """Load the parent, train the child, average, score both members and the parent."""
    args = parse_args()
    parent, parent_cfg, info = load_trained(args.parent, device="cpu", use_ema=True)
    check_arguments(args, parent_cfg)
    cfg = child_config(parent_cfg, args)
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    run_dir, last_dir = runs_dir / cfg.run_name, runs_dir / fallback_name(cfg.run_name)
    if list(run_dir.glob("*.ckpt")) and not args.force:
        raise SystemExit(f"{run_dir} already holds checkpoints; pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    #⚠️ a forced rerun must not average a previous run's later epochs in
    for stale in run_dir.glob("epoch*.ckpt"):
        stale.unlink()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    model, table = build_child(cfg, parent, args.mode)
    del parent
    what = f"head on {cfg.second_embeddings}" if args.mode == "head" else ("SAM anneal" if cfg.sam_rho > 0 else "anneal")
    print(f"[anneal] {args.parent} (epoch {info['epoch']}) -> {cfg.run_name}: {what}, lr {cfg.lr:g} ({cfg.schedule}, "
          f"no warmup), {cfg.max_epochs} epochs, {100 * (cfg.fragment_p + cfg.isolated_p + cfg.run_p):.0f}% holdout-shaped "
          f"items, BatchNorm frozen, precision {cfg.precision}")
    parameter_names = {name for name, _ in model.named_parameters()}
    epochs = train(cfg, model, table, run_dir, args.max_steps)

    #the trained module is on disk now; free its memory before three scorings
    del model, table
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not epochs:
        raise SystemExit(f"No epoch checkpoint in {run_dir}: the fit saved nothing.")

    #the average of the last epochs is the member; the last epoch alone, a second one
    chosen = epochs[-args.average_last:]
    if len(chosen) < args.average_last:
        print(f"[anneal] WARNING: only {len(chosen)} epoch checkpoint(s) to average, not {args.average_last}")
    averaged = average_checkpoints(chosen, parameter_names)
    torch.save(averaged, run_dir / "best.ckpt")
    last = torch.load(epochs[-1], map_location="cpu", weights_only=False)
    #its own run name, so its holdout CSV and dumps never overwrite the average's
    last["hyper_parameters"]["run_name"] = last_dir.name
    last_dir.mkdir(parents=True, exist_ok=True)
    torch.save(last, last_dir / "best.ckpt")
    print(f"[anneal] best.ckpt = epochs {averaged['averaged_epochs']} averaged; {last_dir.name}/best.ckpt = epoch "
          f"{int(last['epoch'])} alone")

    #every report on the same protocols, the staged cache, this machine
    device = "cuda" if cfg.accelerator != "cpu" and torch.cuda.is_available() else "cpu"
    settings = dict(cache_dir=cfg.cache_dir, batch_size=cfg.batch_size, num_workers=cfg.num_workers, device=device)
    parent_report = evaluate_checkpoint(args.parent, protocols=("fragment",), **settings)
    reports = {}
    for name, directory in (("averaged", run_dir), ("last", last_dir)):
        report = evaluate_checkpoint(directory / "best.ckpt", protocols=("reference", "fragment"),
                                     dump_dir=directory / "dumps", **settings)
        report["final_epoch_metrics"] = last_epoch_metrics(run_dir)
        (directory / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        reports[name] = report

    #the sanity verdict: within the margin of the parent on fragments, or not
    k = cfg.metric_k
    scores = {"parent": fragment_scores(parent_report, k)}
    scores.update({name: fragment_scores(report, k) for name, report in reports.items()})
    verdicts = {name: "keep" if scores[name]["mean"] >= scores["parent"]["mean"] - args.margin / 100 else "drop"
                for name in reports}
    record = {
        "mode": args.mode,
        "parent": args.parent,
        "parent_epoch": info["epoch"],
        "run_name": cfg.run_name,
        "fallback": last_dir.name,
        "settings": {
            "lr": cfg.lr, "schedule": cfg.schedule, "epochs": cfg.max_epochs, "fragment_p": cfg.fragment_p,
            "isolated_p": cfg.isolated_p, "run_p": cfg.run_p, "freeze_batchnorm": cfg.freeze_batchnorm,
            "freeze_backbone": cfg.freeze_backbone, "sam_rho": cfg.sam_rho, "precision": cfg.precision,
            "second_embeddings": cfg.second_embeddings, "average_last": args.average_last, "seed": cfg.seed,
            "max_steps": args.max_steps,
        },
        "averaged_epochs": averaged["averaged_epochs"],
        f"fragment_bacc{k}": scores,
        "margin_points": args.margin,
        "verdict": verdicts,
    }
    (run_dir / "anneal.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for name in ("parent", *reports):
        line = scores[name]
        print(f"[anneal] fragment BAcc@{k} {name:9s} validation {100 * line['validation']:.2f}  test "
              f"{100 * line['test']:.2f}  mean {100 * line['mean']:.2f}"
              + (f"  -> {verdicts[name]}" if name in verdicts else ""))
    print(f"[anneal] wrote {run_dir / 'anneal.json'}")


if __name__ == "__main__":
    main()
