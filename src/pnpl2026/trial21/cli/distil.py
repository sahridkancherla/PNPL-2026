"""Transductive self-distillation: fine-tune a trained member on the holdout's own epochs.

    PYTHONPATH=src python -m pnpl2026.trial21.cli.distil \\
        --ckpt runs/trial13_g2_s0/best.ckpt \\
        --teacher submissions/ens_k18_spw_ts_hubh_lone.csv \\
        --run-name trial21_d_g2_s0 [--steps 600 --kl-weight 0.5 --holdout-items 16]

Part A of trial 21 (`info/holdout-adaptation.md`). Every whole-point gain on
the board closed a gap between training conditions and what the holdout
serves; the gaps still open are the holdout SESSION (a recording no weight
or BatchNorm statistic has seen) and its TOKEN FORMAT (list words only, with
invisible neighbours and unknown gaps). The 156 sentence epochs and 92 lone
words are the only data that have both. This fine-tunes a student on them,
label-free, with the best ensemble as the teacher.

Each optimiser step has two forward passes:

    training batch    the student's own loss on its own training items,
                      shaped exactly as it was trained (the recipe's
                      shaping, dedup, heads) -- the anchor that keeps the
                      encoder honest
    holdout batch     `--holdout-items` holdout items in the form `predict`
                      serves them (a sentence as one item, a lone word as a
                      1 s item), through the network in TRAINING mode, and
                      KL(teacher || student) over the 50 scored words

Both distributions are softmaxes of per-row z-scored scores at one shared
temperature, the ensemble script's own space, so the student's SigLIP scale
is not what the KL moves. The temperature is set so the teacher's mean top-1
probability is `--teacher-confidence` (about its likely top-1 accuracy),
unless `--teacher-temperature` fixes it.

The teacher is a function of MEG, not of text: a student cannot learn the
training stories' list-word order from it, the thing that sank L and the
word-pair fusions. BatchNorm runs in training mode throughout, so its
statistics drift towards the holdout's; `--final-bn holdout` then
re-estimates them on all holdout items (AdaBN), which is what the holdout
passes saw during the fine-tune. No selection: no local score can judge
this, and `--steps` is fixed in advance.

Works on either family (`families.py`): a per-word checkpoint (G2, K2,
K18...) or a continuous one (SPW, the hybrid). Writes
`runs/<run-name>/best.ckpt` in the family's own format, so `cli.predict`
reads it as any other checkpoint, plus `distil.json` with the settings and
the loss curve.

⚠️ This is the kind of holdout use the organisers called "not prohibited,
but against the spirit of the competition" (CLAUDE.md, 2026-10-08). It is
reproducible from the code and the teacher file alone, and the teacher is
itself a submitted file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from ...config import REPO_ROOT, RUNS_DIR
from ..families import lit_module_for, load_any
from . import predict as continuous_predict
from .ensemble import read_submission, standardise_rows

#how far the temperature search looks, in log space
TEMPERATURE_BOUNDS = (0.02, 20.0)

#the gradient clip, as a guard against a bad holdout batch early on
GRADIENT_CLIP = 1.0


def parse_args():
    """Command-line arguments for one distillation run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="the student's starting checkpoint, either family")
    parser.add_argument("--teacher", required=True, help="a submission CSV: the teacher's distributions")
    parser.add_argument("--run-name", required=True, help="runs/<run-name>/ receives best.ckpt and distil.json")
    parser.add_argument("--steps", type=int, default=600, help="optimiser steps, fixed in advance")
    parser.add_argument("--warmup-steps", type=int, default=30, help="linear warmup, then a constant rate")
    parser.add_argument("--lr", type=float, default=3.0e-5, help="the network's learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--kl-weight", type=float, default=0.5, help="weight of the holdout KL term")
    parser.add_argument("--holdout-items", type=int, default=16, help="holdout items per step")
    parser.add_argument("--batch-size", type=int, default=None, help="training items per step; default the run's own")
    parser.add_argument("--teacher-confidence", type=float, default=0.3, help="target mean top-1 probability of the teacher")
    parser.add_argument("--teacher-temperature", type=float, default=None, help="fix the temperature instead")
    parser.add_argument("--no-lone", action="store_true", help="sentence items only; leave the 92 lone words out")
    parser.add_argument(
        "--final-bn",
        choices=("holdout", "blend"),
        default="holdout",
        help="after the fine-tune: re-estimate BatchNorm on all holdout items, or keep the blended running statistics",
    )
    parser.add_argument("--scaling", choices=("training", "holdout"), default="training")
    parser.add_argument("--scaler-file", default="data/trial21_training_scaler.npz")
    parser.add_argument("--holdout-dir", default=None, help="a local folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--track", default="deep")
    parser.add_argument("--cache-dir", default=None, help="override the run's cache_dir (a staged copy)")
    parser.add_argument("--num-workers", type=int, default=None, help="default the run's own")
    parser.add_argument("--device", default=None, help="default cuda when available")
    parser.add_argument("--amp", action="store_true", help="fp16 autocast with a loss scaler, for an RTX6000")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    return parser.parse_args()


# -- the teacher ----------------------------------------------------------


def fit_temperature(scores, confidence):
    """The temperature at which softmax(scores / T) has a given mean top-1 probability.

    ARGS:
        scores (np.ndarray): [n_rows, 50] per-row z-scored scores
        confidence (float): target mean of each row's largest probability

    RETURNS:
        float: T, found by bisection in log space
    """
    #a larger T flattens every row, so the mean top probability falls with T
    def top_probability(temperature):
        shifted = scores / temperature
        shifted = shifted - shifted.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return float((weights.max(axis=1) / weights.sum(axis=1)).mean())

    low, high = (math.log(bound) for bound in TEMPERATURE_BOUNDS)
    if top_probability(math.exp(low)) < confidence or top_probability(math.exp(high)) > confidence:
        raise SystemExit(f"--teacher-confidence {confidence} is outside what temperatures {TEMPERATURE_BOUNDS} reach.")
    for _ in range(60):
        middle = 0.5 * (low + high)
        if top_probability(math.exp(middle)) > confidence:
            low = middle
        else:
            high = middle
    return math.exp(0.5 * (low + high))


def load_teacher(path, temperature, confidence):
    """The teacher's per-row distributions over the 50 scored words.

    ARGS:
        path (Path): a submission CSV
        temperature (float or None): fixed, or None to fit it
        confidence (float): the fit's target, mean top-1 probability

    RETURNS:
        probabilities : [n_rows, 50] float64 torch tensor
        temperature : float
        scores : [n_rows, 50] float64 numpy, the z-scored scores
    """
    from pnpl.competition import load_vocabulary

    index, primary, _ = read_submission(Path(path), load_vocabulary("primary"), load_vocabulary("moses"))
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{path}: the index column is not 0..{len(index) - 1} in order.")

    #the ensemble script's space: each row's log-probabilities z-scored
    scores = standardise_rows(primary)
    if temperature is None:
        temperature = fit_temperature(scores, confidence)
    shifted = scores / temperature
    shifted = shifted - shifted.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    probabilities = weights / weights.sum(axis=1, keepdims=True)
    return torch.from_numpy(probabilities), float(temperature), scores


def zscore_rows_torch(scores):
    """Each row z-scored across its candidates, differentiably, as `standardise_rows` does.

    ARGS:
        scores (torch.Tensor): [N, 50]

    RETURNS:
        [N, 50]
    """
    #dim=1 is the candidate axis; population spread, as numpy's std
    centred = scores - scores.mean(dim=1, keepdim=True)
    spread = centred.pow(2).mean(dim=1, keepdim=True).sqrt()
    return centred / spread.clamp_min(1e-6)


# -- the two families' hooks ---------------------------------------------


class PerWordHooks:
    """How a per-word student's training items and holdout items are served."""

    def __init__(self, module, cfg, device, args):
        """
        ARGS:
            module (LitTrialEighteen): the student
            cfg (Trial18Config): its config, cache_dir and workers already overridden
            device (torch.device)
            args (argparse.Namespace)
        """
        self.module, self.cfg, self.device, self.args = module, cfg, device, args
        self.window_samples = cfg.samples(cfg.window_seconds)
        self.predictor = continuous_predict.perword_holdout.Predictor(module, self.window_samples, device)

    def training_loader(self):
        """The student's own training set, shaped as it was trained; the tables attached.

        RETURNS:
            DataLoader
        """
        from ..perword.data.datamodule import build_partition, training_shaping
        from ..perword.data.dataset import collate_tokens
        from ..perword.embeddings import TargetTable

        cfg = self.cfg
        table = TargetTable(cfg.resolved_word_embeddings(), centre=cfg.centre_embeddings)

        #the all-word tables are not checkpointed; the loss needs them back
        self.module.target_table = table.table.to(self.device)
        if cfg.has_second_head:
            second = TargetTable(cfg.resolved_second_embeddings(), centre=cfg.centre_embeddings)
            self.module.target_table_second = second.table.to(self.device)
        dataset = build_partition(
            cfg, table, list(cfg.train_partitions), cfg.training_protocol,
            cfg.samples(cfg.jitter_seconds), training_shaping(cfg), cfg.resolved_others_cache_dir(),
        )
        return make_loader(dataset, self.args.batch_size or cfg.batch_size, cfg.num_workers, collate_tokens)

    def holdout_units(self, holdout, scaler_for):
        """Every holdout sequence on its own, with its rows.

        ARGS:
            holdout (LibriBrainCompetitionHoldout)
            scaler_for (callable)

        RETURNS:
            list of (rows, members)
        """
        helpers = continuous_predict.perword_holdout
        return helpers.units_of(helpers.gather_in_context(holdout, self.window_samples, scaler_for))

    def make_batch(self, units):
        """A batch of holdout sequences on the device.

        ARGS:
            units (list): of (rows, members)

        RETURNS:
            dict
        """
        return self.predictor.make_batch([members for _, members in units])


class ContinuousHooks:
    """How a continuous student's training items and holdout items are served."""

    def __init__(self, module, cfg, device, args):
        """
        ARGS:
            module (LitTrialTwentyOne): the student
            cfg (Trial21Config): its config, cache_dir and workers already overridden
            device (torch.device)
            args (argparse.Namespace)
        """
        self.module, self.cfg, self.device, self.args = module, cfg, device, args
        self.predictor = continuous_predict.Predictor(module, device)

    def training_loader(self):
        """The student's own training set, shaped as it was trained; the tables attached.

        RETURNS:
            DataLoader
        """
        from ..data.datamodule import build_partition, training_shaping
        from ..data.dataset import collate_items
        from ..embeddings import TargetTable
        from ..models.tower import load_descriptions

        cfg = self.cfg
        table = TargetTable(cfg.resolved_word_embeddings(), centre=cfg.centre_embeddings)

        #the all-word tables are not checkpointed; the losses need them back
        self.module.target_table = table.table.to(self.device)
        if cfg.has_second_head:
            second = TargetTable(cfg.resolved_second_embeddings(), centre=cfg.centre_embeddings)
            self.module.target_table_second = second.table.to(self.device)
        if cfg.has_tower:
            descriptions = load_descriptions(cfg.resolved_word_descriptions(), table.words)
            self.module.description_table = descriptions["table"].to(self.device)
        dataset = build_partition(
            cfg, table, list(cfg.train_partitions), cfg.training_protocol,
            cfg.samples(cfg.jitter_seconds), training_shaping(cfg), cfg.resolved_others_cache_dir(), training=True,
        )
        return make_loader(dataset, self.args.batch_size or cfg.batch_size, cfg.num_workers, collate_items)

    def holdout_units(self, holdout, scaler_for):
        """Every holdout item on its own, with its rows.

        ARGS:
            holdout (LibriBrainCompetitionHoldout)
            scaler_for (callable)

        RETURNS:
            list of (rows, (signal, onsets))
        """
        batches = continuous_predict.gather_in_context(holdout, scaler_for, float(self.cfg.sfreq))
        return continuous_predict.units_of(batches)

    def make_batch(self, units):
        """A batch of holdout items on the device.

        ARGS:
            units (list): of (rows, item)

        RETURNS:
            dict
        """
        return self.predictor.make_batch([item for _, item in units])


def make_loader(dataset, batch_size, num_workers, collate):
    """A training DataLoader with the trials' usual settings.

    ARGS:
        dataset (Dataset)
        batch_size (int): items per batch
        num_workers (int)
        collate (callable)

    RETURNS:
        DataLoader
    """
    extra = {"persistent_workers": True, "prefetch_factor": 4} if num_workers > 0 else {}
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        drop_last=True,
        **extra,
    )


def endless(loader):
    """Batches forever, the loader restarted (reshuffled, reshaped) when it runs out.

    ARGS:
        loader (DataLoader)

    YIELDS:
        batch dicts
    """
    while True:
        for batch in loader:
            yield batch


def silence_logging(module):
    """Make a LightningModule's `self.log` a no-op outside a Trainer, and clear its epoch stores.

    ARGS:
        module (LightningModule)
    """
    module.log = lambda *args, **kwargs: None


def clear_stores(module):
    """Drop the per-epoch metric buffers `_step` fills; nothing reads them here.

    ARGS:
        module (LightningModule)
    """
    for store in getattr(module, "_epoch_store", {}).values():
        for values in store.values():
            values.clear()


# -- the run --------------------------------------------------------------


def main():
    """Load the student, the teacher and the data; fine-tune; save a checkpoint `predict` can read."""
    args = parse_args()
    from pnpl.competition import LibriBrainCompetitionHoldout

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    L.seed_everything(args.seed, workers=True)
    random.seed(args.seed)

    run_dir = RUNS_DIR / args.run_name
    if (run_dir / "best.ckpt").exists() and not args.force:
        raise SystemExit(f"{run_dir} already holds a checkpoint; choose another --run-name or pass --force.")
    run_dir.mkdir(parents=True, exist_ok=True)

    #the student, with its averaged weights if it has any: the live network from here on
    module, cfg, info, family = load_any(args.ckpt, device, use_ema=True)
    overrides = {"run_name": args.run_name}
    if args.cache_dir:
        overrides["cache_dir"] = args.cache_dir
    if args.num_workers is not None:
        overrides["num_workers"] = args.num_workers
    cfg = replace(cfg, **overrides)
    module.cfg = cfg
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    print(f"[distil] student {args.ckpt} ({family} family, {cfg.model}, epoch {info['epoch']}) on {device}")

    hooks = (PerWordHooks if family == "perword" else ContinuousHooks)(module, cfg, device, args)
    loader = hooks.training_loader()
    print(f"[distil] training items: {len(loader.dataset):,} sequences, {loader.batch_size} per step")

    #the holdout, item by item, in the form predict serves it
    holdout = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir)
    scaler_for = continuous_predict.scaler_factory(args.scaling, args.scaler_file)
    units = hooks.holdout_units(holdout, scaler_for)
    sources = np.array([meta["source"] for meta in holdout.metadata])
    if args.no_lone:
        units = [unit for unit in units if sources[unit[0][0]] != "word"]
    n_rows = sum(len(rows) for rows, _ in units)
    n_lone = sum(1 for rows, _ in units if sources[rows[0]] == "word")
    print(f"[distil] holdout items: {len(units)} ({n_lone} lone words), {n_rows} rows; {args.holdout_items} per step")

    #the teacher's distributions, one per canonical row
    teacher, temperature, teacher_scores = load_teacher(args.teacher, args.teacher_temperature, args.teacher_confidence)
    if len(teacher) != len(holdout):
        raise SystemExit(f"The teacher has {len(teacher)} rows; the holdout {len(holdout)}.")
    teacher = teacher.float().to(device)
    top10 = np.argsort(-teacher_scores, axis=1)[:, :10]
    share = np.bincount(top10.ravel(), minlength=50) / len(teacher_scores)
    print(f"[distil] teacher {args.teacher}: temperature {temperature:.3f}, mean top-1 probability "
          f"{float(teacher.max(dim=1).values.mean()):.3f}, top-10 share {share.min():.2f}-{share.max():.2f}")

    #every parameter trains; a constant rate after a short warmup
    silence_logging(module)
    module.train()
    trainable = [parameter for parameter in module.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: min((step + 1) / max(args.warmup_steps, 1), 1.0)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    print(f"[distil] {sum(p.numel() for p in trainable):,} trainable parameters; lr {args.lr:g}, "
          f"{args.steps} steps, KL weight {args.kl_weight:g}" + (", fp16 autocast" if args.amp else ""))

    curve = []
    batches = endless(loader)
    started = time.time()
    for step in range(args.steps):
        batch = {name: value.to(device, non_blocking=True) for name, value in next(batches).items()}
        chosen = random.sample(units, min(args.holdout_items, len(units)))
        rows = torch.tensor([row for unit_rows, _ in chosen for row in unit_rows], device=device)
        holdout_batch = hooks.make_batch(chosen)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type == "cuda"):
            #the student's own loss on its own items, exactly as in training
            loss_train = module._step(batch, "train")
            clear_stores(module)

            #the holdout items: the student's 50-way distribution against the teacher's
            primary, _ = module.candidate_scores(module(holdout_batch))
        log_student = F.log_softmax(zscore_rows_torch(primary.float()) / temperature, dim=1)
        target = teacher[rows]
        loss_kl = (target * (torch.log(target.clamp_min(1e-12)) - log_student)).sum(dim=1).mean()
        loss = loss_train.float() + args.kl_weight * loss_kl

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(trainable, GRADIENT_CLIP)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            point = {"step": step, "loss_train": float(loss_train), "loss_kl": float(loss_kl), "lr": scheduler.get_last_lr()[0]}
            curve.append(point)
            print(f"[distil] step {step:5d}  train {point['loss_train']:.4f}  kl {point['loss_kl']:.4f}  "
                  f"lr {point['lr']:.2e}  {time.time() - started:6.0f}s")

    #BatchNorm statistics as the holdout passes saw them: the holdout's own
    module.eval()
    n_norms = 0
    if args.final_bn == "holdout":
        chunks = [units[start : start + 32] for start in range(0, len(units), 32)]
        n_norms = lit_module_for(family).adapt_batch_norm(module, (hooks.make_batch(chunk) for chunk in chunks), device)
        print(f"[distil] final BatchNorm: {n_norms} layers re-estimated on all {len(units)} holdout items")
    module.eval()

    #a checkpoint in the family's own format: predict rebuilds it from this alone
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in module.state_dict().items()},
        "hyper_parameters": cfg.to_dict(),
        "epoch": info["epoch"],
        "global_step": (info["global_step"] or 0) + args.steps,
        "distil": {
            "student": str(args.ckpt),
            "teacher": str(args.teacher),
            "family": family,
            "temperature": temperature,
            "steps": args.steps,
            "kl_weight": args.kl_weight,
            "lr": args.lr,
            "holdout_items": args.holdout_items,
            "final_bn": args.final_bn,
            "bn_layers_reestimated": n_norms,
            "no_lone": bool(args.no_lone),
        },
    }
    torch.save(checkpoint, run_dir / "best.ckpt")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")
    (run_dir / "distil.json").write_text(
        json.dumps({**checkpoint["distil"], "curve": curve, "seconds": time.time() - started}, indent=2), encoding="utf-8"
    )
    print(f"[distil] wrote {run_dir / 'best.ckpt'}; predict it with pnpl2026.trial21.cli.predict")


if __name__ == "__main__":
    main()
