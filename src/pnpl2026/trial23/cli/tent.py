"""Test-time entropy minimisation (Tent) of a per-word member on the holdout's sentences.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.tent --ckpt runs/trial13_g2_s0/best.ckpt \\
        --holdout-dir data/holdout [--epochs 5 --lr 1e-3]

The final night's TTA member (arXiv 2509.24700's form): the member scores the
holdout's 156 sentences as `cli.predict` serves them (`cli.distil`'s
`PerWordHooks`), and only the BatchNorm layers' scales and shifts learn to
make those scores more confident -- the mean entropy of the 50-way softmax of
each row's z-scored scores, at the temperature that gives the member's own
mean top-1 probability 0.3 before adapting. Everything else is frozen and the
network stays in eval mode: BatchNorm keeps its running statistics (AdaBN,
which re-estimates them, cost the mix points), dropout is off. No label is
read; lone words are left out (their rows are I3's in every file).

Writes `runs/<run>_tent/best.ckpt` in the member's own format (run name
`<run>_tent`, so `cli.predict` writes `<run>_tent_in_context_ts.csv`) and
`tent.json`: the entropy before and after, and the top-10 overlap of the
adapted member's holdout ranking with its parent's -- a large move is the
warning A0 gave. `pbs/trial23_tent.pbs` then evaluates and predicts it.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ...config import RUNS_DIR
from ..families import load_any
from . import predict as continuous_predict
from .distil import PerWordHooks, fit_temperature, silence_logging, zscore_rows_torch

#the BatchNorm parameters that learn; every other tensor stays the member's
NORMS = torch.nn.modules.batchnorm._BatchNorm


def parse_args():
    """Command-line arguments for the adaptation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="a per-word member's best.ckpt")
    parser.add_argument("--run-name", default=None, help="default <the checkpoint's run directory>_tent")
    parser.add_argument("--epochs", type=int, default=5, help="passes over the holdout's sentences")
    parser.add_argument("--items", type=int, default=16, help="sentences per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--confidence", type=float, default=0.3, help="the softmax temperature's target mean top-1")
    parser.add_argument("--scaling", choices=("training", "holdout"), default="training")
    parser.add_argument("--scaler-file", default="data/trial23_training_scaler.npz")
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--track", default="deep")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def holdout_scores(module, hooks, units):
    """The member's primary scores on every unit's rows, in canonical row order.

    ARGS:
        module (LitTrialEighteen): in eval mode
        hooks (PerWordHooks)
        units (list): of (rows, members)

    RETURNS:
        [n_rows_total, 50] numpy, rows in the order of `units`
    """
    found = []
    for start in range(0, len(units), 32):
        chosen = units[start : start + 32]
        primary, _ = module.candidate_scores(module(hooks.make_batch(chosen)))
        found.append(primary.float().cpu().numpy())
    return np.concatenate(found, axis=0)


def top10_overlap(before, after):
    """The mean share of each row's top 10 kept after the adaptation.

    ARGS:
        before, after (np.ndarray): [n, 50]

    RETURNS:
        float in [0, 1]
    """
    first = np.argsort(-before, axis=1)[:, :10]
    second = np.argsort(-after, axis=1)[:, :10]
    return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(first, second)]))


def mean_entropy(scores, temperature):
    """The mean entropy of each row's softmax of z-scored scores.

    ARGS:
        scores (np.ndarray): [n, 50]
        temperature (float)

    RETURNS:
        float, nats
    """
    standard = zscore_rows_torch(torch.from_numpy(scores).double())
    log_probabilities = F.log_softmax(standard / temperature, dim=1)
    return float(-(log_probabilities.exp() * log_probabilities).sum(dim=1).mean())


def main():
    """Adapt the member's BatchNorm scales and shifts, then write its checkpoint and tent.json."""
    args = parse_args()
    from pnpl.competition import LibriBrainCompetitionHoldout

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    run_name = args.run_name or f"{Path(args.ckpt).parent.name}_tent"
    run_dir = RUNS_DIR / run_name
    if (run_dir / "best.ckpt").exists() and not args.force:
        raise SystemExit(f"{run_dir} already holds a checkpoint; pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)

    module, cfg, info, family = load_any(args.ckpt, device, use_ema=True)
    if family != "perword":
        raise SystemExit("Tent here adapts BatchNorm, which only the per-word family has.")
    hooks = PerWordHooks(module, cfg, device, Namespace(batch_size=None))

    #the holdout's sentences, as predict serves them
    holdout = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir)
    scaler_for = continuous_predict.scaler_factory(args.scaling, args.scaler_file)
    sources = np.array([meta["source"] for meta in holdout.metadata])
    units = [unit for unit in hooks.holdout_units(holdout, scaler_for) if sources[unit[0][0]] == "sentence"]

    #the temperature from the member's own scores, before anything moves
    module.eval()
    before = holdout_scores(module, hooks, units)
    #the fit reads per-row z-scored scores, the space the softmax is taken in
    temperature = float(fit_temperature(zscore_rows_torch(torch.from_numpy(before).double()).numpy(), args.confidence))
    entropy_before = mean_entropy(before, temperature)

    #only BatchNorm's scale and shift learn; the network stays in eval mode
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    trainable = []
    for layer in module.modules():
        if isinstance(layer, NORMS) and layer.affine:
            for parameter in (layer.weight, layer.bias):
                parameter.requires_grad_(True)
                trainable.append(parameter)
    if not trainable:
        raise SystemExit(f"{args.ckpt} has no BatchNorm to adapt.")
    silence_logging(module)
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"[tent] {args.ckpt} ({cfg.model}) on {device}: {len(units)} sentences, {sum(p.numel() for p in trainable):,} "
          f"BatchNorm parameters, temperature {temperature:.3f}, entropy {entropy_before:.4f} nats before")

    started, step = time.time(), 0
    for epoch in range(args.epochs):
        order = random.sample(units, len(units))
        for start in range(0, len(order), args.items):
            chosen = order[start : start + args.items]
            primary, _ = module.candidate_scores(module(hooks.make_batch(chosen)))
            log_probabilities = F.log_softmax(zscore_rows_torch(primary.float()) / temperature, dim=1)
            #the mean entropy of the batch's rows; dim=1 is the candidate axis
            entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=1).mean()
            optimiser.zero_grad(set_to_none=True)
            entropy.backward()
            optimiser.step()
            step += 1
        print(f"[tent] epoch {epoch + 1}: last batch entropy {float(entropy.detach()):.4f} nats, {step} steps")

    #the adapted member's ranking against its parent's, and its entropy
    module.eval()
    after = holdout_scores(module, hooks, units)
    record = {"parent": str(args.ckpt), "run_name": run_name, "epochs": args.epochs, "steps": step, "lr": args.lr,
              "temperature": temperature, "entropy_before": entropy_before, "entropy_after": mean_entropy(after, temperature),
              "top10_overlap_with_parent": top10_overlap(before, after), "sentences": len(units),
              "seconds": time.time() - started}

    #the member's own checkpoint format, under the new run name
    cfg_dict = dict(cfg.to_dict(), run_name=run_name)
    checkpoint = {"state_dict": {name: value.detach().cpu() for name, value in module.state_dict().items()},
                  "hyper_parameters": cfg_dict, "epoch": info["epoch"], "global_step": info["global_step"], "tent": record}
    torch.save(checkpoint, run_dir / "best.ckpt")
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding="utf-8")
    (run_dir / "tent.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[tent] entropy {record['entropy_before']:.4f} -> {record['entropy_after']:.4f} nats; the top 10 keeps "
          f"{100 * record['top10_overlap_with_parent']:.1f}% of the parent's; wrote {run_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
