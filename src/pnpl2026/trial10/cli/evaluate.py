"""Score a trial-10 checkpoint on pnpl's validation and test partitions.

    PYTHONPATH=src python -m pnpl2026.trial10.cli.evaluate \\
        --ckpt runs/trial10_f0_s0/best.ckpt

Two protocols on each partition, from the same windows of the same recordings:

    reference   every word a token, full 3 s windows. On the TEST partition
                this is the competition paper's setting: its d'Ascoli
                reference scored 73.23% BAcc@10, 25.60% BAcc@1, and 83.60%
                Moses BAcc@10
    holdout     what the competition holdout serves: scored words only, runs
                of at most 9, MEG stopping 1.1-1.7 s after the last word, 9.6%
                of words alone with 1 s. The gap between the two is how much of
                the reference survives the holdout's conditions

Writes `evaluation.json` beside the checkpoint. `cli.train` calls this on
best.ckpt when a run ends; run it by hand to rescore, or to score last.ckpt.

Needs only the reference cache and the checkpoint -- no t5 table: scoring is
retrieval among the 50 candidates stored in the checkpoint.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...metrics import balanced_topk_accuracy
from ..config import Trial10Config
from ..data import build_partition, collate_tokens
from ..lit import VOCABULARY_SIZE, load_trained

#the competition paper's Table 1, d'Ascoli reference, standard test split
PAPER_REFERENCE = {"bacc10": 0.7323, "bacc1": 0.2560, "moses_bacc10": 0.8360}

PARTITIONS = ("validation", "test")
PROTOCOLS = ("reference", "holdout")


class NoTargets:
    """Stands in for the t5 table when only the scored columns matter."""

    def rows(self, words):
        """No word has a training target here.

        ARGS:
            words (np.ndarray): [n] words

        RETURNS:
            [n] int64 array of -1
        """
        return np.full(len(words), -1, dtype=np.int64)


def parse_args():
    """Command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="best.ckpt or last.ckpt of a trial-10 run")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="the reference cache; default the run's, or data/trial10_reference if that is gone",
    )
    parser.add_argument("--partitions", nargs="+", default=list(PARTITIONS), choices=PARTITIONS)
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS), choices=PROTOCOLS)
    parser.add_argument("--batch-size", type=int, default=None, help="sequences per batch; default the run's")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="default cuda when available")
    parser.add_argument("--out", default=None, help="default evaluation.json beside the checkpoint")
    parser.add_argument("--no-ema", action="store_true", help="score the live weights, not the average")
    return parser.parse_args()


def topk(logits, targets, k):
    """BAcc@k over the 50 columns, or None when nothing is scored.

    ARGS:
        logits (torch.Tensor): [n, 50]
        targets (torch.Tensor): [n]
        k (int)

    RETURNS:
        float or None
    """
    if len(targets) == 0:
        return None
    return balanced_topk_accuracy(logits, targets, k, VOCABULARY_SIZE)


@torch.no_grad()
def score_dataset(module, dataset, batch_size, num_workers, device, k=10):
    """Run one partition under one protocol and score it.

    ARGS:
        module (LitTrialTen): in eval mode, on `device`
        dataset (SequenceWindows)
        batch_size (int): sequences per batch
        num_workers (int)
        device (torch.device)
        k (int): the scored k

    RETURNS:
        dict of counts and metrics, JSON-ready
    """
    #⚠️ never shuffled: per-token tasks below are rebuilt from this order
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_tokens
    )
    collected = {"primary": [], "primary_targets": [], "isolated": [], "moses": [], "moses_targets": []}
    scored_masks = []
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        primary, moses = module.candidate_logits(module(batch)["embedding"])

        scored = batch["primary"] >= 0
        collected["primary"].append(primary[scored].cpu())
        collected["primary_targets"].append(batch["primary"][scored].cpu())
        collected["isolated"].append(batch["isolated"][scored].cpu())
        in_moses = batch["moses"] >= 0
        collected["moses"].append(moses[in_moses].cpu())
        collected["moses_targets"].append(batch["moses"][in_moses].cpu())
        scored_masks.append(scored.cpu())

    stacked = {name: torch.cat(values) for name, values in collected.items()}
    primary, targets, isolated = stacked["primary"], stacked["primary_targets"], stacked["isolated"]

    metrics = {
        "n_sequences": len(dataset),
        "n_tokens": int(sum(len(sequence.onsets) for sequence in dataset.sequences)),
        "n_scored": int(len(targets)),
        "n_classes_present": int(len(torch.unique(targets))),
        f"bacc{k}": topk(primary, targets, k),
        "bacc1": topk(primary, targets, 1),
        f"moses_bacc{k}": topk(stacked["moses"], stacked["moses_targets"], k),
        "moses_bacc1": topk(stacked["moses"], stacked["moses_targets"], 1),
        "n_moses_scored": int(len(stacked["moses_targets"])),
    }

    #the holdout's two kinds of word, scored apart; the reference has no isolated
    if isolated.any():
        metrics[f"bacc{k}_sentence"] = topk(primary[~isolated], targets[~isolated], k)
        metrics[f"bacc{k}_isolated"] = topk(primary[isolated], targets[isolated], k)

    #per task: tokens come out in sequence order, so each one's recording is known
    recording_of_token = np.concatenate(
        [np.full(len(sequence.onsets), sequence.recording) for sequence in dataset.sequences]
    )
    recording_of_scored = recording_of_token[torch.cat(scored_masks).numpy()]
    tasks = np.array([dataset.recordings.records[index]["task"] for index in recording_of_scored])
    metrics["by_task"] = {}
    for task in sorted(set(tasks)):
        chosen = torch.from_numpy(tasks == task)
        metrics["by_task"][task] = {
            "n_scored": int(chosen.sum()),
            f"bacc{k}": topk(primary[chosen], targets[chosen], k),
        }
    return metrics


def evaluate_checkpoint(
    checkpoint_path,
    cache_dir=None,
    partitions=PARTITIONS,
    protocols=PROTOCOLS,
    batch_size=None,
    num_workers=0,
    device=None,
    use_ema=True,
):
    """Score one checkpoint on every partition under every protocol.

    ARGS:
        checkpoint_path (Path or str)
        cache_dir (Path, str or None): override the run's cache location
        partitions (tuple): of "validation", "test"
        protocols (tuple): of "reference", "holdout"
        batch_size (int or None): default the run's
        num_workers (int)
        device (str or None): default cuda when available
        use_ema (bool): averaged weights when the checkpoint has them

    RETURNS:
        dict, JSON-ready
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module, cfg, info = load_trained(checkpoint_path, device, use_ema)

    #a staged cache from a finished job is gone; fall back to the repo's
    if cache_dir is not None:
        cfg = dataclasses.replace(cfg, cache_dir=str(cache_dir))
    elif not (cfg.resolved_cache_dir() / "manifest.json").exists():
        print(f"[evaluate] {cfg.resolved_cache_dir()} is gone; using {Trial10Config.cache_dir}")
        cfg = dataclasses.replace(cfg, cache_dir=Trial10Config.cache_dir)

    results = {}
    for protocol in protocols:
        results[protocol] = {}
        for partition in partitions:
            dataset = build_partition(cfg, NoTargets(), partition, protocol)
            metrics = score_dataset(module, dataset, batch_size or cfg.batch_size, num_workers, device, cfg.metric_k)
            results[protocol][partition] = metrics
            print(
                f"[evaluate] {protocol:9s} {partition:10s} BAcc@{cfg.metric_k} "
                f"{format_score(metrics[f'bacc{cfg.metric_k}'])}  BAcc@1 {format_score(metrics['bacc1'])}  "
                f"Moses@{cfg.metric_k} {format_score(metrics[f'moses_bacc{cfg.metric_k}'])}  "
                f"({metrics['n_scored']:,} scored of {metrics['n_tokens']:,} tokens)"
            )

    return {
        "checkpoint": str(checkpoint_path),
        "run_name": cfg.run_name,
        "seed": cfg.seed,
        "epoch": info["epoch"],
        "global_step": info["global_step"],
        "paper_reference_test": PAPER_REFERENCE,
        "results": results,
    }


def format_score(value):
    """A metric as a percentage, or a dash when it was not computed.

    ARGS:
        value (float or None)

    RETURNS:
        str
    """
    return "   -  " if value is None else f"{100 * value:5.2f}%"


def main():
    """Score a checkpoint and write evaluation.json."""
    args = parse_args()
    report = evaluate_checkpoint(
        args.ckpt,
        cache_dir=args.cache_dir,
        partitions=tuple(args.partitions),
        protocols=tuple(args.protocols),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_ema=not args.no_ema,
    )
    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "evaluation.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[evaluate] wrote {out_path}")


if __name__ == "__main__":
    main()
