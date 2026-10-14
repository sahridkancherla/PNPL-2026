"""Score a per-word checkpoint on pnpl's validation and test partitions.

Trial 18's `cli/evaluate.py`, as functions, on this trial's per-word code:
trial 23's own `cli/evaluate.py` scores the continuous family, and hands a
per-word checkpoint here (`families.family_of` decides). Used by
`cli/anneal.py` to score what it trains and its parent.

    reference   every word a token, full 3 s windows
    fragment    the holdout's shape: runs of 4-8 consecutive vocabulary words,
                the MEG stopping 1.13-1.68 s after the last, 9.6% of scored
                words alone with 1 s
    sparse, isolated, chains, chains38   as `data/sequences.py` defines them

⚠️ Every checkpoint is scored with THIS trial's protocol settings
(`config.PROTOCOL_FIELDS`), whatever its own config saved, so all runs are
judged on identical sequences.

The dumps hold each token's 50 + 50 scores, targets, isolated flag,
sequence and task, in loader order: what `local_checks.read_dump` and the
transduction's `member_setup` read.
"""

from __future__ import annotations

import copy
import dataclasses
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...metrics import balanced_topk_accuracy
from .config import PROTOCOL_FIELDS, PROTOCOLS, Trial18Config, with_protocol_defaults
from .data import build_partition, collate_tokens
from .lit import VOCABULARY_SIZE, adapt_batch_norm, load_trained

#the competition paper's Table 1, d'Ascoli reference, standard test split
PAPER_REFERENCE = {"bacc10": 0.7323, "bacc1": 0.2560, "moses_bacc10": 0.8360}

PARTITIONS = ("validation", "test")

#the suffix a BatchNorm-adapted score is filed under
ADAPTED_SUFFIX = "+bn"


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


def make_loader(dataset, batch_size, num_workers):
    """The fixed-order loader every pass over a dataset uses.

    ARGS:
        dataset (SequenceWindows)
        batch_size (int): sequences per batch
        num_workers (int)

    RETURNS:
        DataLoader
    """
    #⚠️ never shuffled: per-token tasks and dumps are rebuilt from this order
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_tokens)


@torch.no_grad()
def score_dataset(module, dataset, batch_size, num_workers, device, k=10):
    """Run one partition under one protocol and score it.

    ARGS:
        module (LitTrialEighteen): in eval mode, on `device`
        dataset (SequenceWindows)
        batch_size (int): sequences per batch
        num_workers (int)
        device (torch.device)
        k (int): the scored k

    RETURNS:
        metrics : dict, JSON-ready
        dump : dict of numpy arrays, one row per token in loader order
    """
    names = ("primary_logits", "primary_logits_first", "moses_logits", "primary", "moses", "isolated")
    collected = {name: [] for name in names}
    for batch in make_loader(dataset, batch_size, num_workers):
        batch = {name: value.to(device) for name, value in batch.items()}
        out = module(batch)

        #what is ranked: one head's scores, or Variant K's two heads combined
        primary_logits, moses_logits = module.candidate_scores(out)

        #the first head alone as well, so K's second head can be judged
        collected["primary_logits_first"].append(module.candidate_logits(out["embedding"])[0].cpu())

        #every token is kept, scored or not, so positions line up with sequences
        collected["primary_logits"].append(primary_logits.cpu())
        collected["moses_logits"].append(moses_logits.cpu())
        for name in ("primary", "moses", "isolated"):
            collected[name].append(batch[name].cpu())
    stacked = {name: torch.cat(values) for name, values in collected.items()}

    #each token's sequence and recording, from the dataset's own order
    sequence_of_token = np.concatenate(
        [np.full(len(sequence.onsets), index) for index, sequence in enumerate(dataset.sequences)]
    )
    recording_of_token = np.concatenate(
        [np.full(len(sequence.onsets), sequence.recording) for sequence in dataset.sequences]
    )
    tasks = np.array([record["task"] for record in dataset.recordings.records])[recording_of_token]

    scored = stacked["primary"] >= 0
    primary, targets, isolated = stacked["primary_logits"][scored], stacked["primary"][scored], stacked["isolated"][scored]
    in_moses = stacked["moses"] >= 0
    metrics = {
        "n_sequences": len(dataset.sequences),
        "n_tokens": int(len(scored)),
        "n_scored": int(scored.sum()),
        "n_classes_present": int(len(torch.unique(targets))),
        f"bacc{k}": topk(primary, targets, k),
        "bacc1": topk(primary, targets, 1),
        f"moses_bacc{k}": topk(stacked["moses_logits"][in_moses], stacked["moses"][in_moses], k),
        "moses_bacc1": topk(stacked["moses_logits"][in_moses], stacked["moses"][in_moses], 1),
        "n_moses_scored": int(in_moses.sum()),
    }

    #Variant K: the first head alone; equal to bacc above with one head
    if module.cfg.has_second_head:
        metrics[f"bacc{k}_first_head"] = topk(stacked["primary_logits_first"][scored], targets, k)

    #the holdout's two kinds of word, scored apart; the reference has no isolated
    if isolated.any():
        metrics[f"bacc{k}_sentence"] = topk(primary[~isolated], targets[~isolated], k)
        metrics[f"bacc{k}_isolated"] = topk(primary[isolated], targets[isolated], k)
        metrics["n_isolated"] = int(isolated.sum())

    #per task, on scored tokens
    scored_tasks = tasks[scored.numpy()]
    metrics["by_task"] = {}
    for task in sorted(set(scored_tasks)):
        chosen = torch.from_numpy(scored_tasks == task)
        metrics["by_task"][task] = {"n_scored": int(chosen.sum()), f"bacc{k}": topk(primary[chosen], targets[chosen], k)}

    dump = {name: value.numpy() for name, value in stacked.items()}
    dump.update({"sequence": sequence_of_token, "task": tasks})
    return metrics, dump


def evaluate_checkpoint(
    checkpoint_path,
    cache_dir=None,
    partitions=PARTITIONS,
    protocols=PROTOCOLS,
    batch_size=None,
    num_workers=0,
    device=None,
    use_ema=True,
    dump_dir=None,
    bn_adapt=False,
):
    """Score one per-word checkpoint on every partition under every protocol.

    ARGS:
        checkpoint_path (Path or str)
        cache_dir (Path, str or None): override the run's cache location
        partitions (tuple): of "validation", "test"
        protocols (tuple): of PROTOCOLS
        batch_size (int or None): default the run's
        num_workers (int)
        device (str or None): default cuda when available
        use_ema (bool): averaged weights when the checkpoint has them
        dump_dir (Path or None): write per-token dumps here
        bn_adapt (bool): also score after AdaBN, per protocol and partition

    RETURNS:
        dict, JSON-ready
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module, cfg, info = load_trained(checkpoint_path, device, use_ema)

    #⚠️ every run judged on this trial's protocol settings, not its own
    cfg = with_protocol_defaults(cfg)

    #a staged cache from a finished job is gone; fall back to the repo's
    if cache_dir is not None:
        cfg = dataclasses.replace(cfg, cache_dir=str(cache_dir))
    elif not (cfg.resolved_cache_dir() / "manifest.json").exists():
        print(f"[evaluate] {cfg.resolved_cache_dir()} is gone; using {Trial18Config.cache_dir}")
        cfg = dataclasses.replace(cfg, cache_dir=Trial18Config.cache_dir)

    #the trained weights, to restore after every BatchNorm re-estimation
    trained_state = copy.deepcopy(module.state_dict()) if bn_adapt else None
    batch_size = batch_size or cfg.batch_size
    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    for protocol in protocols:
        for partition in partitions:
            dataset = build_partition(cfg, NoTargets(), partition, protocol)
            passes = [(protocol, False)] + ([(protocol + ADAPTED_SUFFIX, True)] if bn_adapt else [])
            for label, adapted in passes:
                #AdaBN on this dataset's own inputs, then score; undone after
                if adapted:
                    adapt_batch_norm(module, make_loader(dataset, batch_size, num_workers), device)
                metrics, dump = score_dataset(module, dataset, batch_size, num_workers, device, cfg.metric_k)
                if adapted:
                    module.load_state_dict(trained_state)

                results.setdefault(label, {})[partition] = metrics
                if dump_dir is not None:
                    np.savez_compressed(Path(dump_dir) / f"{label}_{partition}.npz", **dump)
                print(
                    f"[evaluate] {label:12s} {partition:10s} BAcc@{cfg.metric_k} "
                    f"{format_score(metrics[f'bacc{cfg.metric_k}'])}  BAcc@1 {format_score(metrics['bacc1'])}  "
                    f"sentence {format_score(metrics.get(f'bacc{cfg.metric_k}_sentence'))}  "
                    f"isolated {format_score(metrics.get(f'bacc{cfg.metric_k}_isolated'))}  "
                    f"({metrics['n_scored']:,} scored of {metrics['n_tokens']:,} tokens)"
                )

    return {
        "checkpoint": str(checkpoint_path),
        "run_name": cfg.run_name,
        "model": cfg.model,
        "seed": cfg.seed,
        "epoch": info["epoch"],
        "global_step": info["global_step"],
        "protocol_settings": {name: getattr(cfg, name) for name in PROTOCOL_FIELDS},
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
