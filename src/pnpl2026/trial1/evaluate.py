"""Score a trained checkpoint and report where it succeeds and where it fails.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.

The headline BAcc@10 is one number averaged over 50 classes of wildly
different size, so on its own it cannot distinguish a model that is mediocre
everywhere from one that is strong on frequent words and at chance on rare
ones. Those two want opposite fixes, so this reports the per-class breakdown
alongside the scalar.

⚠️ Every per-class recall here carries a standard error set by that class's
count, and the rare classes are small. `people` at n=34 has SE ~8.6 points.
The report prints the count beside every recall for exactly this reason:
a class-level difference smaller than its own error bar is not a finding.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..data.vocab import PRIMARY
from ..metrics import balanced_topk_accuracy


@torch.no_grad()
def collect_predictions(model, loader, device="cpu"):
    """Run the model over a loader and return logits and targets.

    ⚠️ The loader must not be shuffled. Nothing downstream checks the order,
    and a shuffled loader still produces a plausible-looking score.

    ARGS:
        model (nn.Module): the network, moved to `device` by the caller
        loader (DataLoader): un-shuffled, over the split being scored
        device (str): where to run the forward passes

    RETURNS:
        logits : [n_windows, n_classes] float32 on CPU
        targets : [n_windows] int64 on CPU
    """
    model.eval()
    logits, targets = [], []

    for windows, labels in loader:
        batch_logits = model(windows.to(device))

        #back to CPU immediately; the whole split's logits will not fit on a GPU
        logits.append(batch_logits.float().cpu())
        targets.append(labels.cpu())

    return torch.cat(logits), torch.cat(targets)


def per_class_recall(logits, targets, k, n_classes):
    """Recall@k for every class, with the count it was measured on.

    ARGS:
        logits (torch.Tensor): [n_windows, n_classes]
        targets (torch.Tensor): [n_windows] true class ids
        k (int): top-k the true class must appear in
        n_classes (int): 50

    RETURNS:
        rows (list): one dict per class present, each with word, class_id,
            count, recall and standard_error, sorted worst recall first
    """
    #k cannot exceed the number of columns, or topk raises
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    hit = (topk == targets.unsqueeze(1)).any(dim=1).float()

    rows = []
    for class_id in range(n_classes):
        mask = targets == class_id

        #a class absent from this split has no recall; the metric skips it too
        count = int(mask.sum())
        if count == 0:
            continue

        recall = float(hit[mask].mean())

        #binomial standard error, the scale below which a difference is noise
        standard_error = float(np.sqrt(recall * (1 - recall) / count))
        rows.append(
            {
                "word": PRIMARY[class_id],
                "class_id": class_id,
                "count": count,
                "recall": recall,
                "standard_error": standard_error,
            }
        )

    return sorted(rows, key=lambda row: row["recall"])


def build_report(logits, targets, cfg):
    """Assemble the full evaluation report.

    ARGS:
        logits (torch.Tensor): [n_windows, n_classes]
        targets (torch.Tensor): [n_windows]
        cfg (Trial1Config): the run config, for n_classes and metric_k

    RETURNS:
        report (dict): headline metrics, chance floors, and per-class rows
    """
    n_classes = cfg.n_classes
    k = cfg.metric_k

    #the scored metric, from the repo's one implementation
    bacc_k = balanced_topk_accuracy(logits, targets, k, n_classes)
    bacc_1 = balanced_topk_accuracy(logits, targets, 1, n_classes)
    rows = per_class_recall(logits, targets, k, n_classes)

    #chance is k/n_classes, so 0.20 at k=10 -- reading BAcc@1 against 0.20
    #makes a model that is learning look broken
    return {
        "n_windows": int(len(targets)),
        "n_classes_present": len(rows),
        f"bacc{k}": bacc_k,
        "bacc1": bacc_1,
        f"chance_bacc{k}": k / n_classes,
        "chance_bacc1": 1 / n_classes,
        f"bacc{k}_above_chance": bacc_k - k / n_classes,
        "per_class": rows,
    }


def print_report(report, cfg, n_extremes=8):
    """Print the report in a form that can be read off a job log.

    ARGS:
        report (dict): from `build_report`
        cfg (Trial1Config): for metric_k
        n_extremes (int): how many worst and best classes to show
    """
    k = cfg.metric_k
    print(f"\n[eval] {report['n_windows']} windows, "
          f"{report['n_classes_present']}/{cfg.n_classes} classes present")

    #every score is printed next to its floor, never alone
    print(f"[eval] BAcc@{k} = {report[f'bacc{k}']:.4f}   "
          f"(chance {report[f'chance_bacc{k}']:.2f}, "
          f"{report[f'bacc{k}_above_chance']:+.4f} above)")
    print(f"[eval] BAcc@1  = {report['bacc1']:.4f}   "
          f"(chance {report['chance_bacc1']:.2f}, tiebreaker only)")

    rows = report["per_class"]
    if not rows:
        return

    def show(title, subset):
        print(f"\n[eval] {title}")
        print(f"       {'word':<12} {'n':>6} {'recall':>8} {'+/- SE':>8}")
        for row in subset:
            print(
                f"       {row['word']:<12} {row['count']:>6} "
                f"{row['recall']:>8.3f} {row['standard_error']:>8.3f}"
            )

    #sorted worst-first, so the head and tail are the two ends of the range
    show(f"worst {n_extremes} classes at k={k}", rows[:n_extremes])
    show(f"best {n_extremes} classes at k={k}", rows[-n_extremes:][::-1])

    #a model scoring well only on common words is a different problem from one
    #that is mediocre everywhere, and the fixes point in opposite directions
    counts = np.array([row["count"] for row in rows], dtype=float)
    recalls = np.array([row["recall"] for row in rows], dtype=float)
    if len(rows) > 2 and counts.std() > 0 and recalls.std() > 0:
        correlation = float(np.corrcoef(np.log(counts), recalls)[0, 1])
        print(
            f"\n[eval] corr(log class count, recall) = {correlation:+.3f} -- "
            "strongly positive means the model is riding the class prior, "
            "which a balanced metric does not reward."
        )


def write_report(report, path):
    """Write the report to JSON beside the checkpoint.

    ARGS:
        report (dict): from `build_report`
        path (Path): destination file
    """
    #the per-class rows are the part worth keeping; a scalar in a log scrolls away
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"\n[eval] report -> {path}")
