"""Per-class recall, beyond the headline scalar.

BAcc@10 is one number averaged over 50 classes, and it hides the thing worth
knowing: whether the model has learned 50 words a bit or 8 words well. With a
121x class imbalance those two look identical from the outside.

⚠️ Read every per-class number against its own count. `people` has ~341 examples
in total, so at a 10% holdout its recall is estimated from ~34 trials and carries
a standard error of about 8.6 points. A class at 0.40 and a class at 0.55 are not
distinguishable at that sample size.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data import vocab


def per_class_recall(logits, labels, k, n_classes):
    """Recall@k for every class present in `labels`.

    ARGS:
        logits (torch.Tensor): [n_windows, n_classes]
        labels (torch.Tensor): [n_windows] int64
        k (int): how many top predictions count as a hit
        n_classes (int): 50

    RETURNS:
        recalls (np.ndarray): [n_classes] float, NaN for absent classes
        counts (np.ndarray): [n_classes] int, how many examples each had
    """
    #k cannot exceed the number of classes, or topk raises
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices

    #dim=1 is the k axis: "did the true label appear anywhere in the top k"
    hit = (topk == labels.unsqueeze(1)).any(dim=1).float().numpy()
    labels = labels.numpy()

    recalls = np.full(n_classes, np.nan)
    counts = np.zeros(n_classes, dtype=np.int64)
    for class_id in range(n_classes):
        #⚠️ masked by POSITION, not by index value: labels repeat, so an
        #index-based mask would pick the wrong rows
        present = labels == class_id
        counts[class_id] = int(present.sum())
        if counts[class_id]:
            recalls[class_id] = float(hit[present].mean())
    return recalls, counts


def report(logits, labels, k, n_classes, title="validation"):
    """Print a per-class recall table, worst class first.

    ARGS:
        logits (torch.Tensor): [n_windows, n_classes]
        labels (torch.Tensor): [n_windows] int64
        k (int): the metric's k
        n_classes (int): 50
        title (str): what to call this split in the heading

    RETURNS:
        recalls (np.ndarray): [n_classes], NaN where absent
    """
    recalls, counts = per_class_recall(logits, labels, k, n_classes)
    present = ~np.isnan(recalls)

    chance = k / n_classes
    print(f"\n=== {title}: per-class Recall@{k} (chance {chance:.2f}) ===")
    print(f"{'word':<12}{'n':>7}{'recall':>9}{'SE':>8}   vs chance")

    #worst first: the classes dragging a balanced metric down are the ones a
    #headline average is specifically designed to hide
    order = np.argsort(np.where(present, recalls, np.inf))
    for class_id in order:
        if not present[class_id]:
            continue
        count = counts[class_id]
        recall = recalls[class_id]

        #binomial standard error, the floor on reading any single row
        standard_error = float(np.sqrt(max(recall * (1 - recall), 1e-9) / count))
        delta = recall - chance
        flag = "" if abs(delta) > 2 * standard_error else "  (within noise)"
        print(
            f"{vocab.PRIMARY[class_id]:<12}{count:>7}{recall:>9.3f}"
            f"{standard_error:>8.3f}   {delta:+.3f}{flag}"
        )

    print(f"\nBAcc@{k} over {int(present.sum())} present classes: "
          f"{np.nanmean(recalls):.4f}")

    #the count of classes at or below chance says whether the model is broad or
    #is scoring by betting on the frequent words
    at_chance = int((recalls[present] <= chance).sum())
    print(f"classes at or below chance: {at_chance}/{int(present.sum())}")
    return recalls
