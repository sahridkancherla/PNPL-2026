"""The competition metric.

BAcc@k -- macro-averaged Recall@k: the per-class rate at which the true word
appears in the model's top k, averaged over the classes present in the labels.

Chance is k/n_classes, so BAcc@10 over 50 words sits at ~0.20 and BAcc@1 at
~0.02. Comparing a top-1 score against the top-10 floor makes a model that is
learning look broken, so both are logged.

Sources disagree on which k the leaderboard uses: the tutorial notebook
implements BAcc@10 and calls it "the official-style metric", while
pnpl/competition/submission.py documents "balanced accuracy on argmax" (k=1).
pnpl ships no 2026 scorer to settle it, so `metric_k` is configurable and both
are reported. Confirm against the competition rules before trusting either.
"""

from __future__ import annotations

import torch


def balanced_topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, k: int, n_classes: int
) -> float:
    """Macro-averaged Recall@k over the classes appearing in `labels`.

    Mirrors the tutorial notebook's reference implementation.
    """
    k = min(k, logits.shape[1])
    topk = logits.topk(k, dim=1).indices
    hit = (topk == labels.unsqueeze(1)).any(dim=1).float()
    recalls = [hit[labels == c].mean() for c in range(n_classes) if (labels == c).any()]
    if not recalls:
        return 0.0
    return float(torch.stack(recalls).mean())
