"""The competition metric.

BAcc@k -- macro-averaged Recall@k: the per-class rate at which the true word
appears in the model's top k, averaged over the classes present in the labels.
BAcc@10 is the leaderboard metric; BAcc@1 is logged alongside it purely as a
tiebreaker and is not itself scored.

Chance is k/n_classes, so BAcc@10 over 50 words sits at ~0.20 and BAcc@1 at
~0.02 -- reading the BAcc@1 number against the 0.20 floor makes a model that
is learning look broken.
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
