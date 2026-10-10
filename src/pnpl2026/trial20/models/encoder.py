"""What trial 19 keeps of the per-word encoder: its attention pool.

The per-word `WordEncoder` of trials 10-18 is gone from this copy; the
continuous family (`continuous.py`) has its own stem. `AttentionPool` is
reused as the readout's pooling over the steps around an onset.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Collapse time by a learned weighting over the real samples.

    Single-head: one score per sample, softmaxed over the window.

    ATTRS:
        score (nn.Linear): D -> 1, no bias -- a constant cancels in the softmax
    """

    def __init__(self, n_features):
        """
        ARGS:
            n_features (int): D
        """
        super().__init__()
        self.score = nn.Linear(n_features, 1, bias=False)

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]
            valid (torch.Tensor or None): [batch, n_times] bool

        RETURNS:
            [batch, n_features]
        """
        scores = self.score(x)

        #-inf is zero weight after the softmax; every window has real samples
        if valid is not None:
            scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

        #dim=1 is TIME: the weights sum to one across the window
        weights = scores.softmax(dim=1)
        return (x * weights).sum(dim=1)
