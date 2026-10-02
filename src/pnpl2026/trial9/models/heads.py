"""Pooling over a word's frames, and the two per-word heads.

Variant D's heads read one vector per WORD -- the sentence transformer's
output, or the pooled window itself when there is no sentence transformer:

    `ClosedSetHead`   50-way logits, class-weighted cross-entropy
    `SemanticHead`    ONE linear map to the t5 embedding, as the design
                      specifies -- trials 5 and 6 used two layers

The session adversary is gone: trial 5's stayed at chance even with its
strength near zero, and the design drops it until a probe finds something.

⚠️ Channels last, `[batch, n_times, n_features]`.
"""

from __future__ import annotations

import torch.nn as nn


class AttentionPool(nn.Module):
    """Collapse time by a learned weighting, ignoring the padding.

    A word's evidence arrives at a particular latency after onset, so a flat
    mean over the window dilutes it. This scores every frame, softmaxes the
    scores over the window's REAL frames, and returns the weighted sum.

    ATTRS:
        score (nn.Linear): D -> 1, the per-frame relevance score
    """

    def __init__(self, n_features):
        """
        ARGS:
            n_features (int): D arriving from the encoder
        """
        super().__init__()

        #⚠️ bias=False: the scores go through a softmax over TIME, so the same
        #constant on every frame cancels exactly and could never learn
        self.score = nn.Linear(n_features, 1, bias=False)

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]
            valid (torch.Tensor or None): [batch, n_times] bool, False on
                padding frames, which get zero weight

        RETURNS:
            pooled : [batch, n_features] - one vector per window
        """
        scores = self.score(x)

        #-inf before the softmax is exactly zero weight after it. Every window
        #has at least 1 s of real frames, so no row is all -inf
        if valid is not None:
            scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

        #dim=1 is TIME: the weights must sum to one across the window
        weights = scores.softmax(dim=1)
        return (x * weights).sum(dim=1)


class MeanPool(nn.Module):
    """Collapse time by averaging the real frames. The cheapest baseline."""

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]
            valid (torch.Tensor or None): [batch, n_times] bool

        RETURNS:
            pooled : [batch, n_features]
        """
        if valid is None:
            return x.mean(dim=1)

        #dim=1 is time; divide by the real frame count, not the padded one
        weights = valid.unsqueeze(2).to(x.dtype)
        return (x * weights).sum(dim=1) / weights.sum(dim=1)


POOLING = {
    "mean": MeanPool,
    "attention": AttentionPool,
}


class SemanticHead(nn.Module):
    """One linear map from a word's vector to its t5 embedding.

    ATTRS:
        net (nn.Sequential): dropout then D -> embed_dim
    """

    def __init__(self, n_features, embed_dim, dropout=0.1):
        """
        ARGS:
            n_features (int): D, the word vector
            embed_dim (int): target embedding width, 1024 for t5-large
            dropout (float): on the word vector
        """
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, embed_dim))

    def forward(self, words):
        """
        ARGS:
            words (torch.Tensor): [n_words, n_features]

        RETURNS:
            embedding : [n_words, embed_dim] - NOT normalised; the loss does that
        """
        return self.net(words)


class ClosedSetHead(nn.Module):
    """Dropout, then one linear map to the 50 scored words.

    ATTRS:
        net (nn.Sequential): dropout then D -> n_classes
    """

    def __init__(self, n_features, n_classes, dropout=0.1):
        """
        ARGS:
            n_features (int): D, the word vector
            n_classes (int): 50
            dropout (float): on the word vector
        """
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_features, n_classes))

    def forward(self, words):
        """
        ARGS:
            words (torch.Tensor): [n_words, n_features]

        RETURNS:
            logits : [n_words, n_classes] - raw scores, NOT softmaxed
        """
        return self.net(words)
