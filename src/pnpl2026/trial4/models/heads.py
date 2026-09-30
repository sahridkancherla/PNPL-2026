"""What sits on top of the encoder: pooling, and the three heads.

Three heads, because the scored metric and the thing worth learning are not the
same object:

    `SemanticHead`      predicts a language-model embedding of the word. This is
                        what the 73.23% reference model does, and it is the one
                        head that can share statistical strength between classes
    `ClosedSetHead`     50-way logits. Optimises the scored metric directly
    `SessionAdversary`  trained through a gradient reversal, so the encoder
                        learns to be session-INVARIANT

⚠️ On the adversary. The reference model has a subject-specific layer because
subjects are known at test time. Our holdout is a HELD-OUT RECORDING with no
session id, so a per-session parameter has nothing to load. The only usable
version of that idea here is the inverse: punish the encoder for carrying
session identity at all.

⚠️ Channels last, `[batch, n_times, n_features]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    """Collapse time by a learned weighting rather than a flat average.

    A word's evidence arrives at a particular latency after onset, so a flat
    mean over the whole window dilutes it with the samples either side. This
    scores every frame, softmaxes the scores into weights summing to one over
    time, and returns the weighted sum. It is the same single-head, unit-output
    pooling the reference model uses.

    ⚠️ Strictly more expressive than mean pooling and strictly easier to break:
    an untrained scorer is near-uniform, so early training looks like mean
    pooling and any advantage only shows up later.

    ATTRS:
        score (nn.Linear): D -> 1, the per-frame relevance score
    """

    def __init__(self, n_features):
        """
        ARGS:
            n_features (int): D arriving from the encoder
        """
        super().__init__()

        #⚠️ bias=False, and not for tidiness: the scores go through a softmax
        #over TIME, so adding the same constant to every frame cancels exactly.
        #A bias here is a parameter that can never receive gradient.
        self.score = nn.Linear(n_features, 1, bias=False)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]

        RETURNS:
            pooled : [batch, n_features] - one vector per window
        """
        #dim=1 is TIME: the weights must sum to one across the window, which is
        #the axis being collapsed. dim=-1 here would normalise over features
        weights = self.score(x).softmax(dim=1)

        #weighted sum over time; the weights broadcast across the feature axis
        return (x * weights).sum(dim=1)


class MeanPool(nn.Module):
    """Collapse time by averaging. The cheapest honest baseline.

    Discards all latency structure, which for word decoding is a real loss, but
    it adds no parameters and has no failure mode.
    """

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]

        RETURNS:
            pooled : [batch, n_features]
        """
        #dim=1 is time; an implicit reduction would flatten the batch too
        return x.mean(dim=1)


POOLING = {
    "mean": MeanPool,
    "attention": AttentionPool,
}


class SemanticHead(nn.Module):
    """Project the pooled vector into the target word-embedding space.

    Two layers rather than one: the pooled encoder vector and a t5 embedding are
    not the same kind of object, and a single linear map has to be both the
    translation and the nonlinearity.

    ATTRS:
        net (nn.Sequential): D -> hidden -> embed_dim
    """

    def __init__(self, n_features, embed_dim, dropout=0.1):
        """
        ARGS:
            n_features (int): D arriving from the pooling
            embed_dim (int): target embedding width, e.g. 1024 for t5-large
            dropout (float): before the projection
        """
        super().__init__()

        #a hidden width between the two ends, so neither is a bottleneck
        hidden = max(n_features, embed_dim // 2)
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, pooled):
        """
        ARGS:
            pooled (torch.Tensor): [batch, n_features]

        RETURNS:
            embedding : [batch, embed_dim] - NOT normalised; the loss does that
        """
        return self.net(pooled)


class ClosedSetHead(nn.Module):
    """Dropout, then one linear map to the 50 scored words.

    ATTRS:
        net (nn.Sequential): dropout then D -> n_classes
    """

    def __init__(self, n_features, n_classes, dropout=0.1):
        """
        ARGS:
            n_features (int): D arriving from the pooling
            n_classes (int): 50
            dropout (float): on the pooled vector, where overfitting is fastest
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, n_classes),
        )

    def forward(self, pooled):
        """
        ARGS:
            pooled (torch.Tensor): [batch, n_features]

        RETURNS:
            logits : [batch, n_classes] - raw scores, NOT softmaxed
        """
        return self.net(pooled)


class _ReverseGradient(torch.autograd.Function):
    """Identity forwards, sign-flipped and scaled backwards."""

    @staticmethod
    def forward(ctx, x, strength):
        """
        ARGS:
            ctx: autograd context
            x (torch.Tensor): any shape
            strength (float): how hard to push against the encoder

        RETURNS:
            `x` unchanged
        """
        ctx.strength = strength

        #view_as rather than returning x directly, so autograd records a node
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        """
        RETURNS:
            the negated, scaled gradient, and None for `strength`
        """
        #the negation is the whole mechanism: the adversary descends while the
        #encoder it reads from ascends, so features it can use get destroyed
        return -ctx.strength * grad, None


class SessionAdversary(nn.Module):
    """Predict which recording session a window came from, and be defeated.

    Trained through a gradient reversal, so minimising this head's loss teaches
    the encoder to DISCARD session identity. Sessions differ 30x in size and the
    head drifts between them; anything the encoder learns that is specific to a
    training session is worth nothing on a held-out recording.

    ⚠️ `strength` must be ramped from zero. An adversary at full strength from
    step one prevents the encoder from learning anything at all, and the run
    looks like a broken architecture rather than a scheduling mistake.

    ATTRS:
        net (nn.Sequential): D -> hidden -> n_sessions
    """

    def __init__(self, n_features, n_sessions, dropout=0.1):
        """
        ARGS:
            n_features (int): D arriving from the pooling
            n_sessions (int): how many TRAINING sessions there are
            dropout (float): before the hidden layer
        """
        super().__init__()

        #deliberately small: a strong adversary makes the minimax unstable, and
        #the point is to remove the easy session cues, not to win a game
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, n_features // 2),
            nn.GELU(),
            nn.Linear(n_features // 2, n_sessions),
        )

    def forward(self, pooled, strength):
        """
        ARGS:
            pooled (torch.Tensor): [batch, n_features]
            strength (float): gradient-reversal coefficient. 0 detaches the
                encoder entirely, so the head trains and changes nothing

        RETURNS:
            logits : [batch, n_sessions]
        """
        return self.net(_ReverseGradient.apply(pooled, strength))
