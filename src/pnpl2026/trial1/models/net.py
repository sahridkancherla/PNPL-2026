"""The trial-1 network: stem, a stack of EncoderBlocks, a pooled head.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED: shape-checked only, never trained.
Treat every design choice here as a first guess to be measured, not a decision.

The block preserves both width and length, so it can be neither the first nor
the last thing in the network. This module supplies the two ends:

    [batch, n_channels, n_times]   whitened components straight from the cache
      -> stem      1x1, mixes components into d_model
      -> blocks    N x EncoderBlock, shape preserved throughout
      -> head      collapse time, then one linear map to the 50 words
      -> [batch, n_classes]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .block import EncoderBlock

#width the stem opens the whitened components out to, and how many blocks read it
DEFAULT_MODEL_WIDTH = 192
DEFAULT_N_BLOCKS = 4

#dropout sits only on the pooled vector, where the classifier can overfit fastest
DEFAULT_HEAD_DROPOUT = 0.1


class AttentionPool(nn.Module):
    """Collapse time by a learned weighting rather than a flat average.

    A word's evidence arrives at a particular latency after onset, so a flat
    mean over the whole window dilutes it with the samples either side. This
    scores every timepoint, softmaxes the scores into weights summing to one,
    and returns the weighted sum.

    ⚠️ Strictly more expressive than mean pooling, and strictly easier to
    break: an untrained scorer is near-uniform, so early training looks like
    mean pooling and any advantage only appears later. Measure it against
    `MeanPool` rather than assuming it wins.

    ATTRS:
        score (nn.Conv1d): D -> 1, the per-timepoint relevance score
    """

    def __init__(self, n_channels):
        """
        ARGS:
            n_channels (int): D, the width arriving from the block stack
        """
        super().__init__()

        #one score per timepoint, from the whole feature vector at that point
        self.score = nn.Conv1d(n_channels, 1, kernel_size=1)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            pooled : [batch, n_channels] - one vector per window
        """
        #softmax over TIME, not over features: the weights must sum to one
        #across the window, which is the axis being collapsed
        weights = torch.softmax(self.score(x), dim=2)

        #weighted sum over time; weights broadcast across the feature axis
        return (x * weights).sum(dim=2)


class MeanPool(nn.Module):
    """Collapse time by averaging. The cheapest honest baseline.

    Discards all latency structure, which for word decoding is a real loss --
    but it adds no parameters and has no failure mode, so it is what the first
    end-to-end number should come from.
    """

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            pooled : [batch, n_channels] - one vector per window
        """
        #dim=2 is time; an implicit reduction here would flatten the batch too
        return x.mean(dim=2)


POOLING = {
    "mean": MeanPool,
    "attention": AttentionPool,
}


class TrialOneNet(nn.Module):
    """Stem, block stack, pooled classifier head.

    ATTRS:
        stem (nn.Sequential): 1x1 lifting n_channels -> d_model, then a norm
        blocks (nn.Sequential): n_blocks EncoderBlocks, shape preserved
        pool (nn.Module): collapses the time axis to one vector per window
        classifier (nn.Sequential): dropout then d_model -> n_classes
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        d_model=DEFAULT_MODEL_WIDTH,
        n_blocks=DEFAULT_N_BLOCKS,
        pooling="mean",
        head_dropout=DEFAULT_HEAD_DROPOUT,
        with_quadrature=True,
        ffn_expansion=2,
    ):
        """Assemble the network.

        ARGS:
            n_channels (int): input width -- 73 whitened components, not 306
            n_classes (int): 50, the scored vocabulary
            d_model (int): D the blocks work in. Past ~2x the input rank this
                buys little, since the signal is rank-limited by the whitener
            n_blocks (int): how many EncoderBlocks to stack
            pooling (str): "mean" or "attention", see POOLING
            head_dropout (float): dropout on the pooled vector only
            with_quadrature (bool): passed to every block, for ablation
            ffn_expansion (int): passed to every block
        """
        super().__init__()

        #kept so forward() can check what it is handed, per the shape-check
        self.n_channels = n_channels

        if pooling not in POOLING:
            raise KeyError(f"Unknown pooling {pooling!r}; known: {sorted(POOLING)}")

        #1x1 because the blocks own the time axis -- the stem only decides how
        #the 73 components are mixed into the width the blocks work in
        self.stem = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
        )

        #every block is identical and shape-preserving, so depth is one number
        self.blocks = nn.Sequential(*[
            EncoderBlock(
                n_channels=d_model,
                ffn_expansion=ffn_expansion,
                with_quadrature=with_quadrature,
            )
            for _ in range(n_blocks)
        ])

        self.pool = POOLING[pooling](d_model) if pooling == "attention" else POOLING[pooling]()

        #dropout before the only weights that see the pooled vector
        self.classifier = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times] whitened

        RETURNS:
            logits : [batch, n_classes] - raw scores, NOT softmaxed
        """
        #a window served with the wrong component count is the silent failure
        #this whole trial is exposed to: 306 raw sensors instead of 73 whitened
        #components still convolves happily if the stem was built for 306
        n_channels = windows.shape[1]
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}. "
                f"The model is fed whitened components, not raw sensors -- "
                f"check n_channels against the cache's n_components."
            )

        #the cache stores float16; the stem's weights are float32
        x = self.stem(windows.float())

        #shape in == shape out, every block, by construction
        x = self.blocks(x)

        #[batch, d_model, n_times] -> [batch, d_model]
        pooled = self.pool(x)

        #logits, because cross_entropy applies its own log-softmax
        return self.classifier(pooled)
