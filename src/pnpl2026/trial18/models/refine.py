"""Variant M: HDND's refinement -- a word's own conv features nudge its prediction.

HDND (Li et al. 2026, arXiv 2609.24095) keeps d'Ascoli's encoder and
sentence transformer -- ours -- frozen once trained, and learns a small
correction on top of their prediction z0:

    pyramid   the gated outputs of conv stages 2 and 4 (ConvSequence layers
              3 and 7), each averaged in 1, 2 and 4 time bins
                                                  -> [n_depths, 7 * hidden]
    project   standardised by training statistics, one bias-free map per
              depth, tanh                         -> v_l [width]
    query     z0 normalised, standardised, a bias-free map, tanh
                                                  -> q [width]
    fuse      a_l = softmax over depths of q . v_l / sqrt(width)
              u = sum_l a_l v_l  (+ q with query_skip, the paper's eta)
    residual  r = W u, W zero at the start; B(r) = c r / sqrt(c^2 + |r|^2),
              so |B(r)| < c; centred by its mean over training batches
    output    z = z0 + strength * (B(r) - centre)

Two departures from the paper, both for this competition's holdout:

    the pyramid pools each window's FIRST SECOND, not all 3 s. Every item
    has its first second whole -- lone words are exactly 1 s, holdout runs
    end 1.13 s or more after their last onset -- while later samples are cut
    short on the holdout and whole in training. The first second also holds
    the word's own response: the per-window evidence the board has rewarded.

    no query skip by default. The residual then depends on z0 only through
    which depth it weights, so the correction comes from the word's own MEG.
    A correction learned from z0 alone resembles trial 14's closed-set head,
    which lost on the board.

The standardisation statistics are filled once from training batches before
the refiner trains (`set_statistics`), then frozen. `strength` is 1 while it
trains and is chosen on validation afterwards.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bin_weights(bins, n_samples):
    """Indicator rows for every time bin of every resolution.

    ARGS:
        bins (list): resolutions, e.g. [1, 2, 4]
        n_samples (int): samples pooled from each window's start

    RETURNS:
        [sum(bins), n_samples] float32 - row j is 1 on bin j's samples
    """
    rows = []
    for resolution in bins:
        #integer edges, so the bins of one resolution tile the span exactly
        edges = [(part * n_samples) // resolution for part in range(resolution + 1)]
        for start, stop in zip(edges[:-1], edges[1:]):
            row = torch.zeros(n_samples)
            row[start:stop] = 1.0
            rows.append(row)
    return torch.stack(rows, dim=0)


class PyramidRefiner(nn.Module):
    """A bounded correction to the contextual prediction, from pooled conv features.

    ATTRS:
        width (int): the shared projected width
        bound (float): c, the residual's norm limit
        query_skip (bool): add the query to the fused vector
        centre_momentum (float): the running centre's update rate
        project_depths (nn.ModuleList): one bias-free Linear per depth
        project_query (nn.Linear): bias-free, embed_dim -> width
        project_residual (nn.Linear): bias-free, width -> embed_dim, zero at
            the start, so training begins from the base's prediction
        bin_weights (buffer): [n_bins, n_samples]
        pyramid_mean, pyramid_spread (buffers): [n_depths, n_features]
        query_mean, query_spread (buffers): [embed_dim]
        residual_centre (buffer): [embed_dim]
        strength (buffer): scalar, the share of the correction applied
        calibrated (buffer): scalar bool, statistics filled
    """

    def __init__(
        self,
        hidden,
        n_depths,
        embed_dim,
        bins=(1, 2, 4),
        n_samples=50,
        width=128,
        bound=0.25,
        query_skip=False,
        centre_momentum=0.01,
    ):
        """
        ARGS:
            hidden (int): channels of each tapped layer, 160
            n_depths (int): tapped layers, 2
            embed_dim (int): the prediction's width, 1024
            bins (tuple): time resolutions pooled
            n_samples (int): samples pooled from each window's start, 1 s
            width (int): 64 or 128 in the paper
            bound (float): 0.15 or 0.25 in the paper
            query_skip (bool): the paper's eta
            centre_momentum (float): per training batch
        """
        super().__init__()
        self.width = int(width)
        self.bound = float(bound)
        self.query_skip = bool(query_skip)
        self.centre_momentum = float(centre_momentum)

        #every resolution's bins, as one [n_bins, n_samples] matrix
        self.register_buffer("bin_weights", bin_weights(list(bins), int(n_samples)))
        n_features = self.bin_weights.shape[0] * int(hidden)

        #training statistics; identity until `set_statistics` fills them
        self.register_buffer("pyramid_mean", torch.zeros(n_depths, n_features))
        self.register_buffer("pyramid_spread", torch.ones(n_depths, n_features))
        self.register_buffer("query_mean", torch.zeros(embed_dim))
        self.register_buffer("query_spread", torch.ones(embed_dim))
        self.register_buffer("residual_centre", torch.zeros(embed_dim))
        self.register_buffer("strength", torch.tensor(1.0))
        self.register_buffer("calibrated", torch.tensor(False))

        #P_l, the query map and W_s, all bias-free as in the paper
        self.project_depths = nn.ModuleList([nn.Linear(n_features, self.width, bias=False) for _ in range(n_depths)])
        self.project_query = nn.Linear(embed_dim, self.width, bias=False)
        self.project_residual = nn.Linear(self.width, embed_dim, bias=False)

        #zero, so the refined prediction starts as the base's own
        nn.init.zeros_(self.project_residual.weight)

    def pool(self, tapped, lengths):
        """Each depth's features averaged within every bin of the first second.

        ARGS:
            tapped (list): n_depths tensors [N, hidden, n_times], zero on padding
            lengths (torch.Tensor): [N] real samples per window

        RETURNS:
            [N, n_depths, n_bins * hidden] float32
        """
        n_samples = self.bin_weights.shape[1]

        #real samples only: a window shorter than a bin's span averages less
        steps = torch.arange(n_samples, device=lengths.device)
        valid = (steps.unsqueeze(0) < lengths.unsqueeze(1)).float()
        weights = self.bin_weights.unsqueeze(0) * valid.unsqueeze(1)
        counts = weights.sum(dim=2, keepdim=True).clamp_min(1.0)

        pooled = []
        for features in tapped:
            #the first second, zero-extended if the batch's windows are shorter
            head = features[:, :, :n_samples].float()
            if head.shape[2] < n_samples:
                head = F.pad(head, (0, n_samples - head.shape[2]))

            #[N, n_bins, n_samples] x [N, hidden, n_samples] -> [N, n_bins, hidden]
            means = torch.einsum("nbs,nhs->nbh", weights, head) / counts
            pooled.append(means.flatten(start_dim=1))

        #dim=1 is the depth axis
        return torch.stack(pooled, dim=1)

    @torch.no_grad()
    def set_statistics(self, pyramid_mean, pyramid_spread, query_mean, query_spread):
        """Freeze the standardisation statistics measured on training batches.

        ARGS:
            pyramid_mean, pyramid_spread (torch.Tensor): [n_depths, n_features]
            query_mean, query_spread (torch.Tensor): [embed_dim]
        """
        #a constant feature has no spread to divide by; its floor keeps it at zero
        self.pyramid_mean.copy_(pyramid_mean)
        self.pyramid_spread.copy_(pyramid_spread.clamp_min(1e-6))
        self.query_mean.copy_(query_mean)
        self.query_spread.copy_(query_spread.clamp_min(1e-6))
        self.calibrated.fill_(True)

    def residual(self, normalised, pyramids):
        """The bounded residual B(r), before centring.

        ARGS:
            normalised (torch.Tensor): [N, embed_dim] the unit-length prediction
            pyramids (torch.Tensor): [N, n_depths, n_features]

        RETURNS:
            [N, embed_dim] - each row's norm below `bound`
        """
        #q from the prediction, v_l from each depth, both standardised first
        query = torch.tanh(self.project_query((normalised - self.query_mean) / self.query_spread))
        standard = (pyramids - self.pyramid_mean) / self.pyramid_spread
        vectors = torch.stack(
            [torch.tanh(project(standard[:, depth])) for depth, project in enumerate(self.project_depths)],
            dim=1,
        )

        #dim=2 is the width axis, dim=1 the depth axis: one weight per depth
        weights = torch.softmax((vectors * query.unsqueeze(1)).sum(dim=2) / math.sqrt(self.width), dim=1)
        fused = (weights.unsqueeze(2) * vectors).sum(dim=1)
        if self.query_skip:
            fused = fused + query

        #|r|^2 rather than |r|: the bound stays differentiable at r = 0
        raw = self.project_residual(fused)
        squared = (raw * raw).sum(dim=1, keepdim=True)
        return self.bound * raw / torch.sqrt(self.bound**2 + squared)

    def forward(self, embedding, pyramids):
        """The refined prediction.

        ARGS:
            embedding (torch.Tensor): [N, embed_dim] the base's prediction z0
            pyramids (torch.Tensor): [N, n_depths, n_features] from `pool`

        RETURNS:
            dict:
                "refined" : [N, embed_dim] - z0 + strength * delta
                "base" : [N, embed_dim] - z0 at unit length
                "delta" : [N, embed_dim] - the centred residual
                "centre" : [1, embed_dim] - what was subtracted
        """
        normalised = F.normalize(embedding.float(), dim=1)
        bounded = self.residual(normalised, pyramids.float())

        #training: the batch's own mean, folded into the running centre;
        #evaluation: the running centre, fixed
        if self.training:
            centre = bounded.mean(dim=0, keepdim=True)
            self.residual_centre.lerp_(centre.squeeze(0).detach(), self.centre_momentum)
        else:
            centre = self.residual_centre.unsqueeze(0)
        delta = bounded - centre
        return {"refined": normalised + self.strength * delta, "base": normalised, "delta": delta, "centre": centre}
