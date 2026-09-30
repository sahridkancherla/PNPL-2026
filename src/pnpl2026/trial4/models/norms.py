"""Zipformer's normalisation, activations and stochastic depth.

Three small pieces borrowed from Zipformer (ICLR 2024), each of which that
paper measured as better than the Conformer default it replaces:

    BiasNorm   instead of LayerNorm  -- keeps length information
    SwooshR/L  instead of Swish      -- measured better
    DropPath                         -- ordinary stochastic depth

⚠️ Every module here expects **channels last**, `[batch, n_times, n_features]`.
The encoder works in that layout because attention does; only the convolution
module transposes, and it transposes back. A `[B, D, T]` tensor handed to
BiasNorm normalises over time instead of over features and will not raise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#the constants that make each Swoosh pass near the origin with unit-ish slope,
#taken from the Zipformer paper rather than re-derived
SWOOSH_R_OFFSET = 0.313261687
SWOOSH_L_OFFSET = 0.035


class SwooshR(nn.Module):
    """`log(1 + exp(x - 1)) - 0.08x - 0.313261687`.

    Zipformer's general-purpose replacement for Swish. It is the one to use in
    a feed-forward or convolution module, where the branch is expected to be
    doing something most of the time.
    """

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): any shape

        RETURNS:
            same shape as `x`
        """
        #softplus rather than log1p(exp(.)) so large inputs do not overflow
        return nn.functional.softplus(x - 1.0) - 0.08 * x - SWOOSH_R_OFFSET


class SwooshL(nn.Module):
    """`log(1 + exp(x - 4)) - 0.08x - 0.035`.

    Shifted so that it sits near zero over a wide negative range, which is what
    a branch with a bypass wants: "normally off" is a cheap default and the
    module has to earn its contribution.
    """

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): any shape

        RETURNS:
            same shape as `x`
        """
        return nn.functional.softplus(x - 4.0) - 0.08 * x - SWOOSH_L_OFFSET


class BiasNorm(nn.Module):
    """`x * exp(gamma) / RMS[x - bias]`, over the feature axis.

    LayerNorm subtracts the mean and divides by the standard deviation, which
    throws away the length of the vector entirely. A network that wants to
    carry "how much signal is here" in the magnitude of its activations then
    has to smuggle it through some other channel. BiasNorm divides by the RMS
    of `x - bias` instead: the learned `bias` gives the network a direction it
    can grow along without the norm cancelling it.

    ATTRS:
        bias (nn.Parameter): [n_features], the direction excluded from the RMS
        log_scale (nn.Parameter): scalar gamma, exponentiated
        eps (float): floor inside the RMS, keeping its gradient finite
    """

    def __init__(self, n_features, eps=1e-5):
        """
        ARGS:
            n_features (int): D, the size of the last axis
            eps (float): added to the mean square before the square root
        """
        super().__init__()

        #starts at zero, so the first forward pass is a plain RMS norm
        self.bias = nn.Parameter(torch.zeros(n_features))

        #a log-parameterised scale cannot go negative, and starts at exactly 1
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.eps = eps

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]

        RETURNS:
            normalised : same shape as `x`
        """
        #dim=-1 is the feature axis; reducing over time instead would mix
        #timepoints together and is the silent failure this module invites
        mean_square = (x - self.bias).pow(2).mean(dim=-1, keepdim=True)

        #the RMS of the BIASED tensor scales the UNBIASED one, which is the
        #whole point -- the bias direction survives the normalisation
        return x * torch.exp(self.log_scale) / (mean_square + self.eps).sqrt()


class DropPath(nn.Module):
    """Drop a whole residual branch for a whole sample, at training time.

    Stochastic depth. At `p=0.1` one sample in ten skips this branch entirely,
    which is a stronger regulariser than dropout on the same branch because it
    forces the residual path alone to be a usable network.

    ⚠️ Trial 3 set this to zero because a three-epoch run cannot distinguish
    "regularised" from "worse". Over thirty epochs it can.

    ATTRS:
        p (float): probability of dropping the branch, per sample
    """

    def __init__(self, p=0.0):
        """
        ARGS:
            p (float): drop probability. 0 makes this module the identity
        """
        super().__init__()

        if not 0.0 <= p < 1.0:
            raise ValueError(f"DropPath p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, ...] -- only the batch axis is indexed

        RETURNS:
            same shape as `x`, some samples zeroed
        """
        #identity at eval time and at p=0, with no random draw at all
        if self.p == 0.0 or not self.training:
            return x

        #one Bernoulli per SAMPLE, broadcast over every other axis
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        keep = 1.0 - self.p
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep

        #dividing by `keep` holds the expected value, so eval needs no rescaling
        return x * mask / keep
