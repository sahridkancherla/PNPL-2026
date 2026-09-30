"""The Conformer block: macaron feed-forwards around attention and convolution.

Gulati et al. (2020), with the pieces this task actually needs:

    x = x + 0.5 * FFN_1(x)
    x = x + MHSA(x)          rotary relative positions
    x = x + Conv(x)          depthwise, plus the quadrature envelope branch
    x = x + 0.5 * FFN_2(x)
    x = BiasNorm(x)

Why a Conformer and not the plain pre-norm attention stack trial 3 used: a
Conformer won the 2025 PNPL phoneme track on this exact dataset at 65.83%
F1-macro against a 6.10% baseline, and that paper measured the Conformer as
+9.0% over the convolutional architecture it replaced. Attention alone gives
the block no cheap way to model a 300 ms local response; the depthwise
convolution is that way.

⚠️ Everything here is **channels last**, `[batch, n_times, n_features]`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .norms import BiasNorm, DropPath, SwooshR

#sqrt has an infinite gradient at zero, so the argument never reaches it
QUADRATURE_EPSILON = 1e-6


def rotary_tables(n_times, head_dim, device, dtype, base=10000.0):
    """Build the cosine and sine tables for rotary position embedding.

    Rotary encodes position by ROTATING each pair of feature dimensions by an
    angle proportional to the position. The dot product between two rotated
    vectors then depends only on their separation, which is what makes it a
    relative scheme -- and relative is what a word window wants, since the
    evidence for a word sits at a latency after onset rather than at an
    absolute index.

    ARGS:
        n_times (int): T, sequence length
        head_dim (int): per-head feature count. Must be even
        device (torch.device): where to build the tables
        dtype (torch.dtype): floating type of the tables
        base (float): geometric base for the frequency ladder

    RETURNS:
        cos : [1, 1, n_times, head_dim]
        sin : [1, 1, n_times, head_dim]
    """
    #each rotation acts on a PAIR of dimensions, so an odd head_dim leaves one
    #dimension unpaired and the rotation is not well defined
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for rotary, got {head_dim}")

    #a geometric ladder of frequencies: fast pairs resolve nearby positions,
    #slow pairs carry long-range separation
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inverse_frequency = 1.0 / (base ** (exponent / head_dim))

    positions = torch.arange(n_times, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)

    #each angle is used by both members of its pair, hence the duplication
    angles = torch.cat([angles, angles], dim=-1)

    cos = angles.cos().to(dtype).view(1, 1, n_times, head_dim)
    sin = angles.sin().to(dtype).view(1, 1, n_times, head_dim)
    return cos, sin


def apply_rotary(x, cos, sin):
    """Rotate each feature pair of `x` by the angles in `cos`/`sin`.

    ARGS:
        x (torch.Tensor): [batch, n_heads, n_times, head_dim]
        cos (torch.Tensor): [1, 1, n_times, head_dim]
        sin (torch.Tensor): [1, 1, n_times, head_dim]

    RETURNS:
        rotated : same shape as `x`
    """
    #split into the two halves that form the pairs, then rotate as a complex
    #multiply: (a + ib) * (cos + i sin)
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    rotated = torch.cat([-second, first], dim=-1)
    return x * cos + rotated * sin


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with rotary positions, able to reuse attention weights.

    Zipformer's observation: within a stack of blocks all running at the same
    frame rate, the attention WEIGHTS barely change from block to block, while
    the values they combine do. So the first block in a stack computes the
    weights and every later block borrows them, paying only for its own value
    and output projections.

    ⚠️ This matters here for a reason beyond elegance: this model trains on a
    thermally-throttled laptop, where halving the cost of attention is halving
    wall-clock, not just FLOPs.

    ATTRS:
        n_heads (int): how many attention heads
        head_dim (int): d_model // n_heads
        query_key (nn.Linear or None): None when this block always borrows
        value (nn.Linear): D -> D
        out (nn.Linear): D -> D
    """

    def __init__(self, d_model, n_heads, dropout=0.0, computes_weights=True):
        """
        ARGS:
            d_model (int): D, the width of the block
            n_heads (int): head count. Must divide d_model
            dropout (float): applied to the attention weights
            computes_weights (bool): False builds no query/key projection at
                all, because this block will always be handed weights
        """
        super().__init__()

        #an indivisible width silently truncates the last head's features
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by n_heads {n_heads}"
            )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        #⚠️ checked HERE and not just inside rotary_tables: rotary rotates
        #feature PAIRS, so an odd head_dim is invalid, and a constraint that
        #only fires on the first forward pass is one a config sweep discovers
        #after the job has already been queued
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"d_model {d_model} over n_heads {n_heads} gives head_dim "
                f"{self.head_dim}, which is odd. Rotary rotates feature pairs, "
                f"so head_dim must be even."
            )

        self.computes_weights = computes_weights

        #one projection producing both query and key, split after the fact
        self.query_key = nn.Linear(d_model, 2 * d_model) if computes_weights else None
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x):
        """Split the feature axis into heads.

        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            [batch, n_heads, n_times, head_dim]
        """
        batch, n_times, _ = x.shape
        return x.view(batch, n_times, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, weights=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]
            weights (torch.Tensor or None): [batch, n_heads, n_times, n_times]
                attention weights to reuse. None computes them

        RETURNS:
            output : [batch, n_times, d_model]
            weights : the weights used, so the next block can borrow them
        """
        batch, n_times, d_model = x.shape

        #a block built without a query/key projection cannot invent one
        if weights is None:
            if self.query_key is None:
                raise RuntimeError(
                    "This attention module was built with computes_weights=False "
                    "and was called without weights to reuse."
                )

            query, key = self.query_key(x).chunk(2, dim=-1)
            query, key = self._heads(query), self._heads(key)

            #rotary is applied to queries and keys only, never to values: it
            #encodes WHERE to look, not WHAT is there
            cos, sin = rotary_tables(n_times, self.head_dim, x.device, x.dtype)
            query, key = apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)

            scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)

            #dim=-1 is the KEY axis: the weights must sum to one over what is
            #being attended to, not over the queries doing the attending
            weights = self.dropout(scores.softmax(dim=-1))

        value = self._heads(self.value(x))
        pooled = weights @ value

        #back to [batch, n_times, d_model] before the output projection
        pooled = pooled.transpose(1, 2).reshape(batch, n_times, d_model)
        return self.out(pooled), weights


class ConvolutionModule(nn.Module):
    """Gated depthwise convolution, with the quadrature envelope branch.

    The standard Conformer convolution module -- pointwise expansion, GLU,
    depthwise convolution, activation, pointwise projection -- plus one
    addition carried over from trial 1.

    **The quadrature branch.** A second depthwise pair runs over the same input
    and the module takes `sqrt(a^2 + b^2)`. Where the pair settles 90 degrees
    apart that is the envelope of the analytic signal, and it does not care
    where in its cycle the oscillation sits. Single-trial MEG *phase* is not
    reproducible across repetitions of a word; induced *power* is. Every other
    path here is linear and therefore phase-sensitive, so a response arriving a
    few milliseconds late partially cancels itself. This branch cannot tell.

    ⚠️ Nothing constrains the pair to be orthogonal. Quadrature is what the
    branch is FOR, not something it enforces.

    ATTRS:
        depthwise (nn.Conv1d): the linear path, groups = d_model
        in_phase (nn.Conv1d or None): one half of the quadrature pair
        quadrature (nn.Conv1d or None): the other half, same span
    """

    def __init__(
        self,
        d_model,
        kernel_size,
        dropout=0.0,
        with_quadrature=True,
        epsilon=QUADRATURE_EPSILON,
    ):
        """
        ARGS:
            d_model (int): D, preserved end to end
            kernel_size (int): depthwise taps, odd so padding is exactly same
            dropout (float): on the module's output
            with_quadrature (bool): False is the ablation
            epsilon (float): floor inside the sqrt, keeping its gradient finite
        """
        super().__init__()

        #an even kernel has no true centre, so same-padding floors and the
        #output comes back short -- which surfaces at the residual add, not here
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for exact same-padding, got {kernel_size}"
            )

        self.with_quadrature = with_quadrature
        self.epsilon = epsilon
        padding = (kernel_size - 1) // 2

        self.norm_in = BiasNorm(d_model)

        #a Linear is a 1x1 convolution without the transpose, and this module
        #is in channels-last layout at both ends
        self.pointwise_in = nn.Linear(d_model, 2 * d_model)

        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, groups=d_model, bias=False
        )

        if with_quadrature:
            #the pair must SHARE a span, or their squares are not two views of
            #one thing and the sum stops meaning amplitude
            self.in_phase = nn.Conv1d(
                d_model, d_model, kernel_size, padding=padding, groups=d_model, bias=False
            )
            self.quadrature = nn.Conv1d(
                d_model, d_model, kernel_size, padding=padding, groups=d_model, bias=False
            )
        else:
            self.in_phase = None
            self.quadrature = None

        #the envelope branch doubles what the output projection has to read
        mixed = 2 * d_model if with_quadrature else d_model
        self.norm_mid = BiasNorm(mixed)
        self.activation = SwooshR()
        self.pointwise_out = nn.Linear(mixed, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            [batch, n_times, d_model]
        """
        #the gate is what makes this a *gated* convolution: half the expanded
        #width decides how much of the other half passes
        gated = nn.functional.glu(self.pointwise_in(self.norm_in(x)), dim=-1)

        #convolutions want channels first; this is the only transpose pair here
        channels_first = gated.transpose(1, 2)
        paths = [self.depthwise(channels_first)]

        if self.with_quadrature:
            real_part = self.in_phase(channels_first)
            imaginary_part = self.quadrature(channels_first)

            #the epsilon is load-bearing: without it a silent channel returns
            #NaN gradients the moment both halves output exactly zero
            paths.append(
                torch.sqrt(real_part**2 + imaginary_part**2 + self.epsilon)
            )

        mixed = torch.cat(paths, dim=1).transpose(1, 2)
        return self.dropout(self.pointwise_out(self.activation(self.norm_mid(mixed))))


class FeedForward(nn.Module):
    """One of the two macaron feed-forwards.

    Ordinary two-layer position-wise MLP. It is at the block's periphery rather
    than in the middle because Macaron-Net measured that placement as more
    stable, and the Conformer paper's ablation agrees.

    ATTRS:
        net (nn.Sequential): norm, expand, activate, project, drop
    """

    def __init__(self, d_model, expansion=4, dropout=0.0):
        """
        ARGS:
            d_model (int): D, preserved
            expansion (int): hidden width multiple. Speech Conformers use 4
            dropout (float): applied twice, after the activation and the output
        """
        super().__init__()

        hidden = expansion * d_model
        self.net = nn.Sequential(
            BiasNorm(d_model),
            nn.Linear(d_model, hidden),
            SwooshR(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            [batch, n_times, d_model]
        """
        return self.net(x)


class ConformerBlock(nn.Module):
    """Macaron feed-forwards around attention and convolution, shape preserved.

    The half-step residuals on the two feed-forwards are from the paper and are
    not cosmetic: a full-step residual on both makes the block's output variance
    grow faster with depth than the normalisation can absorb.

    ATTRS:
        ffn_first (FeedForward): the leading half-step branch
        attention (MultiHeadSelfAttention): may borrow weights from a sibling
        convolution (ConvolutionModule): the local path
        ffn_second (FeedForward): the trailing half-step branch
        norm_out (BiasNorm): closes the block
    """

    def __init__(
        self,
        d_model,
        n_heads,
        kernel_size,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.0,
        with_quadrature=True,
        computes_attention=True,
    ):
        """
        ARGS:
            d_model (int): D, preserved end to end
            n_heads (int): attention heads
            kernel_size (int): depthwise taps, odd
            ffn_expansion (int): hidden multiple in both feed-forwards
            dropout (float): inside every submodule
            drop_path (float): stochastic depth on each of the four branches
            with_quadrature (bool): passed to the convolution module
            computes_attention (bool): False means this block always borrows
                attention weights and builds no query/key projection
        """
        super().__init__()

        self.ffn_first = FeedForward(d_model, ffn_expansion, dropout)
        self.attention = MultiHeadSelfAttention(
            d_model, n_heads, dropout, computes_weights=computes_attention
        )
        self.convolution = ConvolutionModule(
            d_model, kernel_size, dropout, with_quadrature
        )
        self.ffn_second = FeedForward(d_model, ffn_expansion, dropout)
        self.norm_out = BiasNorm(d_model)

        #one DropPath per branch, so each can be skipped independently
        self.drop_ffn_first = DropPath(drop_path)
        self.drop_attention = DropPath(drop_path)
        self.drop_convolution = DropPath(drop_path)
        self.drop_ffn_second = DropPath(drop_path)

    def forward(self, x, weights=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]
            weights (torch.Tensor or None): attention weights to reuse

        RETURNS:
            output : [batch, n_times, d_model] - same shape as the input
            weights : the attention weights, for the next block to borrow
        """
        #the 0.5 is the half-step residual, and it is in the paper for a reason
        x = x + self.drop_ffn_first(0.5 * self.ffn_first(x))

        attended, weights = self.attention(x, weights=weights)
        x = x + self.drop_attention(attended)

        x = x + self.drop_convolution(self.convolution(x))
        x = x + self.drop_ffn_second(0.5 * self.ffn_second(x))

        return self.norm_out(x), weights
