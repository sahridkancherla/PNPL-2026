"""A flat pre-norm transformer encoder with rotary positions.

Flat, meaning **one frame rate throughout** -- no pyramid, no multi-rate
stacks, no upsampling. Trial 3 had all three and scored 29.47% against trial
2's 34.70% with less than half the parameters. Multi-rate exists to make
attention cheap on long sequences; at 50 tokens there is no compute to save,
so the mechanism was solving a problem this task does not have.

This module is where the capacity goes. At `depth 8, d_model 256` it is ~6.3M
of the network's ~8.0M parameters -- 79% -- which is the single structural
lesson from the reference model, where the transformer is 98.8% of 203.7M. Our
best result to date came from our SMALLEST model, and the sequence model is the
part that was starved.

Trial 6 changes two things, neither of which changes the maths:

    attention runs through `F.scaled_dot_product_attention`, which never
    materialises the [batch, heads, T, T] weights. At 50 frames that is a
    nicety; at C6's 250 frames the explicit version stores ~18 GB of them for
    the backward pass at batch 512

    depth 0 is valid, for C2: no blocks, only the final norm, so the static
    bias still arrives and the pooled vector's scale is still controlled

⚠️ Channels LAST here, `[batch, n_times, d_model]`, because that is what
attention and LayerNorm want. `net.py` owns the one transpose.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norms import BiasNorm, DropPath


def rotary_tables(n_times, head_dim, device, dtype, base=10000.0):
    """Build the cosine and sine tables for rotary position embedding.

    Rotary encodes position by ROTATING each pair of feature dimensions by an
    angle proportional to position. The dot product of two rotated vectors then
    depends only on their separation, which makes the scheme relative -- and
    relative is what a word window wants, since the evidence for a word sits at
    a LATENCY after onset rather than at an absolute index.

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

    #a geometric ladder: fast pairs resolve nearby frames, slow pairs carry
    #long separations
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
    #split into the two halves forming the pairs, then rotate as a complex
    #multiply: (a + ib) * (cos + i sin)
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    rotated = torch.cat([-second, first], dim=-1)
    return x * cos + rotated * sin


def build_norm(kind, n_features):
    """Pick the normalisation a block uses.

    ARGS:
        kind (str): "layer" or "bias"
        n_features (int): D

    RETURNS:
        an nn.Module normalising the last axis
    """
    if kind == "layer":
        return nn.LayerNorm(n_features)
    if kind == "bias":
        return BiasNorm(n_features)
    raise KeyError(f"Unknown norm {kind!r}; known: 'layer', 'bias'")


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with rotary relative positions.

    ATTRS:
        n_heads (int): head count
        head_dim (int): d_model // n_heads, even
        query_key (nn.Linear): D -> 2D, split after the fact
        value (nn.Linear): D -> D
        out (nn.Linear): D -> D
        dropout_p (float): on the attention weights, training only
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        """
        ARGS:
            d_model (int): D, the width of the block
            n_heads (int): head count. Must divide d_model
            dropout (float): applied to the attention weights
        """
        super().__init__()

        #an indivisible width silently truncates the last head's features
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by n_heads {n_heads}"
            )

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        #⚠️ checked here rather than inside rotary_tables: a constraint that
        #only fires on the first forward pass is one a sweep discovers after
        #the job has already queued
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"d_model {d_model} over n_heads {n_heads} gives head_dim "
                f"{self.head_dim}, which is odd. Rotary rotates feature pairs."
            )

        self.query_key = nn.Linear(d_model, 2 * d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        #a float, not an nn.Dropout: the fused kernel applies it to weights
        #it never materialises. Parameter-free either way, so the state dict
        #matches trial 5's
        self.dropout_p = dropout

    def _heads(self, x):
        """Split the feature axis into heads.

        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            [batch, n_heads, n_times, head_dim]
        """
        batch, n_times, _ = x.shape
        return x.view(batch, n_times, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            out : [batch, n_times, d_model]
        """
        batch, n_times, d_model = x.shape

        query, key = self.query_key(x).chunk(2, dim=-1)
        query, key = self._heads(query), self._heads(key)
        value = self._heads(self.value(x))

        #rotary goes on queries and keys only, never values: it encodes WHERE
        #to look, not WHAT is there
        cos, sin = rotary_tables(n_times, self.head_dim, x.device, x.dtype)
        query, key = apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)

        #softmax(QK^T / sqrt(head_dim)) V over the KEY axis, identical to
        #trial 5's explicit version; the default scale is 1/sqrt(head_dim)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        #back to [batch, n_times, d_model]; contiguous because the transpose
        #leaves a non-contiguous view that view() refuses
        merged = attended.transpose(1, 2).contiguous()
        return self.out(merged.view(batch, n_times, d_model))


class FeedForward(nn.Module):
    """Widen, activate, narrow. Two thirds of a block's parameters.

    ATTRS:
        net (nn.Sequential): D -> expansion*D -> D
    """

    def __init__(self, d_model, expansion=4, dropout=0.0):
        """
        ARGS:
            d_model (int): D
            expansion (int): hidden multiple
            dropout (float): after the activation and after the projection
        """
        super().__init__()

        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            same shape as `x`
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention, then pre-norm feed-forward, both residual.

    Pre-norm rather than post-norm because post-norm needs a warmup to train at
    all at this depth, and pre-norm is what makes a deep stack trainable with
    an ordinary schedule.

    ATTRS:
        norm_attention (nn.Module): before the attention branch
        attention (MultiHeadSelfAttention)
        norm_feed_forward (nn.Module): before the FFN branch
        feed_forward (FeedForward)
        drop_path (DropPath): stochastic depth on BOTH branches
    """

    def __init__(
        self,
        d_model,
        n_heads,
        ffn_expansion=4,
        dropout=0.0,
        drop_path=0.0,
        norm="layer",
    ):
        """
        ARGS:
            d_model (int): D
            n_heads (int): attention heads
            ffn_expansion (int): hidden multiple in the FFN
            dropout (float): inside attention and the FFN
            drop_path (float): stochastic depth for this block
            norm (str): "layer" or "bias", see build_norm
        """
        super().__init__()

        self.norm_attention = build_norm(norm, d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)

        self.norm_feed_forward = build_norm(norm, d_model)
        self.feed_forward = FeedForward(d_model, ffn_expansion, dropout)

        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            same shape as `x`
        """
        #the residual carries the un-normalised stream, which is what lets a
        #dropped branch leave the block as an exact identity
        x = x + self.drop_path(self.attention(self.norm_attention(x)))
        x = x + self.drop_path(self.feed_forward(self.norm_feed_forward(x)))
        return x


class FlatTransformer(nn.Module):
    """`depth` identical blocks at one frame rate, then a final norm.

    ATTRS:
        blocks (nn.ModuleList): the TransformerBlocks
        final_norm (nn.Module): applied once at the end, as pre-norm requires
        d_model (int): width in and out
    """

    def __init__(
        self,
        d_model=256,
        depth=8,
        n_heads=8,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        norm="layer",
    ):
        """
        ARGS:
            d_model (int): D
            depth (int): how many blocks. 0 leaves the final norm alone
            n_heads (int): attention heads per block
            ffn_expansion (int): hidden multiple in each FFN
            dropout (float): inside every block
            drop_path (float): MAXIMUM stochastic depth, ramped with depth
            norm (str): "layer" or "bias"
        """
        super().__init__()

        #ramped, not flat: the literature is consistent that early blocks want
        #less stochastic depth than late ones, and a flat rate at depth 8 makes
        #the first block as droppable as the last
        rates = [drop_path * layer / max(depth - 1, 1) for layer in range(depth)]

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                    drop_path=rates[layer],
                    norm=norm,
                )
                for layer in range(depth)
            ]
        )

        #⚠️ pre-norm leaves the output of the last block un-normalised, so a
        #final norm is not optional decoration -- without it the pooled vector's
        #scale drifts with depth
        self.final_norm = build_norm(norm, d_model)
        self.d_model = d_model

    def forward(self, x, static_bias=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]
            static_bias (torch.Tensor or None): [batch, d_model], added to
                every frame. This is where the amplitude the instance norm
                discarded comes back in

        RETURNS:
            out : [batch, n_times, d_model]
        """
        #unsqueeze(1) explicitly, naming the TIME axis it broadcasts over; a
        #bare broadcast against [batch, d_model] would align on the wrong axis
        if static_bias is not None:
            x = x + static_bias.unsqueeze(1)

        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)
