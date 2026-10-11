"""The sentence transformer: d'Ascoli et al.'s, 16 layers x d 1024 x 16 heads.

A lean copy of trial 8's `FlatTransformer` -- pre-norm blocks, rotary
positions over the word index, a final LayerNorm -- with three changes:

    attention dropout and feed-forward dropout are separate knobs. The paper
    gives attention dropout 0.1 and nothing else

    no per-block reach and no causal mode; every word reads the whole sentence

    the rotary tables are built once per forward, not once per block -- at 16
    blocks that is 15 redundant builds

This is where 98.8% of the reference's 203.7M parameters sit.

⚠️ Channels LAST here, `[sequences, words, d_model]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotary_tables(n_times, head_dim, device, dtype, base=10000.0):
    """Build the cosine and sine tables for rotary position embedding.

    ARGS:
        n_times (int): sequence length
        head_dim (int): per-head width, even
        device (torch.device)
        dtype (torch.dtype)
        base (float): geometric base of the frequency ladder

    RETURNS:
        cos : [1, 1, n_times, head_dim]
        sin : [1, 1, n_times, head_dim]
    """
    #fast pairs resolve neighbours, slow pairs carry long separations
    exponent = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    inverse_frequency = 1.0 / (base ** (exponent / head_dim))
    positions = torch.arange(n_times, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inverse_frequency)

    #both members of a pair share an angle, hence the duplication
    angles = torch.cat([angles, angles], dim=-1)
    cos = angles.cos().to(dtype).view(1, 1, n_times, head_dim)
    sin = angles.sin().to(dtype).view(1, 1, n_times, head_dim)
    return cos, sin


def apply_rotary(x, cos, sin):
    """Rotate each feature pair of `x` by its position's angle.

    ARGS:
        x (torch.Tensor): [batch, n_heads, n_times, head_dim]
        cos (torch.Tensor): [1, 1, n_times, head_dim]
        sin (torch.Tensor): [1, 1, n_times, head_dim]

    RETURNS:
        same shape as `x`
    """
    #(a + ib)(cos + i sin), with the pairs split across the two halves
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    rotated = torch.cat([-second, first], dim=-1)
    return x * cos + rotated * sin


class DropPath(nn.Module):
    """Drop a whole residual branch per sample, in training. Off by default.

    ATTRS:
        p (float): drop probability
    """

    def __init__(self, p=0.0):
        """
        ARGS:
            p (float): 0 makes this the identity
        """
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"DropPath p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, ...]

        RETURNS:
            same shape, some samples zeroed and the rest rescaled
        """
        if self.p == 0.0 or not self.training:
            return x

        #one draw per SAMPLE, broadcast over every other axis
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        keep = 1.0 - self.p
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep
        return x * mask / keep


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with rotary positions, through the fused SDPA kernel.

    ATTRS:
        n_heads (int)
        head_dim (int): d_model // n_heads, even
        query_key (nn.Linear): D -> 2D
        value (nn.Linear): D -> D
        out (nn.Linear): D -> D
        dropout_p (float): on the attention weights, training only
    """

    def __init__(self, d_model, n_heads, dropout=0.0):
        """
        ARGS:
            d_model (int): D
            n_heads (int): must divide D, to an even head width
            dropout (float): attention dropout
        """
        super().__init__()

        #checked at build, so a bad config fails before the job queues
        if d_model % n_heads != 0 or (d_model // n_heads) % 2 != 0:
            raise ValueError(f"d_model {d_model} / n_heads {n_heads} must be an even integer.")

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query_key = nn.Linear(d_model, 2 * d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def _heads(self, x):
        """[batch, n_times, d_model] -> [batch, n_heads, n_times, head_dim]."""
        batch, n_times, _ = x.shape
        return x.view(batch, n_times, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, rotary, mask=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]
            rotary (tuple): (cos, sin) from `rotary_tables`
            mask (torch.Tensor or None): [batch, 1, n_times, n_times] bool,
                True where a query may attend to a key

        RETURNS:
            [batch, n_times, d_model]
        """
        batch, n_times, d_model = x.shape
        query, key = self.query_key(x).chunk(2, dim=-1)
        query, key = self._heads(query), self._heads(key)
        value = self._heads(self.value(x))

        #rotary on queries and keys only: it says where to look, not what is there
        cos, sin = rotary
        query, key = apply_rotary(query, cos, sin), apply_rotary(key, cos, sin)

        #softmax(QK^T / sqrt(head_dim)) V, never materialising the weights
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )

        #contiguous: the transpose leaves a view that view() refuses
        merged = attended.transpose(1, 2).contiguous()
        return self.out(merged.view(batch, n_times, d_model))


class FeedForward(nn.Module):
    """Widen, GELU, narrow.

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
            same shape
        """
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm attention, then pre-norm feed-forward, both residual.

    ATTRS:
        norm_attention (nn.LayerNorm)
        attention (MultiHeadSelfAttention)
        norm_feed_forward (nn.LayerNorm)
        feed_forward (FeedForward)
        drop_path (DropPath)
    """

    def __init__(self, d_model, n_heads, ffn_expansion=4, attention_dropout=0.1, dropout=0.0, drop_path=0.0):
        """
        ARGS:
            d_model (int): D
            n_heads (int)
            ffn_expansion (int)
            attention_dropout (float): on the attention weights
            dropout (float): inside the feed-forward
            drop_path (float): stochastic depth for this block
        """
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, attention_dropout)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, ffn_expansion, dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, rotary, mask=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]
            rotary (tuple): (cos, sin)
            mask (torch.Tensor or None): see `attention_mask`

        RETURNS:
            same shape
        """
        #the residual carries the un-normalised stream
        x = x + self.drop_path(self.attention(self.norm_attention(x), rotary, mask))
        x = x + self.drop_path(self.feed_forward(self.norm_feed_forward(x)))
        return x


class SentenceTransformer(nn.Module):
    """`depth` blocks over the words of each sequence, then a final norm.

    ATTRS:
        blocks (nn.ModuleList)
        final_norm (nn.LayerNorm): pre-norm leaves the last block un-normalised
        d_model (int)
        head_dim (int)
    """

    def __init__(
        self,
        d_model=1024,
        depth=16,
        n_heads=16,
        ffn_expansion=4,
        attention_dropout=0.1,
        dropout=0.0,
        drop_path=0.0,
    ):
        """
        ARGS:
            d_model (int): 1024 in the reference
            depth (int): 16
            n_heads (int): 16
            ffn_expansion (int): 4
            attention_dropout (float): 0.1, the paper's
            dropout (float): feed-forward dropout; the paper gives none
            drop_path (float): MAXIMUM stochastic depth, ramped over blocks
        """
        super().__init__()

        #ramped: early blocks are dropped less than late ones
        rates = [drop_path * layer / max(depth - 1, 1) for layer in range(depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, ffn_expansion, attention_dropout, dropout, rates[layer])
                for layer in range(depth)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.head_dim = d_model // n_heads

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]
            valid (torch.Tensor or None): [batch, n_times] bool, False on
                padding, which no word may attend to

        RETURNS:
            [batch, n_times, d_model]; padding rows are finite but meaningless
        """
        #one mask and one rotary table, shared by every block
        mask = attention_mask(valid, x.shape[1], x.device)
        rotary = rotary_tables(x.shape[1], self.head_dim, x.device, x.dtype)
        for block in self.blocks:
            x = block(x, rotary, mask)
        return self.final_norm(x)


def attention_mask(valid, n_times, device):
    """The boolean mask SDPA reads: True where a query may attend to a key.

    ARGS:
        valid (torch.Tensor or None): [batch, n_times] bool, False on padding
        n_times (int)
        device (torch.device)

    RETURNS:
        [batch, 1, n_times, n_times] bool, or None when nothing is padded
    """
    if valid is None:
        return None

    #keys at padding are hidden from every query
    allowed = valid.view(-1, 1, 1, n_times).expand(-1, 1, n_times, n_times)

    #⚠️ a query with no visible key gives SDPA a NaN row, which reaches the
    #loss through the backward pass. Every position may see itself
    diagonal = torch.eye(n_times, dtype=torch.bool, device=device).view(1, 1, n_times, n_times)
    return allowed | diagonal
