"""A Conformer stack over one word's window, in place of the dilated conv stack.

Trial 23's final night: the 2025 PNPL phoneme track was won by a compact
Conformer on raw MEG (MEGConformer, arXiv 2512.01443). Here it replaces the
per-word encoder's ten dilated convs (`WordEncoder(temporal="conformer")`);
the merger, 1x1, subject layer, final 1x1 pair and attention pool stay.

    [N, width_in, T] -> 1x1 to `width` -> + sinusoidal positions
      -> `depth` x ConformerBlock over [N, T, width], padding masked
      -> [N, width, T], zero on padding

ConformerBlock (Gulati et al. 2020, pre-norm, Macaron):

    x + 0.5 FFN(LN(x))
    x + MHSA(LN(x))              padded steps are never attended to
    x + ConvModule(LN(x))        1x1 to 2 width, GLU, depthwise conv k, LN,
                                 SiLU, 1x1; padding zeroed before the conv
    x + 0.5 FFN(LN(x)), then LN

No BatchNorm: a window's statistics never depend on the others in its batch.
The stack has no gated layers to tap, so Variant M's refiner cannot read it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_positions(n_times, width, device, dtype):
    """The fixed sinusoidal position table.

    ARGS:
        n_times (int)
        width (int): even
        device (torch.device), dtype (torch.dtype)

    RETURNS:
        [n_times, width]
    """
    position = torch.arange(n_times, device=device, dtype=torch.float32).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / width))
    table = torch.zeros(n_times, width, device=device, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * frequency)
    table[:, 1::2] = torch.cos(position * frequency)
    return table.to(dtype)


class FeedForward(nn.Module):
    """LayerNorm, widen x4, SiLU, dropout, narrow, dropout."""

    def __init__(self, width, dropout):
        """
        ARGS:
            width (int)
            dropout (float)
        """
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width), nn.SiLU(), nn.Dropout(dropout),
                                 nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, x):
        """[N, T, width] -> [N, T, width]"""
        return self.net(x)


class ConvModule(nn.Module):
    """The Conformer's convolution module, padding kept at zero."""

    def __init__(self, width, kernel_size, dropout):
        """
        ARGS:
            width (int)
            kernel_size (int): odd
            dropout (float)
        """
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.pointwise_in = nn.Conv1d(width, 2 * width, 1)
        self.depthwise = nn.Conv1d(width, width, kernel_size, padding=kernel_size // 2, groups=width)
        self.depth_norm = nn.LayerNorm(width)
        self.pointwise_out = nn.Conv1d(width, width, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        """
        ARGS:
            x (torch.Tensor): [N, T, width]
            valid (torch.Tensor): [N, T, 1], 1 on real steps

        RETURNS:
            [N, T, width]
        """
        #[N, T, width] -> [N, width, T] for the convs; dim 1 is the GLU's halves
        h = self.pointwise_in(self.norm(x).transpose(1, 2))
        h = nn.functional.glu(h, dim=1) * valid.transpose(1, 2)
        h = self.depthwise(h).transpose(1, 2)
        h = nn.functional.silu(self.depth_norm(h)).transpose(1, 2)
        return self.dropout(self.pointwise_out(h).transpose(1, 2))


class ConformerBlock(nn.Module):
    """Half feed-forward, self-attention, convolution, half feed-forward."""

    def __init__(self, width, n_heads, kernel_size, dropout):
        """
        ARGS:
            width (int): divisible by n_heads
            n_heads (int)
            kernel_size (int)
            dropout (float)
        """
        super().__init__()
        self.feed_in = FeedForward(width, dropout)
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, n_heads, dropout=dropout, batch_first=True)
        self.attention_dropout = nn.Dropout(dropout)
        self.conv = ConvModule(width, kernel_size, dropout)
        self.feed_out = FeedForward(width, dropout)
        self.out_norm = nn.LayerNorm(width)

    def forward(self, x, valid):
        """
        ARGS:
            x (torch.Tensor): [N, T, width]
            valid (torch.Tensor): [N, T] bool, True on real steps

        RETURNS:
            [N, T, width]
        """
        weight = valid.unsqueeze(2).to(x.dtype)
        x = x + 0.5 * self.feed_in(x)

        #key_padding_mask is True where a key must be ignored
        h = self.attention_norm(x)
        attended, _ = self.attention(h, h, h, key_padding_mask=~valid, need_weights=False)
        x = x + self.attention_dropout(attended)
        x = x + self.conv(x, weight)
        x = x + 0.5 * self.feed_out(x)
        return self.out_norm(x) * weight


class ConformerStack(nn.Module):
    """The stack that replaces ConvSequence: same call, same output layout.

    ATTRS:
        project (nn.Conv1d): width_in -> width
        blocks (nn.ModuleList): of ConformerBlock
    """

    def __init__(self, width_in, width, depth=4, n_heads=4, kernel_size=15, dropout=0.1):
        """
        ARGS:
            width_in (int): the channels it receives
            width (int): its own width and output channels
            depth (int)
            n_heads (int): must divide width
            kernel_size (int): odd
            dropout (float)
        """
        super().__init__()
        if width % n_heads or width % 2:
            raise ValueError(f"conformer width {width} must be even and divisible by {n_heads} heads")
        if kernel_size % 2 == 0:
            raise ValueError(f"conformer kernel {kernel_size} must be odd")
        self.width = width
        self.project = nn.Conv1d(width_in, width, 1)
        self.blocks = nn.ModuleList(ConformerBlock(width, n_heads, kernel_size, dropout) for _ in range(depth))

    def forward_with_taps(self, x, valid=None, tap_layers=()):
        """ConvSequence's call: [N, width_in, T] -> [N, width, T], nothing tapped.

        ARGS:
            x (torch.Tensor): [N, width_in, T]
            valid (torch.Tensor or None): [N, 1, T], 1 on real samples
            tap_layers (tuple): must be empty

        RETURNS:
            x : [N, width, T], zero on padding
            tapped : [] -- a Conformer has no gated layers to tap
        """
        if tap_layers:
            raise ValueError("The Conformer stack has no layers for Variant M to tap.")
        n_words, _, n_times = x.shape
        mask = torch.ones(n_words, 1, n_times, device=x.device, dtype=x.dtype) if valid is None else valid.to(x.dtype)
        steps_valid = mask[:, 0, :] > 0

        #[N, width, T] -> [N, T, width] for the blocks, positions added once
        h = (self.project(x) * mask).transpose(1, 2)
        h = h + sinusoidal_positions(n_times, self.width, h.device, h.dtype).unsqueeze(0)
        for block in self.blocks:
            h = block(h, steps_valid)

        #back to [N, width, T], padding at zero for the final 1x1 pair and the pool
        return h.transpose(1, 2) * mask, []

    def forward(self, x, valid=None):
        """[N, width_in, T] -> [N, width, T]"""
        return self.forward_with_taps(x, valid)[0]
