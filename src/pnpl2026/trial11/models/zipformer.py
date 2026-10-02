"""A Zipformer-style multi-rate encoder over one sentence's frame stream.

The scaffold of Yao et al. 2023 (Zipformer) with trial 8's plain pre-norm
blocks inside it, not the paper's block. Every stack starts and ends at the
full frame rate:

    x  [batch, n_frames, d]  at 50 Hz
      -> Downsample /k      learned weighted average of each k frames
      -> n TransformerBlocks at 50/k Hz, attention over the whole sentence
      -> Upsample  xk       each token repeated k times
      -> Bypass             x + c * (y - x), c learned per channel
    x  [batch, n_frames, d]  at 50 Hz

With factors 1, 2, 4, 8, 4, 2 the rates run 50 -> 25 -> 12.5 -> 6.25 ->
12.5 -> 25 Hz: a token at 6.25 Hz spans 160 ms, about one per word gap. The
residual stream stays at 50 Hz throughout, so fine timing survives every
coarse stack through the bypass.

Factors all 1 is Z0, the control: the same blocks and bypasses, no change of
rate. Z1 against Z0 is what the multi-rate structure itself is worth.

⚠️ Channels LAST here, `[batch, n_frames, d_model]`. The net owns the one
transpose from the convolutional trunk.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import TransformerBlock, attention_mask, build_norm


class Downsample(nn.Module):
    """Merge each run of `factor` frames into one, by a learned weighted average.

    The weights are a softmax over `factor` learned scalars, shared by every
    group and channel -- the paper's `SimpleDownsample`. Padding frames get no
    weight: the average is renormalised over the real frames of each group.

    ATTRS:
        factor (int): frames per output token
        logits (nn.Parameter or None): [factor], softmaxed into the weights;
            None at factor 1, where there is nothing to weigh
    """

    def __init__(self, factor):
        """
        ARGS:
            factor (int): frames merged into one token; 1 is the identity
        """
        super().__init__()
        self.factor = factor

        #zeros: a plain mean at initialisation. No parameter at factor 1, or
        #it would sit in the optimiser and never receive a gradient
        if factor > 1:
            self.logits = nn.Parameter(torch.zeros(factor))
        else:
            self.register_parameter("logits", None)

    def forward(self, x, valid):
        """
        ARGS:
            x (torch.Tensor): [batch, n_frames, d], n_frames a multiple of factor
            valid (torch.Tensor): [batch, n_frames] bool

        RETURNS:
            tokens : [batch, n_frames // factor, d]
            token_valid : [batch, n_frames // factor] bool, True where any of
                the group's frames is real
        """
        if self.factor == 1:
            return x, valid

        batch, n_frames, width = x.shape
        n_tokens = n_frames // self.factor

        #[batch, n_tokens, factor, d]: the frames each token is made from
        grouped = x.view(batch, n_tokens, self.factor, width)
        grouped_valid = valid.view(batch, n_tokens, self.factor)

        #[batch, n_tokens, factor]: the shared weights, zeroed on padding
        weights = self.logits.softmax(dim=0).view(1, 1, self.factor)
        weights = weights * grouped_valid

        #renormalised over real frames; a token with none is all zero
        total = weights.sum(dim=2, keepdim=True).clamp(min=1e-6)
        tokens = (grouped * (weights / total).unsqueeze(3)).sum(dim=2)
        return tokens, grouped_valid.any(dim=2)


def upsample(tokens, factor):
    """Repeat every token `factor` times, back to the frame rate.

    ARGS:
        tokens (torch.Tensor): [batch, n_tokens, d]
        factor (int): repeats per token

    RETURNS:
        frames : [batch, n_tokens * factor, d]
    """
    if factor == 1:
        return tokens
    return tokens.repeat_interleave(factor, dim=1)


class Bypass(nn.Module):
    """Blend a stack's output with its input, per channel: x + c * (y - x).

    c near 0 keeps the input and skips the stack; c near 1 takes the stack's
    output. c is a sigmoid of a learned value, so it stays inside [0, 1].

    ATTRS:
        logits (nn.Parameter): [d], sigmoid gives c
    """

    def __init__(self, width):
        """
        ARGS:
            width (int): d_model
        """
        super().__init__()

        #zeros: c = 0.5, an even blend at initialisation
        self.logits = nn.Parameter(torch.zeros(width))

    def forward(self, original, processed):
        """
        ARGS:
            original (torch.Tensor): [batch, n_frames, d], the stack's input
            processed (torch.Tensor): [batch, n_frames, d], its upsampled output

        RETURNS:
            [batch, n_frames, d]
        """
        blend = self.logits.sigmoid()
        return original + blend * (processed - original)


class ZipformerStack(nn.Module):
    """Downsample, `depth` blocks at the lower rate, upsample, bypass.

    ATTRS:
        factor (int): rate divisor for this stack
        downsample (Downsample)
        blocks (nn.ModuleList): of TransformerBlock
        bypass (Bypass or None): None on the first stack, which has no input
            to blend back -- as in the paper
    """

    def __init__(self, d_model, factor, depth, n_heads, ffn_expansion, dropout, drop_paths, norm, with_bypass):
        """
        ARGS:
            d_model (int): width in and out
            factor (int): 1, 2, 4, 8 ...
            depth (int): blocks at the lower rate, at least 1
            n_heads (int): attention heads per block
            ffn_expansion (int): FFN hidden multiple
            dropout (float): inside each block
            drop_paths (list): one stochastic-depth rate per block
            norm (str): "layer" or "bias"
            with_bypass (bool): False only for the first stack
        """
        super().__init__()
        if depth < 1:
            raise ValueError(f"A stack needs at least one block, got depth {depth}")

        self.factor = factor
        self.downsample = Downsample(factor)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                    drop_path=rate,
                    norm=norm,
                )
                for rate in drop_paths
            ]
        )
        self.bypass = Bypass(d_model) if with_bypass else None

    def forward(self, x, valid):
        """
        ARGS:
            x (torch.Tensor): [batch, n_frames, d] at the full rate
            valid (torch.Tensor): [batch, n_frames] bool

        RETURNS:
            [batch, n_frames, d] at the full rate
        """
        tokens, token_valid = self.downsample(x, valid)

        #padding tokens are hidden from every query; rotary counts in tokens
        #at THIS rate, so each stack learns its own timescale
        mask = attention_mask(token_valid, False, tokens.shape[1], tokens.device)
        for block in self.blocks:
            tokens = block(tokens, mask)

        processed = upsample(tokens, self.factor)
        if self.bypass is None:
            return processed
        return self.bypass(x, processed)


class ZipformerEncoder(nn.Module):
    """Stacks in sequence, then a final norm, all at the full frame rate.

    ATTRS:
        factors (tuple): one rate divisor per stack
        stacks (nn.ModuleList): of ZipformerStack
        final_norm (nn.Module): pre-norm blocks leave the last sum unnormalised
    """

    def __init__(
        self,
        d_model=256,
        factors=(1, 2, 4, 8, 4, 2),
        depths=(1, 1, 1, 1, 1, 1),
        n_heads=8,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        norm="layer",
    ):
        """
        ARGS:
            d_model (int): width throughout
            factors (tuple): rate divisor per stack; all 1 is the flat control
            depths (tuple): blocks per stack, parallel to `factors`
            n_heads (int): attention heads
            ffn_expansion (int): FFN hidden multiple
            dropout (float): inside every block
            drop_path (float): MAXIMUM stochastic depth, ramped over all blocks
            norm (str): "layer" or "bias"
        """
        super().__init__()
        if len(factors) != len(depths):
            raise ValueError(f"{len(factors)} factors for {len(depths)} depths")
        for factor in factors:
            if not isinstance(factor, int) or isinstance(factor, bool) or factor < 1:
                raise ValueError(f"Each factor must be a positive int, got {factor!r}")

        self.factors = tuple(factors)

        #ramped across every block of every stack, as FlatTransformer ramps it
        n_blocks = sum(depths)
        rates = [drop_path * index / max(n_blocks - 1, 1) for index in range(n_blocks)]

        self.stacks = nn.ModuleList()
        start = 0
        for index, (factor, depth) in enumerate(zip(factors, depths)):
            self.stacks.append(
                ZipformerStack(
                    d_model,
                    factor,
                    depth,
                    n_heads,
                    ffn_expansion,
                    dropout,
                    rates[start : start + depth],
                    norm,
                    with_bypass=index > 0,
                )
            )
            start += depth

        self.final_norm = build_norm(norm, d_model)

    def check_frames(self, n_frames):
        """Raise unless every factor tiles `n_frames` exactly.

        ARGS:
            n_frames (int): frames the encoder will be given
        """
        #a factor that does not tile would fail the reshape or, worse, group
        #frames across the end of one window into the next
        uneven = [factor for factor in self.factors if n_frames % factor != 0]
        if uneven:
            raise ValueError(f"{n_frames} frames do not divide by factors {uneven}")

    def forward(self, x, valid):
        """
        ARGS:
            x (torch.Tensor): [batch, n_frames, d]
            valid (torch.Tensor): [batch, n_frames] bool

        RETURNS:
            [batch, n_frames, d], normalised
        """
        self.check_frames(x.shape[1])
        for stack in self.stacks:
            x = stack(x, valid)
        return self.final_norm(x)
