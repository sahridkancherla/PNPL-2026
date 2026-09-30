"""The multi-rate encoder: stacks of Conformer blocks at different frame rates.

A flat Conformer runs every block at one frame rate, which is wasteful at both
ends: the early blocks pay full price to model slow structure, and the late
blocks cannot look further than their kernel without many more layers. A
DOWNWARD PYRAMID of stacks fixes both -- work cheaply at a coarse rate, where a
small kernel spans a long time.

    stack A   rate 1   144 wide   k=31   125 frames   fine detail
      down 2
    stack B   rate 2   192 wide   k=15    63 frames
      down 2
    stack C   rate 4   256 wide   k=9     32 frames   cheap, long-range

The head pools every stack, not just the last -- see `TrialFourNet`.

⚠️ **This is a pyramid, not a U-Net, and the difference is deliberate.**
Zipformer (ICLR 2024) comes back UP with skip connections because an ASR
encoder emits a label per frame for CTC or a transducer: it is dense
prediction, and the skip restores the resolution the bottleneck discarded.
**This model pools over time and emits 50 logits.** There is no output
resolution to reconstruct, so an upsample and skip would cost ~12% of the
encoder's compute to solve a problem this task does not have.

The U-Net remains reachable -- `stack_divisors=(1, 2, 4, 2)` builds it, skips
included -- and is the ablation, not the default.

⚠️ The depthwise kernels SHRINK as the rate drops, deliberately. Each one is
chosen so its span in milliseconds stays near 300 ms. A constant kernel size
across a rate change quadruples the receptive field per stack, and a model that
only sees slow drift trains perfectly happily.

⚠️ Channels last throughout, `[batch, n_times, n_features]`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .conformer import ConformerBlock
from .norms import BiasNorm


class SimpleDownsample(nn.Module):
    """Halve (or quarter) the frame rate by a learned weighted average.

    Not a strided convolution: the weights are a softmax over the positions
    inside each group, shared across every channel, so the layer can learn to
    behave like "take the first", "average both", or anything between, and
    cannot learn a spatial filter by accident.

    ATTRS:
        factor (int): how many frames collapse into one
        weights (nn.Parameter): [factor], softmaxed at use
    """

    def __init__(self, factor):
        """
        ARGS:
            factor (int): frames per output frame. 2 halves the rate
        """
        super().__init__()

        if factor < 2:
            raise ValueError(f"downsample factor must be >= 2, got {factor}")
        self.factor = factor

        #starting at zeros makes the softmax uniform, i.e. a plain average
        self.weights = nn.Parameter(torch.zeros(factor))

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]

        RETURNS:
            [batch, ceil(n_times / factor), n_features]
        """
        batch, n_times, n_features = x.shape

        #a length that is not a multiple of the factor needs padding, and
        #padding by REPEATING the last frame beats padding with zeros, which
        #would drag the final group's average towards nothing
        remainder = n_times % self.factor
        if remainder:
            pad = self.factor - remainder
            x = torch.cat([x, x[:, -1:, :].expand(batch, pad, n_features)], dim=1)
            n_times = n_times + pad

        grouped = x.view(batch, n_times // self.factor, self.factor, n_features)

        #dim=0 is the within-group axis of `weights`; the view broadcasts it
        #across batch, group and feature
        weights = self.weights.softmax(dim=0).view(1, 1, self.factor, 1)
        return (grouped * weights).sum(dim=2)


class SimpleUpsample(nn.Module):
    """Raise the frame rate by repeating each frame.

    Nearest-neighbour, on purpose. The stack that follows has attention and a
    convolution and can smooth whatever it wants; a learned transposed
    convolution here would just be a second place for checkerboard artefacts to
    come from.

    ATTRS:
        factor (int): how many output frames each input frame becomes
    """

    def __init__(self, factor):
        """
        ARGS:
            factor (int): repeat count per frame
        """
        super().__init__()

        if factor < 2:
            raise ValueError(f"upsample factor must be >= 2, got {factor}")
        self.factor = factor

    def forward(self, x, n_times=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times_in, n_features]
            n_times (int or None): trim to exactly this length. Needed because
                the matching downsample may have padded

        RETURNS:
            [batch, n_times_in * factor, n_features] or trimmed to `n_times`
        """
        #dim=1 is time; repeat_interleave on the wrong axis silently duplicates
        #features instead and the shapes still line up
        upsampled = x.repeat_interleave(self.factor, dim=1)

        #the downsample rounded up, so coming back gives >= the original length
        if n_times is not None:
            if upsampled.shape[1] < n_times:
                raise ValueError(
                    f"Upsample produced {upsampled.shape[1]} frames, fewer than "
                    f"the {n_times} requested -- the rate bookkeeping is wrong."
                )
            upsampled = upsampled[:, :n_times, :]
        return upsampled


class ConformerStack(nn.Module):
    """Several Conformer blocks at one frame rate, sharing attention weights.

    The first block computes the attention weights; the rest borrow them. That
    is Zipformer's trick and it is why the later blocks in a stack carry no
    query/key projection at all.

    ATTRS:
        blocks (nn.ModuleList): the blocks, in order
    """

    def __init__(
        self,
        n_blocks,
        d_model,
        n_heads,
        kernel_size,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.0,
        with_quadrature=True,
        reuse_attention=True,
    ):
        """
        ARGS:
            n_blocks (int): how many blocks at this rate
            d_model (int): D, constant within the stack
            n_heads (int): attention heads
            kernel_size (int): depthwise taps, odd
            ffn_expansion (int): hidden multiple in the feed-forwards
            dropout (float): inside every submodule
            drop_path (float): stochastic depth, same for every block here
            with_quadrature (bool): passed to every convolution module
            reuse_attention (bool): False gives every block its own weights
        """
        super().__init__()

        self.reuse_attention = reuse_attention

        #only the first block needs a query/key projection when weights are
        #reused; building one in the others would be dead weight in the checkpoint
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    kernel_size=kernel_size,
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                    drop_path=drop_path,
                    with_quadrature=with_quadrature,
                    computes_attention=(index == 0 or not reuse_attention),
                )
                for index in range(n_blocks)
            ]
        )

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, d_model]

        RETURNS:
            [batch, n_times, d_model] - shape preserved
        """
        weights = None
        for block in self.blocks:
            #passing None makes the block compute its own; passing weights
            #makes it borrow. The stack decides which, not the block
            x, computed = block(x, weights=weights)
            if self.reuse_attention:
                weights = computed
        return x


class MultiRateEncoder(nn.Module):
    """Stem, then a U-Net of Conformer stacks at several frame rates.

    ATTRS:
        stem (nn.Sequential): 1x1 lift from the input width to the first stack
        stacks (nn.ModuleList): the ConformerStacks, in order
        rate_changes (nn.ModuleList): one entry per stack, possibly None
        width_changes (nn.ModuleList): one Linear per stack, possibly Identity
        skip_projections (nn.ModuleDict): width fixes for the U-Net skips
        output_width (int): D of the last stack, what the head receives
    """

    def __init__(
        self,
        n_input,
        stack_blocks=(3, 4, 4, 3),
        stack_widths=(192, 256, 384, 256),
        stack_divisors=(1, 2, 4, 2),
        stack_kernels=(31, 15, 9, 15),
        n_heads=4,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.0,
        with_quadrature=True,
        reuse_attention=True,
    ):
        """Assemble the encoder.

        ARGS:
            n_input (int): the width arriving from the front end
            stack_blocks (tuple): blocks per stack
            stack_widths (tuple): d_model per stack
            stack_divisors (tuple): frame-rate divisor per stack, relative to
                the input rate. 1 is full rate, 4 is a quarter
            stack_kernels (tuple): depthwise taps per stack, odd
            n_heads (int): attention heads, every stack
            ffn_expansion (int): hidden multiple in the feed-forwards
            dropout (float): inside every submodule
            drop_path (float): maximum stochastic depth, ramped linearly with
                depth so early stacks are barely regularised
            with_quadrature (bool): the trial-1 envelope branch, ablatable
            reuse_attention (bool): Zipformer's attention-weight sharing
        """
        super().__init__()

        lengths = {len(stack_blocks), len(stack_widths), len(stack_divisors),
                   len(stack_kernels)}

        #four parallel lists that disagree build a network nobody intended, and
        #zip() would silently truncate to the shortest
        if len(lengths) != 1:
            raise ValueError(
                f"stack_blocks, stack_widths, stack_divisors and stack_kernels "
                f"must be the same length; got lengths {sorted(lengths)}"
            )

        self.stack_divisors = tuple(int(d) for d in stack_divisors)
        self.stack_widths = tuple(int(w) for w in stack_widths)
        self.output_width = int(stack_widths[-1])

        #the stem only decides how the input components are mixed into the
        #first stack's width; the stacks own the time axis
        self.stem = nn.Sequential(
            nn.Linear(n_input, int(stack_widths[0])),
            BiasNorm(int(stack_widths[0])),
        )

        self.stacks = nn.ModuleList()
        self.rate_changes = nn.ModuleList()
        self.width_changes = nn.ModuleList()
        self.skip_projections = nn.ModuleDict()

        #drop_path ramps from 0 to the configured maximum with depth: the early
        #blocks carry the representation everything else builds on, so dropping
        #them is more damaging than dropping a late one
        total_blocks = sum(stack_blocks)
        blocks_so_far = 0

        previous_divisor = 1
        previous_width = int(stack_widths[0])
        seen_widths = {}

        for index, n_blocks in enumerate(stack_blocks):
            divisor = int(stack_divisors[index])
            width = int(stack_widths[index])

            #a rate change must be an integer ratio in one direction or the
            #other; 3 -> 2 is neither and would need a resampler
            self.rate_changes.append(self._rate_change(previous_divisor, divisor))

            #width changes ride on a Linear after the rate change, so the
            #downsample averages in the OLD width and the projection is cheaper
            self.width_changes.append(
                nn.Linear(previous_width, width)
                if previous_width != width
                else nn.Identity()
            )

            #a U-Net skip exists whenever this rate was visited before
            if divisor in seen_widths and seen_widths[divisor] != width:
                self.skip_projections[str(index)] = nn.Linear(
                    seen_widths[divisor], width
                )

            depth_fraction = (blocks_so_far + n_blocks) / max(total_blocks, 1)
            self.stacks.append(
                ConformerStack(
                    n_blocks=int(n_blocks),
                    d_model=width,
                    n_heads=n_heads,
                    kernel_size=int(stack_kernels[index]),
                    ffn_expansion=ffn_expansion,
                    dropout=dropout,
                    drop_path=drop_path * depth_fraction,
                    with_quadrature=with_quadrature,
                    reuse_attention=reuse_attention,
                )
            )

            blocks_so_far += n_blocks
            seen_widths[divisor] = width
            previous_divisor, previous_width = divisor, width

    @staticmethod
    def _rate_change(previous_divisor, divisor):
        """Build the resampler that takes one stack's rate to the next's.

        ARGS:
            previous_divisor (int): rate divisor of the preceding stack
            divisor (int): rate divisor of the stack being built

        RETURNS:
            a SimpleDownsample, a SimpleUpsample, or None for no change
        """
        if divisor == previous_divisor:
            return None

        if divisor > previous_divisor:
            factor, remainder = divmod(divisor, previous_divisor)
            if remainder:
                raise ValueError(
                    f"rate divisor {previous_divisor} -> {divisor} is not an "
                    f"integer downsample"
                )
            return SimpleDownsample(factor)

        factor, remainder = divmod(previous_divisor, divisor)
        if remainder:
            raise ValueError(
                f"rate divisor {previous_divisor} -> {divisor} is not an "
                f"integer upsample"
            )
        return SimpleUpsample(factor)

    def forward(self, x, static_bias=None, return_all_stacks=False):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_input] channels LAST
            static_bias (torch.Tensor or None): [batch, stack_widths[0]], added
                to every frame after the stem. This is how the per-window scale
                that `RobustInstanceNorm` discarded gets back into the network
            return_all_stacks (bool): True returns every stack's output rather
                than only the last. A pooled classifier wants all of them --
                see `TrialFourNet`'s multi-scale head

        RETURNS:
            [batch, n_times_out, output_width] at the last stack's frame rate,
            or a list of one such tensor per stack when `return_all_stacks`
        """
        x = self.stem(x)

        #unsqueeze(1) puts the bias on the time axis so it broadcasts to every
        #frame; without it the shapes still align only when T happens to match
        if static_bias is not None:
            x = x + static_bias.unsqueeze(1)

        #lengths are recorded per rate on the way down, so an upsample can trim
        #back to exactly what the matching downsample was handed
        lengths = {1: x.shape[1]}
        skips = {}
        outputs = []

        for index, stack in enumerate(self.stacks):
            divisor = self.stack_divisors[index]
            change = self.rate_changes[index]

            if isinstance(change, SimpleUpsample):
                #trim to the length this rate had on the way down, if known
                x = change(x, n_times=lengths.get(divisor))
            elif change is not None:
                x = change(x)

            x = self.width_changes[index](x)

            #the skip is added AFTER the width change, so both sides agree
            skip = skips.get(divisor)
            if skip is not None:
                #⚠️ nn.ModuleDict has no .get(), so the membership test is
                #explicit; a projection only exists where the widths differ
                key = str(index)
                if key in self.skip_projections:
                    x = x + self.skip_projections[key](skip)
                else:
                    x = x + skip

            x = stack(x)

            lengths[divisor] = x.shape[1]
            skips[divisor] = x
            outputs.append(x)

        return outputs if return_all_stacks else x
