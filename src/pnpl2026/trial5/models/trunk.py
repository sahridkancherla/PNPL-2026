"""The dilated convolution trunk: the tokeniser that feeds the transformer.

Its one job is to rewrite each time frame as a description of the local neural
event around it -- a transient, an oscillatory burst, a rise-then-fall -- so
that the transformer downstream relates DESCRIPTIONS rather than raw samples.

    same length in, same length out    latency is evidence, so the time axis is
                                       never spent. stride 1, no pooling

    reach of a few hundred ms          no motif is 60 ms long

    reach UNDER the window length      global integration is the transformer's
                                       job, and it is better at it

The third is the one that is easy to get wrong, and `receptive_field` exists so
that a config cannot quietly violate it. See `ReceptiveFieldError`.

⚠️ Channels FIRST here, `[batch, n_features, n_times]`. The transpose to
channels-last happens once, in `net.py`, after this module.
"""

from __future__ import annotations

import torch.nn as nn

#GroupNorm's group count. 8 divides every width this trunk is configured with,
#and small groups behave more like InstanceNorm, which is what a per-window
#normalisation scheme wants
DEFAULT_N_GROUPS = 8


class ReceptiveFieldError(ValueError):
    """Raised when the dilation ladder reaches past the input window."""


def dilation_schedule(depth, growth=2, period=None):
    """Work out the dilation each layer runs at.

    The ladder doubles with depth so that reach grows geometrically rather than
    linearly. `period` resets it to 1 every `period` layers, which is what
    stops a deep stack running away to a dilation of 1024 -- and what keeps a
    shallow stack from overshooting a short window.

    ARGS:
        depth (int): how many convolution layers
        growth (int): multiplier per layer. 1 disables dilation entirely
        period (int or None): reset the ladder every this many layers. None
            never resets

    RETURNS:
        list of int, one dilation per layer

    >>> dilation_schedule(5, growth=2, period=None)
    [1, 2, 4, 8, 16]
    >>> dilation_schedule(5, growth=2, period=4)
    [1, 2, 4, 8, 1]
    """
    dilations = []
    dilation = 1

    for layer in range(depth):
        #the reset fires at the START of a period, so layer 0 is always dense
        if period is not None and layer % period == 0:
            dilation = 1

        dilations.append(dilation)
        dilation *= growth

    return dilations


def receptive_field(dilations, kernel_size):
    """Frames one output can see, given the ladder.

    ARGS:
        dilations (list): one dilation per layer
        kernel_size (int): taps per layer, odd

    RETURNS:
        int: total span in frames
    """
    #each layer adds (k - 1) * dilation of reach, half on each side
    return 1 + sum((kernel_size - 1) * dilation for dilation in dilations)


def resolve_groups(n_features, requested=DEFAULT_N_GROUPS):
    """Largest group count not above `requested` that divides `n_features`.

    ⚠️ Resolved rather than asserted, deliberately. `GroupNorm` refuses a width
    its group count does not divide, so a hard-coded 8 turns any sweep over
    `trunk_hidden` or `d_model` into a crash on the first non-multiple-of-8
    value -- discovered after the job has already queued. Falling back to a
    divisor keeps the normalisation sensible and the run alive.

    ARGS:
        n_features (int): the width to normalise
        requested (int): the group count the config asked for

    RETURNS:
        int: a divisor of `n_features`, at worst 1

    >>> resolve_groups(160, 8)
    8
    >>> resolve_groups(6, 8)
    6
    >>> resolve_groups(7, 8)
    7
    """
    #count down from the requested value, so the result is as close to what
    #was asked for as divisibility allows
    for groups in range(min(requested, n_features), 0, -1):
        if n_features % groups == 0:
            return groups

    #unreachable for any positive width, since 1 divides everything
    return 1


class TrunkLayer(nn.Module):
    """One convolution, normalised, activated, and added back.

    ⚠️ The residual is NOT conditional on the widths matching. The reference
    implementation adds the skip only when `chin == chout`, which silently
    leaves its gradient path broken at both ends of the trunk -- the two
    width-changing layers get no residual at all. A 1x1 projection costs
    `chin * chout` and closes that gap.

    ATTRS:
        conv (nn.Conv1d): the dilated convolution, same-padded
        norm (nn.GroupNorm): over features, within each frame
        activation (nn.GELU): pointwise
        project (nn.Conv1d or None): 1x1 on the skip, when the width changes
    """

    def __init__(
        self,
        n_in,
        n_out,
        kernel_size=3,
        dilation=1,
        n_groups=DEFAULT_N_GROUPS,
        dropout=0.0,
    ):
        """
        ARGS:
            n_in (int): input width
            n_out (int): output width
            kernel_size (int): taps, odd so the padding is exactly `same`
            dilation (int): tap spacing
            n_groups (int): GroupNorm groups. Must divide n_out
            dropout (float): after the activation
        """
        super().__init__()

        #an even kernel has no centre tap, so same-padding is impossible and
        #the output comes back shifted half a sample
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        #resolved, not asserted: see resolve_groups. A width of 6 with 8 groups
        #requested normalises in 6 groups rather than raising
        n_groups = resolve_groups(n_out, n_groups)

        #this padding with an odd kernel is exactly `same`, at any dilation
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            n_in, n_out, kernel_size, padding=padding, dilation=dilation
        )

        #⚠️ GroupNorm, not BatchNorm. BatchNorm's running statistics are fitted
        #on the training recordings, and the holdout is a DIFFERENT recording
        #of the same subject -- head position drifts, SSS gain varies, and
        #sessions differ 30x in size. GroupNorm has no running state at all
        self.norm = nn.GroupNorm(n_groups, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        #only built when the widths differ; an identity skip is free
        self.project = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_in, n_times]

        RETURNS:
            out : [batch, n_out, n_times] - length unchanged
        """
        residual = x if self.project is None else self.project(x)

        x = self.activation(self.norm(self.conv(x)))
        if self.dropout is not None:
            x = self.dropout(x)

        return x + residual


class GatedLinearUnit(nn.Module):
    """A per-frame feature gate. Does NO time mixing.

    Produces twice the width with a 1x1 convolution, splits it, and uses one
    half to sigmoid-gate the other. Purely "which of these features matter at
    this instant" -- the reference runs it at `glu_context = 0`, so despite the
    name there is no temporal receptive field here at all.

    It sits OUTSIDE the residual, so it can attenuate the whole stream rather
    than only a branch.

    ATTRS:
        project (nn.Conv1d): D -> 2D, 1x1
    """

    def __init__(self, n_features):
        """
        ARGS:
            n_features (int): D, preserved across the module
        """
        super().__init__()
        self.project = nn.Conv1d(n_features, 2 * n_features, 1)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_features, n_times]

        RETURNS:
            gated : [batch, n_features, n_times]
        """
        #dim=1 is the feature axis; GLU halves whichever axis it is given, and
        #splitting time here would be a valid shape and complete nonsense
        return nn.functional.glu(self.project(x), dim=1)


class DilatedConvTrunk(nn.Module):
    """The stack: `depth` dilated layers, a GLU every `glu_every`.

    ATTRS:
        dilations (list): the ladder actually used
        reach (int): receptive field in frames
        layers (nn.ModuleList): the TrunkLayers
        gates (nn.ModuleList): one GatedLinearUnit or None per layer
        output_width (int): width handed to the transformer
    """

    def __init__(
        self,
        n_input,
        n_output,
        hidden=160,
        depth=5,
        kernel_size=3,
        dilation_growth=2,
        dilation_period=4,
        glu_every=2,
        n_groups=DEFAULT_N_GROUPS,
        dropout=0.0,
        n_frames=None,
    ):
        """Build the trunk and check its reach against the window.

        ARGS:
            n_input (int): width arriving from the front end
            n_output (int): width the transformer wants
            hidden (int): width of the interior layers
            depth (int): how many convolution layers
            kernel_size (int): taps per layer, odd
            dilation_growth (int): ladder multiplier per layer
            dilation_period (int or None): reset the ladder every this many
                layers. ⚠️ `None` with depth 5 and growth 2 gives a reach of 63
                frames, which overshoots a 50-frame window -- see `n_frames`
            glu_every (int): insert a gate after every this many layers.
                0 disables the gates
            n_groups (int): GroupNorm groups
            dropout (float): inside each layer
            n_frames (int or None): frames the trunk will actually be given.
                When set, a reach exceeding it raises rather than training a
                model that spends its last layer convolving with padding

        RAISES:
            ReceptiveFieldError: when the ladder reaches past `n_frames`
        """
        super().__init__()

        self.dilations = dilation_schedule(depth, dilation_growth, dilation_period)
        self.reach = receptive_field(self.dilations, kernel_size)

        #⚠️ THE CHECK THIS CLASS EXISTS FOR. A reach wider than the window means
        #the deepest layer's taps land in padding: at dilation 16 on 50 frames,
        #a third of what the filter reads is zeros for most output positions.
        #It is not merely wasteful -- it collapses the division of labour with
        #the transformer, which is the whole reason for a two-part design
        if n_frames is not None and self.reach > n_frames:
            raise ReceptiveFieldError(
                f"Trunk reach is {self.reach} frames but the window is only "
                f"{n_frames}. Dilations {self.dilations} overshoot by "
                f"{self.reach - n_frames} frames. Lower dilation_period "
                f"(4 gives {receptive_field(dilation_schedule(depth, dilation_growth, 4), kernel_size)} "
                f"frames) or reduce depth."
            )

        #the reference's shape: input width, then a flat interior, then out
        widths = [n_input] + [hidden] * (depth - 1) + [n_output]

        self.layers = nn.ModuleList(
            [
                TrunkLayer(
                    widths[layer],
                    widths[layer + 1],
                    kernel_size=kernel_size,
                    dilation=self.dilations[layer],
                    n_groups=n_groups,
                    dropout=dropout,
                )
                for layer in range(depth)
            ]
        )

        #⚠️ built off the resolved widths, never assumed: a gate is width
        #specific, and one sized for the wrong layer is a silent shape error
        gates = []
        for layer in range(depth):
            wants_gate = glu_every > 0 and (layer + 1) % glu_every == 0
            gates.append(GatedLinearUnit(widths[layer + 1]) if wants_gate else None)
        self.gates = nn.ModuleList(gates)

        self.output_width = n_output

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_input, n_times]

        RETURNS:
            out : [batch, n_output, n_times] - LENGTH UNCHANGED
        """
        for layer, gate in zip(self.layers, self.gates):
            x = layer(x)

            #applied after the residual add, so it gates the merged stream
            if gate is not None:
                x = gate(x)

        return x
