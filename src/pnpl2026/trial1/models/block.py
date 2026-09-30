"""
to do list:

1. add like a @staticmethod validation function to the encoder block. to make sure the data fits
what we need. 
2. add more error check/dimension check functions
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

"""
hyperparameters
"""

#hyperparameters
BRANCH_KERNELS = (7, 19, 25)
BRANCH_DILATIONS = (1, 1, 4)

# --- AI, 2026-09-24 ---------------------------------------------------------
# Span 49 samples = 196 ms, about one cycle at 5 Hz. Theta is where the
# speech envelope lives, and an envelope detector wants roughly a cycle.
QUADRATURE_KERNEL = 25
QUADRATURE_DILATION = 2

#sqrt has an infinite gradient at zero, so the argument never reaches it
QUADRATURE_EPSILON = 1e-6
# --- end AI -----------------------------------------------------------------


# --- AI, 2026-09-24. UNVERIFIED -- never run against real data. -------------
class QuadratureBranch(nn.Module):
    """A depthwise filter pair read as an amplitude envelope.

    Two depthwise convolutions of identical span run over the input and the
    branch returns `sqrt(a^2 + b^2)` per channel per timepoint. Where the pair
    settles 90 degrees apart -- a quadrature pair -- that is the envelope of
    the analytic signal, and it does not care where in its cycle the
    oscillation happens to sit.

    Why it earns a place: single-trial MEG phase is not reproducible across
    repetitions of a word, but induced power is. Every other branch here is
    linear and therefore phase-sensitive, so a response arriving a few
    milliseconds late partially cancels itself. This branch cannot tell.

    ⚠️ Nothing constrains the pair to be orthogonal. Quadrature is what the
    branch is FOR, not something it enforces -- the sum of squares is simply
    the cheapest form that is invariant to which of the pair fires.

    ⚠️ This is the block's only nonlinear branch, and its output is
    non-negative, so it carries a DC offset the linear branches do not. The
    mixer reads it as one more block of features and the FFN's BatchNorm
    absorbs the offset.

    ATTRS:
        in_phase (nn.Conv1d): one half of the pair, depthwise
        quadrature (nn.Conv1d): the other half, same span, independent weights
        epsilon (float): floor inside the sqrt, keeping its gradient finite
    """

    def __init__(
        self,
        n_channels,
        kernel_size=QUADRATURE_KERNEL,
        dilation=QUADRATURE_DILATION,
        epsilon=QUADRATURE_EPSILON,
    ):
        """Build the filter pair.

        ARGS:
            n_channels (int): D, preserved -- the branch is depthwise
            kernel_size (int): taps per filter, odd so padding is exactly same
            dilation (int): tap spacing, widening the span without more weights
            epsilon (float): added inside the sqrt before it is taken
        """
        super().__init__()

        #the pair must share a span, or their squares are not two views of one
        #thing and the sum stops meaning amplitude
        self.in_phase = EncoderBlock.conv_depthwise(n_channels, kernel_size, dilation)
        self.quadrature = EncoderBlock.conv_depthwise(n_channels, kernel_size, dilation)
        self.epsilon = epsilon

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            envelope : [batch, n_channels, n_times] - non-negative
        """
        #two phase-shifted views of the same band, if the pair learns to be one
        real_part = self.in_phase(x)
        imaginary_part = self.quadrature(x)

        #the epsilon is load-bearing: without it a silent channel returns NaN
        #gradients the moment both halves output exactly zero
        return torch.sqrt(real_part**2 + imaginary_part**2 + self.epsilon)
# --- end AI -----------------------------------------------------------------


@dataclass(eq=False)
class EncoderBlock(nn.Module):

    n_channels: int
    kernels: tuple = BRANCH_KERNELS
    dilations: tuple = BRANCH_DILATIONS
    ffn_expansion: int = 2

    # --- AI, 2026-09-24. One flag, so the branch is ablatable in a config. ---
    with_quadrature: bool = True
    quadrature_kernel: int = QUADRATURE_KERNEL
    quadrature_dilation: int = QUADRATURE_DILATION
    # --- end AI ---

    spans: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """
        Assemble the block
        """
        #nn.Module refuses submodule assignment before its registries exist
        super().__init__()

        self.branches = self.build_branches(
            self.n_channels,
            self.kernels,
            self.dilations,
            self.with_quadrature,
            self.quadrature_kernel,
            self.quadrature_dilation,
        )

        # --- AI, 2026-09-24 ---
        #counted off the built list, NOT len(kernels): the quadrature branch is
        #one more block of D features the mixer has to read, and a stale count
        #here is a silent shape error at the concat
        self.mixer = self.build_mixer(self.n_channels, len(self.branches))
        # --- end AI ---
        self.feed_forward = self.build_feed_forward(
            self.n_channels, self.ffn_expansion
        )

    @staticmethod
    def conv_depthwise(n_channels, kernel_size, dilation):
        # --- AI, 2026-09-24: guard for the to-do at the top of this file. ---
        #an even kernel has no true centre, so the padding below floors and the
        #output comes back SHORT by `dilation`. That surfaces as a confusing
        #RuntimeError at the residual add rather than here, where it started
        if kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd for exact same-padding, got "
                f"{kernel_size}. An even kernel shifts the output half a "
                f"sample and shortens it by {dilation}."
            )
        # --- end AI ---

        #odd kernel plus this padding is exactly `same`, at any dilation
        padding = dilation * (kernel_size - 1) // 2
        return nn.Conv1d(

            #in and out channels remain the same
            n_channels,
            n_channels,

            #kernel size will vary
            kernel_size=kernel_size,
            padding=padding,

            #dilation will vary
            dilation=dilation,

            #for weight efficiency
            groups=n_channels,
            bias=False,
        )

    @classmethod
    def build_branches(
        cls,
        n_channels,
        kernels,
        dilations,
        with_quadrature=True,
        quadrature_kernel=QUADRATURE_KERNEL,
        quadrature_dilation=QUADRATURE_DILATION,
    ):
        """
        Establish the branches for a single neural net block.
        builds one linear branch per kernel, plus the quadrature branch
        """
        #make a list of branches. every branch will have different values of kernels and dilations
        branches = [
            cls.conv_depthwise(n_channels, k, d)
            for k, d in zip(kernels, dilations)]

        # --- AI, 2026-09-24 ---
        #appended LAST, so the linear branches keep the feature offsets they
        #had before this branch existed and an ablation changes nothing else
        if with_quadrature:
            branches.append(
                QuadratureBranch(
                    n_channels, quadrature_kernel, quadrature_dilation
                )
            )
        # --- end AI ---

        return nn.ModuleList(branches)
        

    @staticmethod
    def build_mixer(n_channels, n_branches):
        """
        mixer that is applied once the channels are concatenated 
        """
        return nn.Conv1d(n_branches * n_channels, n_channels, kernel_size=1)

    @staticmethod
    def build_feed_forward(n_channels, expansion):
       
        #hidden layer size
        hidden = expansion * n_channels
        
        return nn.Sequential(
            #still debating the normalisation function
            nn.BatchNorm1d(n_channels),

            #MLP. potentially add more layers
            nn.Conv1d(n_channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, n_channels, kernel_size=1),
        )


    # --- AI, 2026-09-24: dropped a @staticmethod decorator that sat here. ---
    # With it, nn.Module.__call__ passed the input tensor as `self` and left
    # `windows` unfilled, so every forward pass raised TypeError.
    def forward(self, windows):
        
        #concatenate every branch into one
        concatenated_branch = torch.cat([branch(windows) for branch in self.branches], dim=1)

        #apply the mixer
        mixed = self.mixer(concatenated_branch)

        #residual 1
        windows = windows + mixed

        #impelements the build_feed_forward function
        y = self.feed_forward(windows)

        #residual 2
        output = windows + y
        
        return output
