"""The per-word encoder: one word's window in, one vector out.

Variant B's fixed front end and dilated conv trunk, run over a window that is
zero-padded to 3 s and masked to its real length:

    [N, 73, 750]   whitened components; real for `length` samples, then zero
      -> RobustInstanceNorm    median/MAD over REAL samples only
      -> BaselineCorrect       0-0.5 s, the design's -- see ../info/variant-d.md
      -> FixedLowpassDecimate  20 Hz, /5 -> 150 frames
      -> GaussianNoise, ComponentDropout            training only
      -> spatial, initial_linear, DilatedConvTrunk
      -> FlatTransformer over frames   depth 0 in Variant D: only its final
                                       norm and the amplitude bias. D0: 2
      -> AttentionPool                 over real frames only
    [N, 256]

⚠️ Padding is zero from the first operation to the last, matching the holdout,
where a window cut at a sentence's end is zero-padded the same way. The FIR and
the convolutions still see those zeros near a window's end; that edge effect is
identical in training and at test time, which is what matters.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .frontend import (
    BaselineCorrect,
    ComponentDropout,
    FixedLowpassDecimate,
    GaussianNoise,
    RobustInstanceNorm,
)
from .heads import POOLING
from .transformer import FlatTransformer
from .trunk import DilatedConvTrunk


class PerWordEncoder(nn.Module):
    """Front end, trunk, optional frame transformer, masked pool.

    ATTRS:
        n_channels (int): 73 whitened components
        n_times (int): padded window length in samples
        factor (int): decimation factor, samples per frame
        n_frames (int): frames the trunk sees for a full window
    """

    def __init__(
        self,
        n_channels,
        sample_rate=250.0,
        n_times=750,
        lowpass_hz=20.0,
        fir_taps=81,
        decimate_factor=5,
        baseline=(0.0, 0.5),
        with_baseline=True,
        with_instance_norm=True,
        component_dropout=0.0,
        gaussian_noise=0.0,
        spatial_width=270,
        initial_linear=512,
        trunk_hidden=160,
        trunk_depth=5,
        trunk_kernel=3,
        dilation_growth=2,
        dilation_period=None,
        glu_every=2,
        trunk_dropout=0.0,
        d_model=256,
        frame_depth=0,
        n_heads=8,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        transformer_norm="layer",
        pooling="attention",
    ):
        """Assemble the encoder.

        ARGS:
            n_channels (int): input width, the whitened components
            sample_rate (float): Hz of the incoming window
            n_times (int): padded window length; every window arrives this long
            lowpass_hz (float): anti-alias cutoff before decimation
            fir_taps (int): filter length, odd
            decimate_factor (int): 5 takes 250 Hz to 50 Hz
            baseline (tuple): (start, stop) seconds for BaselineCorrect
            with_baseline (bool): False skips the baseline
            with_instance_norm (bool): False skips the robust norm
            component_dropout (float): training-only component dropout
            gaussian_noise (float): training-only noise std
            spatial_width (int): the linear map off the components
            initial_linear (int): width handed to the trunk
            trunk_hidden (int): interior width of the trunk
            trunk_depth (int): convolution layers
            trunk_kernel (int): taps per layer, odd
            dilation_growth (int): ladder multiplier per layer
            dilation_period (int or None): None never resets: 1,2,4,8,16 at
                depth 5, reach 63 frames, which fits 150 where it did not fit 50
            glu_every (int): gate after every this many layers
            trunk_dropout (float): inside each trunk layer
            d_model (int): the word vector's width
            frame_depth (int): transformer blocks over FRAMES. 0 for Variant D
            n_heads (int): attention heads
            ffn_expansion (int): FFN hidden multiple
            dropout (float): attention and FFN dropout
            drop_path (float): maximum stochastic depth
            transformer_norm (str): "layer" or "bias"
            pooling (str): "attention" or "mean"
        """
        super().__init__()

        if pooling not in POOLING:
            raise KeyError(f"Unknown pooling {pooling!r}; known: {sorted(POOLING)}")

        #kept so forward() can check what it is handed, per the shape-check skill
        self.n_channels = n_channels
        self.n_times = n_times
        self.factor = decimate_factor

        self.instance_norm = RobustInstanceNorm() if with_instance_norm else None
        self.baseline = (
            BaselineCorrect(sample_rate=sample_rate, baseline=baseline) if with_baseline else None
        )
        self.decimate = FixedLowpassDecimate(
            n_channels,
            sample_rate=sample_rate,
            lowpass_hz=lowpass_hz,
            n_taps=fir_taps,
            factor=decimate_factor,
        )
        self.n_frames = self.decimate.output_length(n_times)

        #⚠️ after the robust norm, never before: upstream, noise and zeroed
        #components would enter the median and MAD the norm computes
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.component_dropout = ComponentDropout(component_dropout)

        #⚠️ a GELU between these two is load-bearing: two linear maps in a row
        #collapse to one
        self.spatial = nn.Sequential(nn.Conv1d(n_channels, spatial_width, 1), nn.GELU())
        self.initial_linear = nn.Sequential(nn.Conv1d(spatial_width, initial_linear, 1), nn.GELU())

        #the amplitude the robust norm removed, back as a per-window bias
        self.scale_projection = nn.Linear(n_channels, d_model) if with_instance_norm else None

        self.trunk = DilatedConvTrunk(
            n_input=initial_linear,
            n_output=d_model,
            hidden=trunk_hidden,
            depth=trunk_depth,
            kernel_size=trunk_kernel,
            dilation_growth=dilation_growth,
            dilation_period=dilation_period,
            glu_every=glu_every,
            dropout=trunk_dropout,
            n_frames=self.n_frames,
        )

        #depth 0 is not "absent": it adds the amplitude bias and a final norm,
        #exactly as trial 6's C2 did, before the pool
        self.frames = FlatTransformer(
            d_model=d_model,
            depth=frame_depth,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            drop_path=drop_path,
            norm=transformer_norm,
        )
        self.pool = POOLING[pooling](d_model) if pooling == "attention" else POOLING[pooling]()

    def frame_valid(self, lengths):
        """Which frames of each window lie inside its real samples.

        ARGS:
            lengths (torch.Tensor): [N] real samples per window

        RETURNS:
            [N, n_frames] bool
        """
        #the FIR is centred and strided, so frame f sits on input sample f*factor
        centres = torch.arange(self.n_frames, device=lengths.device) * self.factor
        return centres.unsqueeze(0) < lengths.unsqueeze(1)

    def forward(self, windows, lengths):
        """
        ARGS:
            windows (torch.Tensor): [N, n_channels, n_times], zero past each
                window's `lengths`
            lengths (torch.Tensor): [N] int64 real samples per window

        RETURNS:
            words : [N, d_model] - one vector per word
        """
        #a window with the wrong component count or padded length would still
        #run through every layer here, and score something it never trained on
        if windows.shape[1] != self.n_channels or windows.shape[2] != self.n_times:
            raise ValueError(
                f"Expected windows [N, {self.n_channels}, {self.n_times}], got "
                f"{tuple(windows.shape)}. Pad every window to the full length."
            )

        #the cache stores float16; every weight here is float32
        samples = torch.arange(self.n_times, device=windows.device)
        valid_samples = samples.unsqueeze(0) < lengths.unsqueeze(1)

        #padding is zeroed HERE rather than trusted, so whatever a caller left
        #past a window's end can never reach the output
        x = windows.float() * valid_samples.unsqueeze(1)

        static_bias = None
        if self.instance_norm is not None:
            x, log_scale = self.instance_norm(x, valid_samples)
            static_bias = self.scale_projection(log_scale)

        #subtracting a level turns the zero padding into -level, so re-zero it
        if self.baseline is not None:
            x = self.baseline(x) * valid_samples.unsqueeze(1)

        x = self.decimate(x)
        valid = self.frame_valid(lengths)

        #noise would land on the padding too; the mask takes it back off
        x = self.component_dropout(self.gaussian_noise(x)) * valid.unsqueeze(1)

        x = self.initial_linear(self.spatial(x))
        x = self.trunk(x)

        #channels first for the convolutions, channels last from here on
        encoded = self.frames(x.transpose(1, 2), static_bias=static_bias, valid=valid)
        return self.pool(encoded, valid)
