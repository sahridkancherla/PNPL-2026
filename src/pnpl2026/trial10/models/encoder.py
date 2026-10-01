"""The per-word encoder: brainmagick's `SimpleConv` as clip_conv configures it.

    window [N, 306, 150]  robust-scaled per recording at build time
      -> clamp +/-5, then subtract each channel's mean over the first 0.5 s
      -> spatial: ChannelMerger over sensor positions        [N, 270, 150]
      -> initial_linear, a 1x1 conv 270 -> 270, no activation
      -> SubjectLayer, one subject, identity at start
      -> ConvSequence: 10 layers, kernel 3, dilations
         1,2,4,8,16,1,2,4,8,16, BatchNorm, GELU, residual skip,
         a GLU after every second layer                       [N, hidden, 150]
      -> final: 1x1 to 2*hidden, GELU, 1x1 to d_model         [N, d_model, 150]
      -> single-head attention pool over time                 [N, d_model]

Hidden width 160, not brainmagick's 320: the competition paper's 203.7M at
98.8% transformer leaves ~2.4M for everything else, which 160 matches.

⚠️ Windows are zero-padded past `lengths` under the holdout protocol. The
padding is re-zeroed after the baseline and after every layer, so what leaks
into the real samples through a dilated kernel is always zeros, as at the
edge of any window. The reference protocol never pads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .merger import ChannelMerger, FixedPCA, LinearSpatial, SubjectLayer

#the three spatial front ends a config can name
SPATIAL_KINDS = ("merger", "linear", "pca")


class ConvSequence(nn.Module):
    """brainmagick's `ConvSequence`, restricted to what clip_conv sets.

    Layer k is conv -> BatchNorm -> GELU, plus its input when the shapes
    match; after every `glu`-th layer a gated 1D conv of width
    1 + 2 * glu_context. The dilation doubles each layer and resets to 1
    every `dilation_period` layers.

    ATTRS:
        layers (nn.ModuleList): conv, norm, activation per layer
        glus (nn.ModuleList): a gated conv, or None, per layer
        skip (bool): residual connection where the shapes allow it
        dilations (list): the dilation of each layer
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation_growth=2,
        dilation_period=5,
        batch_norm=True,
        skip=True,
        glu=2,
        glu_context=1,
    ):
        """
        ARGS:
            channels (list): widths, input first; len - 1 layers
            kernel_size (int): odd, so padding keeps the length
            dilation_growth (int): multiplier per layer
            dilation_period (int or None): reset the dilation every this many
            batch_norm (bool): BatchNorm1d after each conv
            skip (bool): add the input back where the widths match
            glu (int): a gated conv after every glu-th layer; 0 for none
            glu_context (int): its half-width
        """
        super().__init__()

        #even kernels with dilation do not keep the length centred
        if kernel_size % 2 != 1:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.skip = skip
        self.layers = nn.ModuleList()
        self.glus = nn.ModuleList()
        self.dilations = []

        dilation = 1
        for layer, (width_in, width_out) in enumerate(zip(channels[:-1], channels[1:])):
            #brainmagick resets BEFORE building layer k when k % period == 0
            if dilation_period and layer % dilation_period == 0:
                dilation = 1
            self.dilations.append(dilation)

            #"same" padding, so every layer keeps the window's length
            modules = [
                nn.Conv1d(
                    width_in,
                    width_out,
                    kernel_size,
                    padding=kernel_size // 2 * dilation,
                    dilation=dilation,
                )
            ]
            if batch_norm:
                modules.append(nn.BatchNorm1d(width_out))
            modules.append(nn.GELU())
            self.layers.append(nn.Sequential(*modules))
            dilation *= dilation_growth

            #a gated conv after layers glu-1, 2*glu-1, ...: doubles, then halves
            if glu and (layer + 1) % glu == 0:
                self.glus.append(
                    nn.Sequential(
                        nn.Conv1d(width_out, 2 * width_out, 1 + 2 * glu_context, padding=glu_context),
                        nn.GLU(dim=1),
                    )
                )
            else:
                self.glus.append(None)

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, channels[0], n_times]
            valid (torch.Tensor or None): [batch, 1, n_times], 1 on real
                samples and 0 on padding

        RETURNS:
            [batch, channels[-1], n_times]
        """
        for layer, glu in zip(self.layers, self.glus):
            out = layer(x)

            #the residual only where the widths match, as brainmagick's
            if self.skip and out.shape == x.shape:
                out = out + x
            if glu is not None:
                out = glu(out)

            #padding back to zero, so the next kernel reads zeros there
            x = out if valid is None else out * valid
        return x


class AttentionPool(nn.Module):
    """Collapse time by a learned weighting over the real samples.

    Single-head: one score per sample, softmaxed over the window.

    ATTRS:
        score (nn.Linear): D -> 1, no bias -- a constant cancels in the softmax
    """

    def __init__(self, n_features):
        """
        ARGS:
            n_features (int): D
        """
        super().__init__()
        self.score = nn.Linear(n_features, 1, bias=False)

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, n_times, n_features]
            valid (torch.Tensor or None): [batch, n_times] bool

        RETURNS:
            [batch, n_features]
        """
        scores = self.score(x)

        #-inf is zero weight after the softmax; every window has real samples
        if valid is not None:
            scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

        #dim=1 is TIME: the weights sum to one across the window
        weights = scores.softmax(dim=1)
        return (x * weights).sum(dim=1)


class WordEncoder(nn.Module):
    """One word's window to one vector: the reference's convolutional encoder.

    ATTRS:
        clamp (float): the reference clips scaled MEG to +/- this
        baseline_samples (int): how much of the window's start is its baseline
        spatial (nn.Module): ChannelMerger, LinearSpatial or FixedPCA
        initial_linear (nn.Module): 1x1 conv, or Identity
        subject (nn.Module): SubjectLayer, or Identity
        convs (ConvSequence)
        final (nn.Sequential): 1x1 -> GELU -> 1x1 to d_model
        pool (AttentionPool)
    """

    def __init__(
        self,
        n_channels,
        d_model,
        sample_rate,
        clamp=5.0,
        baseline_seconds=0.5,
        spatial="merger",
        positions=None,
        pca=None,
        merger_channels=270,
        merger_pos_dim=2048,
        merger_dropout=0.2,
        initial_linear=270,
        subject_layer=True,
        hidden=160,
        depth=10,
        kernel_size=3,
        dilation_growth=2,
        dilation_period=5,
        batch_norm=True,
        skip=True,
        glu=2,
        glu_context=1,
    ):
        """
        ARGS:
            n_channels (int): 306 sensors
            d_model (int): output width, the sentence transformer's
            sample_rate (float): 50 Hz, for the baseline length
            clamp (float): +/- limit after the per-recording scaler
            baseline_seconds (float): 0 disables the baseline
            spatial (str): "merger", "linear" or "pca"
            positions (array or None): [n_channels, 2], for the merger
            pca (dict or None): mean, components, scale, for "pca"
            merger_channels (int): 270
            merger_pos_dim (int): 2048
            merger_dropout (float): spatial dropout radius, 0.2
            initial_linear (int): 1x1 width after the spatial layer; 0 skips it
            subject_layer (bool): one subject's linear map
            hidden (int): ConvSequence width
            depth (int): ConvSequence layers
            kernel_size (int): 3
            dilation_growth (int): 2
            dilation_period (int): 5, so 1,2,4,8,16 twice
            batch_norm (bool)
            skip (bool)
            glu (int): every second layer
            glu_context (int): 1
        """
        super().__init__()
        if spatial not in SPATIAL_KINDS:
            raise ValueError(f"spatial must be one of {SPATIAL_KINDS}, got {spatial!r}")

        self.n_channels = n_channels
        self.clamp = float(clamp)
        self.baseline_samples = int(round(baseline_seconds * sample_rate))

        #the spatial layer sets the width everything after it sees
        if spatial == "merger":
            if positions is None:
                raise ValueError(
                    "spatial: merger needs sensor positions, and the cache has none. "
                    "Rebuild it from h5 files with channel names, or set spatial: linear."
                )
            self.spatial = ChannelMerger(positions, merger_channels, merger_pos_dim, merger_dropout)
            width = merger_channels
        elif spatial == "linear":
            self.spatial = LinearSpatial(n_channels, merger_channels)
            width = merger_channels
        else:
            if pca is None:
                raise ValueError("spatial: pca needs the cache's pca.npz.")
            self.spatial = FixedPCA(pca["mean"], pca["components"], pca["scale"])
            width = self.spatial.out_channels

        #brainmagick's initial_linear: a 1x1 conv with no activation after it
        self.initial_linear = nn.Identity()
        if initial_linear:
            self.initial_linear = nn.Conv1d(width, initial_linear, 1)
            width = initial_linear

        self.subject = SubjectLayer(width) if subject_layer else nn.Identity()

        #[width] then `depth` layers of `hidden`: the first layer has no skip
        self.convs = ConvSequence(
            [width] + [hidden] * depth,
            kernel_size=kernel_size,
            dilation_growth=dilation_growth,
            dilation_period=dilation_period,
            batch_norm=batch_norm,
            skip=skip,
            glu=glu,
            glu_context=glu_context,
        )

        #brainmagick's complex_out: widen, activate, project to d_model
        self.final = nn.Sequential(
            nn.Conv1d(hidden, 2 * hidden, 1),
            nn.GELU(),
            nn.Conv1d(2 * hidden, d_model, 1),
        )
        self.pool = AttentionPool(d_model)

    def frame_valid(self, lengths, n_times):
        """Which samples of each padded window are real.

        ARGS:
            lengths (torch.Tensor): [N] real samples per window
            n_times (int): padded length

        RETURNS:
            [N, n_times] bool
        """
        #dim 1 is time: sample t is real when t < length
        steps = torch.arange(n_times, device=lengths.device)
        return steps.unsqueeze(0) < lengths.unsqueeze(1)

    def forward(self, windows, lengths):
        """
        ARGS:
            windows (torch.Tensor): [N, n_channels, n_times], zero-padded
            lengths (torch.Tensor): [N] real samples per window

        RETURNS:
            [N, d_model] - one vector per word
        """
        #the sensor axis is the one a wrong transpose would put time on
        n_words, n_channels, n_times = windows.shape
        assert n_channels == self.n_channels, f"expected {self.n_channels} channels, got {n_channels}"
        assert lengths.shape == (n_words,), f"lengths {tuple(lengths.shape)} for {n_words} windows"

        #float32 for the clamp and the baseline, whatever the cache stored
        x = windows.float().clamp(-self.clamp, self.clamp)
        valid = self.frame_valid(lengths, x.shape[2])
        mask = valid.unsqueeze(1).to(x.dtype)

        #per channel, the mean of the window's first 0.5 s; dim 2 is time
        if self.baseline_samples > 0:
            baseline = x[:, :, : self.baseline_samples].mean(dim=2, keepdim=True)
            x = (x - baseline) * mask

        #sensors -> virtual channels -> subject space, padding kept at zero
        x = self.spatial(x)
        x = self.subject(self.initial_linear(x)) * mask
        x = self.convs(x, mask)

        #[N, d_model, T] -> [N, T, d_model] for the pool
        x = self.final(x).transpose(1, 2)
        return self.pool(x, valid)
