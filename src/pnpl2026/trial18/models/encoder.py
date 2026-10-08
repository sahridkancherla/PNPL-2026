"""The per-word encoder: brainmagick's `SimpleConv` as clip_conv configures it.

    window [N, 306, 150]  robust-scaled per recording at build time
      -> clamp +/-5
      -> trial 15, `input_norm: instance` only: each channel standardised by
         this window's own mean and spread over its real samples, clamp again
      -> subtract each channel's mean over the first 0.5 s
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

⚠️ Windows are zero-padded past `lengths` whenever the MEG stops early -- the
holdout's fragments, trial 13's shaped training items. The padding is
re-zeroed after the baseline and after every layer, so what leaks into the
real samples through a dilated kernel is always zeros, as at the edge of any
window.

Trial 13's one change here: **masked BatchNorm**. Trial 10's BatchNorm took
its batch statistics over padding too, which never mattered because trial 10
never trained on a padded window. Trial 13 does, so the statistics are taken
over real samples only. Same parameters and buffers as `nn.BatchNorm1d`, so
trial 10/12 checkpoints load unchanged, and with no padding the two are
identical.

Trial 15's one change: **per-window input normalisation** (`input_norm`,
default "none"). Training windows are scaled per recording, from that
recording's own continuous MEG; holdout windows come from recordings the
organisers have not revealed, scaled by what `cli.predict` can fit on the
holdout epochs. The 2025 LibriBrain phoneme winner found standardising each
window by its own statistics the single largest fix for exactly that shift.
It has no parameters, so it adds no state-dict keys: earlier checkpoints load
unchanged and run with "none".

Trial 18's one change: `WordEncoder.features` and
`ConvSequence.forward_with_taps` also hand back chosen conv layers' gated
outputs, for Variant M's refiner. `forward` computes exactly what it did.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .merger import ChannelMerger, FixedPCA, LinearSpatial, SubjectLayer

#the three spatial front ends a config can name
SPATIAL_KINDS = ("merger", "linear", "pca")

#how each window is normalised on entry: as trained so far, or by its own statistics
INPUT_NORMS = ("none", "instance")


class MaskedBatchNorm1d(nn.BatchNorm1d):
    """BatchNorm1d whose batch statistics ignore padded samples.

    In eval mode it is exactly `nn.BatchNorm1d`: the running statistics, read
    per channel, do not depend on padding. In training mode, given `valid`,
    the mean and variance come from real samples only, and the running
    statistics are updated from those -- unbiased variance, as PyTorch does.
    """

    def forward(self, x, valid=None):
        """
        ARGS:
            x (torch.Tensor): [batch, channels, n_times]
            valid (torch.Tensor or None): [batch, 1, n_times], 1 on real
                samples and 0 on padding; None means every sample is real

        RETURNS:
            same shape as x
        """
        #nothing padded, or running statistics in use: plain BatchNorm
        if valid is None or not self.training:
            return super().forward(x)

        #float32 for the statistics, whatever the autocast; dims 0 and 2 are
        #batch and time, the axes BatchNorm averages over
        values = x.float()
        weights = valid.float()
        count = weights.sum(dim=(0, 2)).clamp_min(2.0)
        mean = (values * weights).sum(dim=(0, 2)) / count
        variance = (((values - mean.view(1, -1, 1)) ** 2) * weights).sum(dim=(0, 2)) / count

        #running statistics as nn.BatchNorm1d keeps them: momentum None is a
        #cumulative average, which is what re-estimating them on new data wants
        if self.track_running_stats:
            with torch.no_grad():
                self.num_batches_tracked += 1
                factor = 1.0 / float(self.num_batches_tracked) if self.momentum is None else self.momentum
                unbiased = variance * count / (count - 1.0)
                self.running_mean.mul_(1 - factor).add_(mean.to(self.running_mean.dtype), alpha=factor)
                self.running_var.mul_(1 - factor).add_(unbiased.to(self.running_var.dtype), alpha=factor)

        #normalise with the batch statistics, then the learned scale and shift
        out = (values - mean.view(1, -1, 1)) / torch.sqrt(variance.view(1, -1, 1) + self.eps)
        if self.affine:
            out = out * self.weight.view(1, -1, 1) + self.bias.view(1, -1, 1)
        return out.to(x.dtype)


class ConvSequence(nn.Module):
    """brainmagick's `ConvSequence`, restricted to what clip_conv sets.

    Layer k is conv -> masked BatchNorm -> GELU, plus its input when the
    shapes match; after every `glu`-th layer a gated 1D conv of width
    1 + 2 * glu_context. The dilation doubles each layer and resets to 1
    every `dilation_period` layers.

    Each layer stays an `nn.Sequential` so the state-dict keys are trial
    10's; `forward` walks it by hand to hand the mask to the BatchNorm.

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
                modules.append(MaskedBatchNorm1d(width_out))
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
        return self.forward_with_taps(x, valid)[0]

    def forward_with_taps(self, x, valid=None, tap_layers=()):
        """The same pass, also handing back chosen layers' outputs (Variant M).

        ARGS:
            x (torch.Tensor): [batch, channels[0], n_times]
            valid (torch.Tensor or None): [batch, 1, n_times], 1 on real
                samples and 0 on padding
            tap_layers (tuple): layer indices whose output is kept: after the
                gated conv, padding re-zeroed

        RETURNS:
            x : [batch, channels[-1], n_times]
            tapped : list of [batch, channels[k + 1], n_times], one per tap
                layer, in layer order
        """
        #the mask must describe these very samples, or BatchNorm averages the
        #wrong ones without a word
        if valid is not None:
            assert valid.shape == (x.shape[0], 1, x.shape[2]), f"mask {tuple(valid.shape)} for input {tuple(x.shape)}"

        tapped = []
        for layer_index, (layer, glu) in enumerate(zip(self.layers, self.glus)):
            #conv, BatchNorm (given the mask), activation, in order
            out = x
            for module in layer:
                out = module(out, valid) if isinstance(module, MaskedBatchNorm1d) else module(out)

            #the residual only where the widths match, as brainmagick's
            if self.skip and out.shape == x.shape:
                out = out + x
            if glu is not None:
                out = glu(out)

            #padding back to zero, so the next kernel reads zeros there
            x = out if valid is None else out * valid

            #Variant M reads a few depths' gated outputs; nothing else changes
            if layer_index in tap_layers:
                tapped.append(x)
        return x, tapped


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
        input_norm (str): "none", or "instance" -- per window and channel
        input_norm_eps (float): added to the variance before the square root
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
        input_norm="none",
        input_norm_eps=1e-5,
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
            input_norm (str): "none" (trials 10-13), or "instance": each
                window standardised per channel by its own statistics
            input_norm_eps (float): variance floor for "instance"
        """
        super().__init__()
        if spatial not in SPATIAL_KINDS:
            raise ValueError(f"spatial must be one of {SPATIAL_KINDS}, got {spatial!r}")
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {INPUT_NORMS}, got {input_norm!r}")

        self.n_channels = n_channels
        self.clamp = float(clamp)
        self.input_norm = input_norm
        self.input_norm_eps = float(input_norm_eps)
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

    def standardise_window(self, x, mask):
        """Each window's channels to zero mean and unit spread, real samples only.

        The statistics come from the samples before `lengths`, so a truncated
        holdout window is standardised by its own MEG, never by its padding.
        Clamped again afterwards, now in units of the window's own spread.

        ARGS:
            x (torch.Tensor): [N, n_channels, n_times] float32, already clamped
            mask (torch.Tensor): [N, 1, n_times], 1 on real samples, 0 on padding

        RETURNS:
            [N, n_channels, n_times] float32, zero on padding
        """
        #dim 2 is time; padded samples carry zero weight in both statistics
        count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean = (x * mask).sum(dim=2, keepdim=True) / count
        variance = (((x - mean) ** 2) * mask).sum(dim=2, keepdim=True) / count

        #a flat channel has no spread to divide by; the floor keeps it at zero
        standardised = (x - mean) / torch.sqrt(variance + self.input_norm_eps)
        return standardised.clamp(-self.clamp, self.clamp) * mask

    def forward(self, windows, lengths):
        """
        ARGS:
            windows (torch.Tensor): [N, n_channels, n_times], zero-padded
            lengths (torch.Tensor): [N] real samples per window

        RETURNS:
            [N, d_model] - one vector per word
        """
        return self.features(windows, lengths)[0]

    def features(self, windows, lengths, tap_layers=()):
        """One vector per word, and chosen conv layers' outputs (Variant M).

        ARGS:
            windows (torch.Tensor): [N, n_channels, n_times], zero-padded
            lengths (torch.Tensor): [N] real samples per window
            tap_layers (tuple): ConvSequence layers whose outputs are kept

        RETURNS:
            vector : [N, d_model] - one vector per word
            tapped : list of [N, hidden, n_times], zero on padding
        """
        #the sensor axis is the one a wrong transpose would put time on
        n_words, n_channels, n_times = windows.shape
        assert n_channels == self.n_channels, f"expected {self.n_channels} channels, got {n_channels}"
        assert lengths.shape == (n_words,), f"lengths {tuple(lengths.shape)} for {n_words} windows"

        #float32 for the clamp and the baseline, whatever the cache stored
        x = windows.float().clamp(-self.clamp, self.clamp)
        valid = self.frame_valid(lengths, x.shape[2])
        mask = valid.unsqueeze(1).to(x.dtype)

        #trial 15: each window on its own scale, whatever its recording's was
        if self.input_norm == "instance":
            x = self.standardise_window(x, mask)

        #per channel, the mean of the window's first 0.5 s; dim 2 is time
        if self.baseline_samples > 0:
            baseline = x[:, :, : self.baseline_samples].mean(dim=2, keepdim=True)
            x = (x - baseline) * mask

        #sensors -> virtual channels -> subject space, padding kept at zero
        x = self.spatial(x)
        x = self.subject(self.initial_linear(x)) * mask
        x, tapped = self.convs.forward_with_taps(x, mask, tap_layers)

        #[N, d_model, T] -> [N, T, d_model] for the pool
        x = self.final(x).transpose(1, 2)
        return self.pool(x, valid), tapped
