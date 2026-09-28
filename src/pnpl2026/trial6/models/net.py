"""The trial-6 network: trial 5's, with Variant C's switches.

    [batch, 73, T]            whitened components at 250 Hz, from the cache
      stream A, evoked (on by default)
        -> RobustInstanceNorm   per-window median/MAD, plus the scale as a bias
        -> BaselineCorrect      optional -- C3 turns it off
        -> FixedLowpassDecimate 20 Hz, 250 -> 50 Hz. C6: 100 Hz, no decimation
      stream B, band envelopes (off by default; C4 and C5 turn one band on)
        -> EnvelopeStream       |analytic band|, 20 Hz, / 5, log, fitted PCA
      concatenated on the component axis
        -> GaussianNoise, ComponentDropout, TimeMask   training only
        -> spatial              linear off the components, 270 wide
        -> initial_linear       1x1 to 512
        -> DilatedConvTrunk     unchanged from trial 5
        -> FlatTransformer      depth 8 as trial 5; C1 depth 2, C2 depth 0
        -> AttentionPool
        -> three heads          semantic embedding, 50-way logits, session

Every switch defaults to trial 5's behaviour, so `trial6.yaml` with no run
file on top IS trial 5's network -- which is what C0 needs to be.

⚠️ `forward` returns a dict rather than a tensor, because three heads is the
point and packing them into a tuple makes every call site depend on an ordering
nobody can see. The contract matches trial 5's exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .envelope import EnvelopeStream
from .frontend import (
    BaselineCorrect,
    ComponentDropout,
    FixedLowpassDecimate,
    GaussianNoise,
    RobustInstanceNorm,
    TimeMask,
)
from .heads import POOLING, ClosedSetHead, SemanticHead, SessionAdversary
from .transformer import FlatTransformer
from .trunk import DilatedConvTrunk

#t5-large's hidden width, which is what the reference model regresses onto
DEFAULT_EMBED_DIM = 1024


class TrialSixNet(nn.Module):
    """Two input streams, spatial map, conv trunk, flat transformer, three heads.

    ATTRS:
        evoked_stream (bool): whether stream A is on
        instance_norm (RobustInstanceNorm or None): stream A only
        baseline (BaselineCorrect or None): stream A only
        decimate (FixedLowpassDecimate or None): stream A's fixed filter
        envelope_streams (nn.ModuleList): stream B, one EnvelopeStream a band
        spatial (nn.Sequential): the linear map off the concatenated streams
        initial_linear (nn.Sequential): 1x1 up to the trunk's input width
        scale_projection (nn.Linear or None): stream A's discarded amplitude
        trunk (DilatedConvTrunk): the tokeniser
        transformer (FlatTransformer): 0 blocks is a valid depth here
        pool (nn.Module): collapses time to one vector
        semantic (SemanticHead): predicts the word's language-model embedding
        closed_set (ClosedSetHead): 50-way logits
        adversary (SessionAdversary or None): None when n_sessions is 0
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        n_sessions=0,
        embed_dim=DEFAULT_EMBED_DIM,
        sample_rate=250.0,
        n_times=250,
        lowpass_hz=20.0,
        fir_taps=81,
        decimate_factor=5,
        baseline=(0.0, 0.1),
        evoked_stream=True,
        envelope_bands=(),
        envelope_components=32,
        envelope_lowpass_hz=20.0,
        envelope_log=True,
        time_mask_spans=0,
        time_mask_width=0,
        component_dropout=0.0,
        gaussian_noise=0.0,
        spatial_width=270,
        initial_linear=512,
        trunk_hidden=160,
        trunk_depth=5,
        trunk_kernel=3,
        dilation_growth=2,
        dilation_period=4,
        glu_every=2,
        trunk_dropout=0.0,
        d_model=256,
        depth=8,
        n_heads=8,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        transformer_norm="layer",
        pooling="attention",
        head_dropout=0.1,
        with_instance_norm=True,
        with_baseline=True,
    ):
        """Assemble the network.

        ARGS:
            n_channels (int): input width -- 73 whitened components, not 306
            n_classes (int): 50, the scored vocabulary
            n_sessions (int): training sessions, for the adversary. 0 omits it
            embed_dim (int): target word-embedding width
            sample_rate (float): rate of the incoming window, in Hz
            n_times (int): samples in the cropped window. forward() refuses
                any other length, and the trunk's reach is checked against it
            lowpass_hz (float or None): stream A's anti-alias cutoff. None
                means no filter, which is only allowed without decimation
            fir_taps (int): filter length, odd, shared by both streams
            decimate_factor (int): 5 takes 250 Hz to 50 Hz. 1 keeps 250
            baseline (tuple): (start, stop) seconds for BaselineCorrect
            evoked_stream (bool): stream A on. False is C4/C5's envelope-only
            envelope_bands (list): [low_hz, high_hz] per stream-B band. Empty
                turns stream B off, which is the default
            envelope_components (int): PCA components kept per band
            envelope_lowpass_hz (float): smoothing of each envelope
            envelope_log (bool): log the envelope before its PCA
            time_mask_spans (int): spans of time to zero per window, training
            time_mask_width (int): maximum span length in INPUT samples
            component_dropout (float): probability of zeroing a component
            gaussian_noise (float): std of additive noise, training only
            spatial_width (int): width of the linear map off the components
            initial_linear (int): width handed to the trunk
            trunk_hidden (int): interior width of the conv trunk
            trunk_depth (int): convolution layers in the trunk
            trunk_kernel (int): taps per trunk layer, odd
            dilation_growth (int): ladder multiplier per trunk layer
            dilation_period (int or None): reset the ladder every this many
                layers. ⚠️ 4, not None: at 50 frames an unreset ladder reaches
                63 frames and the last layer convolves with padding
            glu_every (int): gate after every this many trunk layers
            trunk_dropout (float): inside each trunk layer
            d_model (int): transformer width
            depth (int): transformer blocks. 0 leaves only its final norm
            n_heads (int): attention heads
            ffn_expansion (int): hidden multiple in each FFN
            dropout (float): inside the transformer
            drop_path (float): maximum stochastic depth, ramped with depth
            transformer_norm (str): "layer" or "bias"
            pooling (str): "attention" or "mean", see heads.POOLING
            head_dropout (float): on the pooled vector, in every head
            with_instance_norm (bool): stream A's robust norm. False ablates it
            with_baseline (bool): stream A's baseline. False is C3
        """
        super().__init__()

        if pooling not in POOLING:
            raise KeyError(f"Unknown pooling {pooling!r}; known: {sorted(POOLING)}")

        #a network with no input stream would train its heads on biases alone
        if not evoked_stream and not envelope_bands:
            raise ValueError(
                "Both streams are off: set evoked_stream true or give at least "
                "one envelope band."
            )

        #⚠️ decimating without a lowpass first is aliasing, and nothing warns
        if lowpass_hz is None and decimate_factor > 1:
            raise ValueError(
                f"decimate_factor {decimate_factor} with no lowpass aliases "
                f"everything above {sample_rate / (2 * decimate_factor)} Hz."
            )

        #kept so forward() can check what it is handed, per the shape-check skill
        self.n_channels = n_channels
        self.n_times = n_times
        self.evoked_stream = evoked_stream

        # -- stream A, evoked -------------------------------------------------
        #the norm and baseline only exist to serve stream A; stream B reads the
        #raw window and whitens itself with its own fitted PCA
        self.instance_norm = (
            RobustInstanceNorm() if evoked_stream and with_instance_norm else None
        )
        self.baseline = (
            BaselineCorrect(sample_rate=sample_rate, baseline=baseline)
            if evoked_stream and with_baseline
            else None
        )

        #built whenever a cutoff is given -- C6 lowpasses at 100 Hz and keeps
        #every sample, which trial 5's "factor 1 means no filter" could not do
        self.decimate = (
            FixedLowpassDecimate(
                n_channels,
                sample_rate=sample_rate,
                lowpass_hz=lowpass_hz,
                n_taps=fir_taps,
                factor=decimate_factor,
            )
            if lowpass_hz is not None
            else None
        )

        #how many frames the trunk will actually see, computed rather than
        #assumed -- the FIR's padding arithmetic decides this, not a division
        self.n_frames = (
            self.decimate.output_length(n_times) if self.decimate is not None else n_times
        )

        # -- stream B, band envelopes ----------------------------------------
        self.envelope_streams = nn.ModuleList(
            [
                EnvelopeStream(
                    n_channels,
                    sample_rate=sample_rate,
                    n_times=n_times,
                    band=band,
                    n_components=envelope_components,
                    lowpass_hz=envelope_lowpass_hz,
                    n_taps=fir_taps,
                    factor=decimate_factor,
                    with_log=envelope_log,
                )
                for band in envelope_bands
            ]
        )

        #⚠️ the two streams are concatenated frame by frame, so their frame
        #counts must agree exactly; the shared taps and factor guarantee it,
        #and this catches the day someone gives one stream its own
        for stream in self.envelope_streams:
            frames = stream.decimate.output_length(n_times)
            if frames != self.n_frames:
                raise ValueError(
                    f"Envelope band {stream.name} gives {frames} frames but "
                    f"stream A gives {self.n_frames}."
                )

        n_inputs = (n_channels if evoked_stream else 0) + envelope_components * len(
            self.envelope_streams
        )

        # -- augmentation, after every normalisation -------------------------
        #⚠️ THE AUGMENTATION LIVES HERE, NOT IN THE LightningModule. Upstream of
        #the instance norm, masked or noisy samples enter the median and MAD it
        #computes, and a training window stops being comparable to a holdout
        #one. All three modules are identities at eval
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.component_dropout = ComponentDropout(component_dropout)

        #width arrives in INPUT samples so the config reads in one unit, and is
        #converted here because the mask runs after decimation
        mask_frames = (
            max(1, int(round(time_mask_width / max(decimate_factor, 1))))
            if time_mask_width > 0
            else 0
        )
        self.time_mask = TimeMask(time_mask_spans, mask_frames)

        # -- the learned network, trial 5's ----------------------------------
        #⚠️ a GELU between this and initial_linear is load-bearing: two linear
        #maps in a row collapse to one
        self.spatial = nn.Sequential(
            nn.Conv1d(n_inputs, spatial_width, 1),
            nn.GELU(),
        )
        self.initial_linear = nn.Sequential(
            nn.Conv1d(spatial_width, initial_linear, 1),
            nn.GELU(),
        )

        #the amplitude stream A's instance norm removed comes back as a
        #per-window bias. ⚠️ Absent without stream A: C4 and C5 ask what the
        #envelopes carry alone, and this is broadband amplitude
        self.scale_projection = (
            nn.Linear(n_channels, d_model) if self.instance_norm is not None else None
        )

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

        self.transformer = FlatTransformer(
            d_model=d_model,
            depth=depth,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            drop_path=drop_path,
            norm=transformer_norm,
        )

        self.pool = (
            POOLING[pooling](d_model) if pooling == "attention" else POOLING[pooling]()
        )

        self.semantic = SemanticHead(d_model, embed_dim, head_dropout)
        self.closed_set = ClosedSetHead(d_model, n_classes, head_dropout)

        #no adversary when the caller did not say how many sessions exist; a
        #head with the wrong class count trains to nonsense without raising
        self.adversary = (
            SessionAdversary(d_model, n_sessions, head_dropout)
            if n_sessions > 0
            else None
        )

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times] channels FIRST

        RETURNS:
            dict with keys:
                "pooled" : [batch, d_model] - what every head reads
                "logits" : [batch, n_classes] - closed-set scores, not softmaxed
                "embedding" : [batch, embed_dim] - unnormalised semantic vector
        """
        #a window served with the wrong component count is the silent failure
        #this trial is exposed to: 306 raw sensors instead of 73 components
        n_channels, n_times = windows.shape[1], windows.shape[2]
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}. "
                f"The model is fed whitened components, not raw sensors -- "
                f"check n_channels against the cache's n_components."
            )

        #⚠️ and the wrong LENGTH is the other one: every layer here accepts any
        #length, so a 1 s holdout window fed to a 0.5 s model would run and
        #score a window it never trained on
        if n_times != self.n_times:
            raise ValueError(
                f"Expected {self.n_times} samples, got {n_times}. Crop the "
                f"window to the configured tmin/tmax before the model."
            )

        #the cache stores float16; every weight here is float32
        raw = windows.float()

        streams = []
        static_bias = None
        if self.evoked_stream:
            evoked = raw
            if self.instance_norm is not None:
                evoked, log_scale = self.instance_norm(evoked)
                static_bias = self.scale_projection(log_scale)

            #⚠️ after the instance norm, not before: the norm's median is
            #robust to the drift this removes
            if self.baseline is not None:
                evoked = self.baseline(evoked)

            if self.decimate is not None:
                evoked = self.decimate(evoked)
            streams.append(evoked)

        #stream B reads the RAW window: through stream A's normaliser, every
        #envelope would be divided by broadband amplitude, and C4 and C5 would
        #no longer be asking about their band alone
        for stream in self.envelope_streams:
            streams.append(stream(raw))

        #dim 1 is the component axis; both streams are [batch, *, n_frames]
        x = torch.cat(streams, dim=1)

        #after everything fixed, before anything learned. All no-ops outside
        #training, so `predict.py` needs to know nothing about them
        x = self.time_mask(self.component_dropout(self.gaussian_noise(x)))

        x = self.initial_linear(self.spatial(x))
        x = self.trunk(x)

        #the one transpose in the whole network: channels first for the fixed
        #front end and the convolutions, channels last for attention
        encoded = self.transformer(x.transpose(1, 2), static_bias=static_bias)
        pooled = self.pool(encoded)

        return {
            "pooled": pooled,
            #logits, because cross_entropy applies its own log-softmax
            "logits": self.closed_set(pooled),
            #unnormalised: the contrastive loss normalises both sides itself
            "embedding": self.semantic(pooled),
        }

    def session_logits(self, pooled, strength):
        """Run the adversary, or return None when there is not one.

        Kept off `forward` because it needs the reversal strength, which is a
        function of the training step and not of the input.

        ARGS:
            pooled (torch.Tensor): [batch, d_model] from `forward`
            strength (float): gradient-reversal coefficient

        RETURNS:
            logits : [batch, n_sessions], or None
        """
        if self.adversary is None:
            return None
        return self.adversary(pooled, strength)
