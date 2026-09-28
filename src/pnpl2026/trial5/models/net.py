"""The trial-5 network: fixed front end, conv trunk, flat transformer, 3 heads.

    [batch, 73, 250]          whitened components, 1 s at 250 Hz, from the cache
      -> RobustInstanceNorm   per-window median/MAD, plus the scale as a bias
      -> BaselineCorrect      subtract the early-window level
      -> FixedLowpassDecimate 20 Hz lowpass, 250 -> 50 Hz, so 250 -> 50 frames
      -> spatial              VARIANT B: linear from the 73 PCA components
      -> initial_linear       1x1 to 512
      -> DilatedConvTrunk     5 layers, the tokeniser. ~0.8M
      -> FlatTransformer      depth 8, d=256, rotary. ~6.3M -- 79% of the model
      -> AttentionPool        collapse time once, at the end
      -> three heads          semantic embedding, 50-way logits, session logits

**The one structural bet.** Trial 2 scored 34.70% with a 1.9M model whose
sequence stage was ~0.6M. The reference model scores 73.23% with 203.7M, of
which 201.3M -- 98.8% -- is its transformer. This network puts 79% of its
parameters in the sequence stage for the first time. Everything else here is
either already measured or deliberately conservative.

⚠️ `forward` returns a dict rather than a tensor, because three heads is the
point and packing them into a tuple makes every call site depend on an ordering
nobody can see. The contract matches trial 4's exactly, which is what lets
`lit.py` stay architecture-independent.

VARIANT A -- a learned `ChannelMerger` over raw 306 sensors and their geometry
-- is NOT built here. It needs the external drive for the continuous recordings
and the sensor positions, and it is the only part of the design gated on that.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .frontend import (
    BaselineCorrect,
    ComponentDropout,
    FixedLowpassDecimate,
    RobustInstanceNorm,
    TimeMask,
)
from .heads import POOLING, ClosedSetHead, SemanticHead, SessionAdversary
from .transformer import FlatTransformer
from .trunk import DilatedConvTrunk

#t5-large's hidden width, which is what the reference model regresses onto
DEFAULT_EMBED_DIM = 1024


class TrialFiveNet(nn.Module):
    """Front end, spatial map, conv trunk, flat transformer, three heads.

    ATTRS:
        instance_norm (RobustInstanceNorm or None): per-window standardisation
        baseline (BaselineCorrect or None): early-window level removal
        decimate (FixedLowpassDecimate or None): fixed anti-alias and downsample
        spatial (nn.Sequential): VARIANT B, the linear map off the components
        initial_linear (nn.Sequential): 1x1 up to the trunk's input width
        scale_projection (nn.Linear): the discarded amplitude, as a static bias
        trunk (DilatedConvTrunk): the tokeniser
        transformer (FlatTransformer): where the capacity is
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
        time_mask_spans=0,
        time_mask_width=0,
        component_dropout=0.0,
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
            n_times (int): samples in the cropped window. Used ONLY to check
                the trunk's reach against the frame count -- see trunk.py
            lowpass_hz (float): anti-alias cutoff before decimation
            fir_taps (int): filter length, odd
            decimate_factor (int): 5 takes 250 Hz to 50 Hz. 1 disables it
            baseline (tuple): (start, stop) seconds for BaselineCorrect
            time_mask_spans (int): spans of time to zero per window, in
                training only. 0 disables it
            time_mask_width (int): maximum span length in INPUT samples,
                converted to frames here so the config keeps one unit
            component_dropout (float): probability of zeroing a whole
                component, in training only
            spatial_width (int): VARIANT B's output width. 270 matches the
                reference's merger, so variant A can replace this stage alone
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
            depth (int): transformer blocks
            n_heads (int): attention heads
            ffn_expansion (int): hidden multiple in each FFN
            dropout (float): inside the transformer
            drop_path (float): maximum stochastic depth, ramped with depth
            transformer_norm (str): "layer" or "bias"
            pooling (str): "attention" or "mean", see heads.POOLING
            head_dropout (float): on the pooled vector, in every head
            with_instance_norm (bool): False is the ablation
            with_baseline (bool): False is the ablation. ⚠️ worth running --
                the instance norm already removes the per-channel median, so
                this may be buying nothing on top of it
        """
        super().__init__()

        if pooling not in POOLING:
            raise KeyError(f"Unknown pooling {pooling!r}; known: {sorted(POOLING)}")

        #kept so forward() can check what it is handed, per the shape-check skill
        self.n_channels = n_channels
        self.n_times = n_times

        self.instance_norm = RobustInstanceNorm() if with_instance_norm else None
        self.baseline = (
            BaselineCorrect(sample_rate=sample_rate, baseline=baseline)
            if with_baseline
            else None
        )

        #factor 1 means "do not decimate", and building a filter for it would
        #only add a pointless lowpass
        self.decimate = (
            FixedLowpassDecimate(
                n_channels,
                sample_rate=sample_rate,
                lowpass_hz=lowpass_hz,
                n_taps=fir_taps,
                factor=decimate_factor,
            )
            if decimate_factor > 1
            else None
        )

        #how many frames the trunk will actually see, computed rather than
        #assumed -- the FIR's padding arithmetic decides this, not a division
        self.n_frames = (
            self.decimate.output_length(n_times) if self.decimate is not None else n_times
        )

        #⚠️ THE AUGMENTATION LIVES HERE, NOT IN THE LightningModule, and the
        #placement is load-bearing rather than tidy. Upstream of the instance
        #norm -- where it used to sit -- masked samples enter the median and MAD
        #that the norm computes, and masking 40 of 250 samples inflates the
        #normalised amplitude by 26%. The norm exists to make a training window
        #and a holdout window comparable; augmenting before it breaks exactly
        #that, on training windows only. Both modules are identities at eval
        self.component_dropout = ComponentDropout(component_dropout)

        #width arrives in INPUT samples so the config reads in one unit, and is
        #converted here because the mask runs after decimation
        mask_frames = (
            max(1, int(round(time_mask_width / max(decimate_factor, 1))))
            if time_mask_width > 0
            else 0
        )
        self.time_mask = TimeMask(time_mask_spans, mask_frames)

        #⚠️ VARIANT B. A GELU between this and initial_linear is load-bearing:
        #two linear maps in a row collapse to one, so without it these are 73
        #-> 512 wearing two names and the extra parameters buy nothing
        self.spatial = nn.Sequential(
            nn.Conv1d(n_channels, spatial_width, 1),
            nn.GELU(),
        )
        self.initial_linear = nn.Sequential(
            nn.Conv1d(spatial_width, initial_linear, 1),
            nn.GELU(),
        )

        #the amplitude the instance norm removed comes back as a per-window
        #bias on the transformer input, so scale is available but no longer
        #sets the network's input range
        self.scale_projection = (
            nn.Linear(n_channels, d_model) if with_instance_norm else None
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
        #still convolves happily if the front end was built for 306
        n_channels = windows.shape[1]
        if n_channels != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {n_channels}. "
                f"The model is fed whitened components, not raw sensors -- "
                f"check n_channels against the cache's n_components."
            )

        #the cache stores float16; every weight here is float32
        x = windows.float()

        static_bias = None
        if self.instance_norm is not None:
            x, log_scale = self.instance_norm(x)
            static_bias = self.scale_projection(log_scale)

        #⚠️ after the instance norm, not before: the norm's median is robust to
        #the drift this removes, and doing it the other way round means the
        #MAD is computed on a signal that has already been re-levelled
        if self.baseline is not None:
            x = self.baseline(x)

        #the lowpass runs while the layout is still the channels-first one
        #convolutions want
        if self.decimate is not None:
            x = self.decimate(x)

        #after the fixed front end, before anything learned. Both are no-ops
        #outside training, so `predict.py` needs to know nothing about them
        x = self.time_mask(self.component_dropout(x))

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
