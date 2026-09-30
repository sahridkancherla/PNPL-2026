"""The trial-4 network: front end, multi-rate Conformer encoder, three heads.

    [batch, 73, 250]          whitened components, 1 s at 250 Hz, from the cache
      -> RobustInstanceNorm   per-window median/MAD, plus the scale as a bias
      -> FixedLowpassDecimate 45 Hz lowpass, 250 -> 125 Hz
      -> MultiRateEncoder     pyramid of Conformer stacks, 125 / 63 / 32 frames
      -> multi-scale pooling  collapse time on EVERY stack, then concatenate
      -> three heads          semantic embedding, 50-way logits, session logits

`forward` returns a dict rather than a tensor, because three heads is the whole
point of the architecture and packing them into a tuple makes the call sites
depend on an ordering nobody can see.
"""

from __future__ import annotations

import torch.nn as nn

import torch

from .encoder import MultiRateEncoder
from .frontend import FixedLowpassDecimate, RobustInstanceNorm
from .heads import POOLING, ClosedSetHead, SemanticHead, SessionAdversary

#t5-large's hidden width, which is what the reference model regresses onto
DEFAULT_EMBED_DIM = 1024


class TrialFourNet(nn.Module):
    """Front end, encoder, and the three heads.

    ATTRS:
        instance_norm (RobustInstanceNorm): per-window robust standardisation
        scale_projection (nn.Linear): the discarded scale, back as a static bias
        decimate (FixedLowpassDecimate): fixed anti-alias lowpass and downsample
        encoder (MultiRateEncoder): the Conformer stacks
        pool (nn.Module): collapses the time axis
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
        lowpass_hz=45.0,
        fir_taps=81,
        decimate_factor=2,
        stack_blocks=(2, 3, 4),
        stack_widths=(144, 192, 256),
        stack_divisors=(1, 2, 4),
        stack_kernels=(31, 15, 9),
        n_heads=4,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        with_quadrature=True,
        reuse_attention=True,
        pooling="attention",
        head_dropout=0.1,
        with_instance_norm=True,
        multi_scale_pool=True,
    ):
        """Assemble the network.

        ARGS:
            n_channels (int): input width -- 73 whitened components, not 306
            n_classes (int): 50, the scored vocabulary
            n_sessions (int): training sessions, for the adversary. 0 omits it
            embed_dim (int): target word-embedding width
            sample_rate (float): rate of the incoming window, in Hz
            lowpass_hz (float): anti-alias cutoff before decimation
            fir_taps (int): filter length, odd
            decimate_factor (int): 1 disables decimation entirely
            stack_blocks (tuple): blocks per encoder stack
            stack_widths (tuple): d_model per stack
            stack_divisors (tuple): frame-rate divisor per stack
            stack_kernels (tuple): depthwise taps per stack, odd
            n_heads (int): attention heads
            ffn_expansion (int): hidden multiple in the feed-forwards
            dropout (float): inside the encoder
            drop_path (float): maximum stochastic depth, ramped with depth
            with_quadrature (bool): the envelope branch. False is the ablation
            reuse_attention (bool): Zipformer's attention-weight sharing
            multi_scale_pool (bool): pool EVERY stack and concatenate, rather
                than pooling only the last. Free -- pooling is a softmax and a
                weighted sum -- and it is what replaces the U-Net's skip: the
                head still sees fine and coarse evidence, without paying for an
                upsample it does not need. False pools the last stack only
            pooling (str): "attention" or "mean", see heads.POOLING
            head_dropout (float): on the pooled vector, in both output heads
            with_instance_norm (bool): False is the ablation, and reverts to
                trusting the cache's fitted standardisation
        """
        super().__init__()

        if pooling not in POOLING:
            raise KeyError(f"Unknown pooling {pooling!r}; known: {sorted(POOLING)}")

        #kept so forward() can check what it is handed, per the shape-check skill
        self.n_channels = n_channels
        self.with_instance_norm = with_instance_norm

        self.instance_norm = RobustInstanceNorm() if with_instance_norm else None

        #the scale the instance norm removed comes back as a static per-window
        #bias, so amplitude is still available but no longer sets the input range
        self.scale_projection = (
            nn.Linear(n_channels, int(stack_widths[0])) if with_instance_norm else None
        )

        #factor 1 means "do not decimate", and building a filter for it would
        #only add a pointless 45 Hz lowpass
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

        self.encoder = MultiRateEncoder(
            n_input=n_channels,
            stack_blocks=stack_blocks,
            stack_widths=stack_widths,
            stack_divisors=stack_divisors,
            stack_kernels=stack_kernels,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            drop_path=drop_path,
            with_quadrature=with_quadrature,
            reuse_attention=reuse_attention,
        )

        #⚠️ one pooling module PER STACK when multi-scale, because each stack
        #has its own width and an attention scorer is width-specific
        self.multi_scale_pool = multi_scale_pool
        widths = self.encoder.stack_widths if multi_scale_pool else (
            self.encoder.output_width,
        )
        self.pools = nn.ModuleList(
            [
                POOLING[pooling](w) if pooling == "attention" else POOLING[pooling]()
                for w in widths
            ]
        )

        #the heads read the CONCATENATION, so the width is the sum
        width = sum(widths)

        self.semantic = SemanticHead(width, embed_dim, head_dropout)
        self.closed_set = ClosedSetHead(width, n_classes, head_dropout)

        #no adversary when the caller did not say how many sessions exist; a
        #head with the wrong class count trains to nonsense without raising
        self.adversary = (
            SessionAdversary(width, n_sessions, head_dropout) if n_sessions > 0 else None
        )

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times] channels FIRST

        RETURNS:
            dict with keys:
                "pooled" : [batch, width] - what every head reads
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

        #the lowpass runs BEFORE the transpose, while the layout is still the
        #channels-first one convolutions want
        if self.decimate is not None:
            x = self.decimate(x)

        #the one transpose in the whole network: channels first for the fixed
        #front end, channels last for everything attention touches
        encoded = self.encoder(
            x.transpose(1, 2),
            static_bias=static_bias,
            return_all_stacks=self.multi_scale_pool,
        )

        #each stack is pooled at its OWN frame rate and the vectors concatenated:
        #fine and coarse evidence both reach the head, with no upsample
        if self.multi_scale_pool:
            pooled = torch.cat(
                [pool(stack) for pool, stack in zip(self.pools, encoded)], dim=1
            )
        else:
            pooled = self.pools[0](encoded)

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
            pooled (torch.Tensor): [batch, width] from `forward`
            strength (float): gradient-reversal coefficient

        RETURNS:
            logits : [batch, n_sessions], or None
        """
        if self.adversary is None:
            return None
        return self.adversary(pooled, strength)
