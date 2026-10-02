"""The trial-11 network: one stream per sentence, a multi-rate encoder over it.

    streams          [S, 73, 1200]  each sentence's MEG from its first scored
                                    onset to its end, zero-padded to 4.8 s
      -> RobustInstanceNorm over the stream's real samples
      -> FixedLowpassDecimate 20 Hz, /5                 [S, 73, 240]  50 Hz
      -> GaussianNoise, ComponentDropout                training only
      -> spatial, initial_linear, DilatedConvTrunk      [S, 256, 240]
      -> + amplitude bias, transpose                    [S, 240, 256]
      -> ZipformerEncoder  50/25/12.5/6.25/12.5/25 Hz   [S, 240, 256]
      -> per word: AttentionPool over its 1 s from onset
                                                        [N, 256]
      -> ClosedSetHead, SemanticHead                    [N, 50], [N, 1024]

Trials 7-9 encoded every word's window separately, then let the word VECTORS
attend to each other. Here the sentence stays one timeline until the very end:
words meet at frame level, at every rate, before anything is pooled. Each
sample is also encoded once, where trial 7 re-encoded it for every word whose
window covered it.

Trial 7's front end and trunk are unchanged, except that there is no
baseline: a stream's first 0.5 s is its first word's response.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .frontend import ComponentDropout, FixedLowpassDecimate, GaussianNoise, RobustInstanceNorm
from .heads import AttentionPool, ClosedSetHead, SemanticHead
from .trunk import DilatedConvTrunk
from .zipformer import ZipformerEncoder

#t5-large's hidden width, which the semantic head regresses onto
DEFAULT_EMBED_DIM = 1024


class StreamZipformerNet(nn.Module):
    """Sentence stream in, one prediction per scored word out.

    ATTRS:
        n_channels (int): 73 whitened components
        n_times (int): the padded stream length in samples
        factor (int): samples per frame
        n_frames (int): frames per padded stream
        readout_frames (int): frames each word is pooled over
        encoder (ZipformerEncoder)
        pool (AttentionPool): one word's frames to one vector
        closed_set (ClosedSetHead)
        semantic (SemanticHead)
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        embed_dim=DEFAULT_EMBED_DIM,
        sample_rate=250.0,
        n_times=1200,
        word_mask_p=0.0,
        component_dropout=0.0,
        gaussian_noise=0.0,
        lowpass_hz=20.0,
        fir_taps=81,
        decimate_factor=5,
        with_instance_norm=True,
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
        stack_factors=(1, 2, 4, 8, 4, 2),
        stack_depths=(1, 1, 1, 1, 1, 1),
        n_heads=8,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        transformer_norm="layer",
        readout_seconds=1.0,
        head_dropout=0.1,
    ):
        """Assemble the network.

        ARGS:
            n_channels (int): 73 whitened components
            n_classes (int): 50, the scored vocabulary
            embed_dim (int): target word-embedding width
            sample_rate (float): Hz of the incoming stream
            n_times (int): padded stream length; every stream arrives this long
            word_mask_p (float): must be 0 -- trial 11 has no word tokens to mask
            component_dropout (float): training-only component dropout
            gaussian_noise (float): training-only noise std
            lowpass_hz (float): anti-alias cutoff before decimation
            fir_taps (int): filter length, odd
            decimate_factor (int): 5 takes 250 Hz to 50 Hz
            with_instance_norm (bool): False skips the robust norm
            spatial_width (int): the linear map off the components
            initial_linear (int): width handed to the trunk
            trunk_hidden (int): interior width of the trunk
            trunk_depth (int): convolution layers
            trunk_kernel (int): taps per layer, odd
            dilation_growth (int): ladder multiplier per layer
            dilation_period (int or None): None never resets: 1,2,4,8,16
            glu_every (int): gate after every this many layers
            trunk_dropout (float): inside each trunk layer
            d_model (int): width of the stream, the trunk's output
            stack_factors (list): rate divisor per Zipformer stack. All 1 is Z0
            stack_depths (list): blocks per stack
            n_heads (int): attention heads
            ffn_expansion (int): FFN hidden multiple
            dropout (float): attention and FFN dropout
            drop_path (float): maximum stochastic depth
            transformer_norm (str): "layer" or "bias"
            readout_seconds (float): how long after its onset a word is read
            head_dropout (float): on the word vector, in both heads
        """
        super().__init__()

        #a masked word has no meaning here: words are not tokens until the end
        if word_mask_p > 0:
            raise ValueError("Trial 11 has no word tokens to mask; set word_mask_p: 0.")

        self.n_channels = n_channels
        self.n_times = n_times
        self.factor = decimate_factor

        self.instance_norm = RobustInstanceNorm() if with_instance_norm else None
        self.decimate = FixedLowpassDecimate(
            n_channels,
            sample_rate=sample_rate,
            lowpass_hz=lowpass_hz,
            n_taps=fir_taps,
            factor=decimate_factor,
        )
        self.n_frames = self.decimate.output_length(n_times)
        self.readout_frames = int(round(readout_seconds * sample_rate / decimate_factor))

        #⚠️ after the robust norm, never before: upstream, noise and zeroed
        #components would enter the median and MAD the norm computes
        self.gaussian_noise = GaussianNoise(gaussian_noise)
        self.component_dropout = ComponentDropout(component_dropout)

        #⚠️ a GELU between these two is load-bearing: two linear maps in a row
        #collapse to one
        self.spatial = nn.Sequential(nn.Conv1d(n_channels, spatial_width, 1), nn.GELU())
        self.initial_linear = nn.Sequential(nn.Conv1d(spatial_width, initial_linear, 1), nn.GELU())

        #the amplitude the robust norm removed, back as a per-stream bias
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

        self.encoder = ZipformerEncoder(
            d_model=d_model,
            factors=tuple(stack_factors),
            depths=tuple(stack_depths),
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            drop_path=drop_path,
            norm=transformer_norm,
        )

        #fails at build, not at the first batch, if a factor does not tile
        self.encoder.check_frames(self.n_frames)

        self.pool = AttentionPool(d_model)
        self.closed_set = ClosedSetHead(d_model, n_classes, head_dropout)
        self.semantic = SemanticHead(d_model, embed_dim, head_dropout)

    def frame_valid(self, lengths):
        """Which frames of each stream lie inside its real samples.

        ARGS:
            lengths (torch.Tensor): [S] real samples per stream

        RETURNS:
            [S, n_frames] bool
        """
        #the FIR is centred and strided, so frame f sits on input sample f*factor
        centres = torch.arange(self.n_frames, device=lengths.device) * self.factor
        return centres.unsqueeze(0) < lengths.unsqueeze(1)

    def encode_streams(self, streams, lengths):
        """Front end, trunk and Zipformer over every stream.

        ARGS:
            streams (torch.Tensor): [S, n_channels, n_times], zero past `lengths`
            lengths (torch.Tensor): [S] int64 real samples per stream

        RETURNS:
            frames : [S, n_frames, d_model]
            valid : [S, n_frames] bool
        """
        #a stream of the wrong width or length would still run through every
        #layer here, and score something it never trained on
        if streams.shape[1] != self.n_channels or streams.shape[2] != self.n_times:
            raise ValueError(
                f"Expected streams [S, {self.n_channels}, {self.n_times}], got "
                f"{tuple(streams.shape)}. Pad every stream to the full length."
            )

        samples = torch.arange(self.n_times, device=streams.device)
        valid_samples = samples.unsqueeze(0) < lengths.unsqueeze(1)

        #padding is zeroed HERE rather than trusted; the cache stores float16
        x = streams.float() * valid_samples.unsqueeze(1)

        static_bias = None
        if self.instance_norm is not None:
            x, log_scale = self.instance_norm(x, valid_samples)
            static_bias = self.scale_projection(log_scale)

        x = self.decimate(x)
        valid = self.frame_valid(lengths)

        #noise would land on the padding too; the mask takes it back off
        x = self.component_dropout(self.gaussian_noise(x)) * valid.unsqueeze(1)

        x = self.trunk(self.initial_linear(self.spatial(x)))

        #channels first for the convolutions, channels last from here on
        x = x.transpose(1, 2)

        #unsqueeze(1) names the TIME axis the per-stream bias broadcasts over
        if static_bias is not None:
            x = x + static_bias.unsqueeze(1)

        return self.encoder(x, valid), valid

    def read_words(self, frames, valid, sequence, offsets):
        """Pool each word's frames from its onset, within its own stream.

        ARGS:
            frames (torch.Tensor): [S, n_frames, d_model]
            valid (torch.Tensor): [S, n_frames] bool
            sequence (torch.Tensor): [N] int64, the stream each word is in
            offsets (torch.Tensor): [N] int64, its onset in samples from the
                stream's start

        RETURNS:
            words : [N, d_model]
        """
        #the nearest frame to each onset; frame f sits on sample f * factor
        onset_frames = torch.div(offsets + self.factor // 2, self.factor, rounding_mode="floor")

        #[N, readout_frames] frame indices, clipped into the stream
        steps = torch.arange(self.readout_frames, device=frames.device)
        index = onset_frames.unsqueeze(1) + steps.unsqueeze(0)
        inside = index < self.n_frames
        index = index.clamp(max=self.n_frames - 1)

        #[N, readout_frames, d]: each word's frames, gathered from its stream
        rows = sequence.unsqueeze(1).expand_as(index)
        gathered = frames[rows, index]

        #real frames of the word's own stream only; the onset is always real
        word_valid = valid[rows, index] & inside
        return self.pool(gathered, word_valid)

    def forward(self, streams, lengths, sequence, offsets):
        """
        ARGS:
            streams (torch.Tensor): [S, n_channels, n_times]
            lengths (torch.Tensor): [S] int64 real samples per stream
            sequence (torch.Tensor): [N] int64 stream index of each word
            offsets (torch.Tensor): [N] int64 onset of each word, in samples
                from its stream's start

        RETURNS:
            dict:
                words : [N, d_model], the pooled vector both heads read
                context : the same tensor -- the name the gradient probe uses
                logits : [N, n_classes]
                embedding : [N, embed_dim]
                masked : [N] bool, all False -- nothing is masked here
        """
        frames, valid = self.encode_streams(streams, lengths)
        words = self.read_words(frames, valid, sequence, offsets)

        return {
            "words": words,
            "context": words,
            "logits": self.closed_set(words),
            "embedding": self.semantic(words),
            "masked": torch.zeros(words.shape[0], dtype=torch.bool, device=words.device),
        }
