"""Trial 19's family: encode each item ONCE as a continuous signal, read every word at its onset.

    signal [B, 306, T] at the cache's rate, zero past each item's length
      -> clamp +/-5, one baseline per ITEM (the first 0.5 s of real samples)
      -> ChannelMerger (+ a learned offset per sensor type)      [B, 270, T]
      -> initial 1x1, SubjectLayer
      -> multi-scale stem: four parallel convs, kernels 500/250/125/62.5 ms,
         concatenated, 1x1 to d_model                            [B, d, T]
      -> `stem_depth` pre-norm blocks: LayerNorm over channels per time step,
         depthwise conv, 1x1 x4, GELU, 1x1, residual -- no BatchNorm
      -> strided conv (non-overlapping, stride s): one step per s samples
                                                                  [B, d, T']
      -> optional phoneme head on the steps (training only)      [B, T', 40]
      -> a learned marker added at each marked word's step
      -> time transformer (SentenceTransformer, reused): rotary over the
         step index, every step reads the whole item             [B, T', d]
      -> readout per word: the steps from `readout_before_s` before its onset
         to `readout_after_s` after, each plus a learned relative-step
         vector, attention-pooled over the steps that exist      [N, d]
      -> heads: t5 (d -> 1024), Variant K's second table, the word tower's
         space (d -> tower_dim)

Against the per-word family (trials 10-18): a short holdout sentence is read
once, not once per word with shifted, zero-padded cuts; words share the
transformer's full time resolution rather than one pooled vector each; an
isolated word is a 1 s item with one onset, no special case.

Step j covers samples [j s, (j + 1) s) exactly (the span is zero-padded to a
whole number of steps first), so a sample's step is floor(sample / s) and a
step's centre is (j s + s / 2) / rate: the phoneme head's labels and the
readout windows are placed by that rule (`sample_step`, `step_label_indices`).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import AttentionPool
from .merger import ChannelMerger, LinearSpatial, SubjectLayer
from .transformer import SentenceTransformer

#t5-large's hidden width, the main target space
T5_WIDTH = 1024

#the phoneme head's classes: silence/none, then pnpl's 39 ARPAbet symbols
N_PHONEME_CLASSES = 40


def kernel_samples(milliseconds, sample_rate):
    """A conv kernel of about this duration, odd so "same" padding centres it.

    ARGS:
        milliseconds (float)
        sample_rate (float): Hz

    RETURNS:
        int, odd, at least 1
    """
    samples = max(int(round(milliseconds * sample_rate / 1000.0)), 1)
    return samples if samples % 2 == 1 else samples + 1


def sample_mask(lengths, n_times):
    """Which samples of each padded item are real.

    ARGS:
        lengths (torch.Tensor): [B] real samples per item
        n_times (int): padded length

    RETURNS:
        [B, 1, n_times] float32, 1 on real samples
    """
    #dim 2 is time: sample t is real when t < length
    steps = torch.arange(n_times, device=lengths.device)
    return (steps.unsqueeze(0) < lengths.unsqueeze(1)).unsqueeze(1).float()


def sample_step(sample, stride):
    """The step a sample falls in.

    ARGS:
        sample (torch.Tensor or int): sample index within the item
        stride (int): samples per step

    RETURNS:
        same type: floor(sample / stride)
    """
    return torch.div(sample, stride, rounding_mode="floor") if torch.is_tensor(sample) else sample // stride


def step_label_indices(n_steps, stride, lag_samples):
    """Which sample's label each step predicts: its centre, `lag_samples` earlier.

    The brain trails the sound, so the step at time t is matched with the
    phoneme at t - lag. Steps whose shifted sample falls before the item
    get -1.

    ARGS:
        n_steps (int)
        stride (int): samples per step
        lag_samples (int)

    RETURNS:
        [n_steps] int64, -1 where there is no label
    """
    centres = torch.arange(n_steps) * stride + stride // 2 - lag_samples
    return torch.where(centres >= 0, centres, torch.full_like(centres, -1))


class ChannelLayerNorm(nn.Module):
    """LayerNorm across channels at each time step of a [B, C, T] tensor.

    ATTRS:
        norm (nn.LayerNorm): over C
    """

    def __init__(self, channels):
        """
        ARGS:
            channels (int): C
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [B, C, T]

        RETURNS:
            same shape; each time step normalised on its own
        """
        #channels last for the norm, then back: nothing from other steps or items
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TimeBlock(nn.Module):
    """A pre-norm residual block over time: norm, depthwise conv, 1x1 x4, GELU, 1x1.

    ATTRS:
        norm (ChannelLayerNorm)
        depthwise (nn.Conv1d): one kernel per channel
        widen, narrow (nn.Conv1d): the 1x1 pair
    """

    def __init__(self, channels, kernel=7, expansion=4):
        """
        ARGS:
            channels (int)
            kernel (int): odd
            expansion (int): the 1x1 widening factor
        """
        super().__init__()
        if kernel % 2 != 1:
            raise ValueError(f"TimeBlock kernel must be odd, got {kernel}")
        self.norm = ChannelLayerNorm(channels)
        self.depthwise = nn.Conv1d(channels, channels, kernel, padding=kernel // 2, groups=channels)
        self.widen = nn.Conv1d(channels, expansion * channels, 1)
        self.narrow = nn.Conv1d(expansion * channels, channels, 1)

    def forward(self, x, mask):
        """
        ARGS:
            x (torch.Tensor): [B, C, T], zero on padding
            mask (torch.Tensor): [B, 1, T], 1 on real samples

        RETURNS:
            [B, C, T], zero on padding
        """
        #padding re-zeroed after the conv, so a kernel only ever reads zeros there
        branch = self.depthwise(self.norm(x)) * mask
        branch = self.narrow(F.gelu(self.widen(branch)))
        return (x + branch) * mask


class ContinuousNet(nn.Module):
    """The continuous sentence encoder with per-onset readout.

    ATTRS:
        stride (int): samples per step
        clamp (float)
        baseline_samples (int): per-item baseline span, 0 for none
        spatial (nn.Module): ChannelMerger or LinearSpatial
        initial_linear (nn.Module)
        subject (nn.Module): SubjectLayer or Identity
        branches (nn.ModuleList): the multi-scale stem's parallel convs
        fuse (nn.Conv1d): 1x1 from the concatenated branches to d_model
        blocks (nn.ModuleList): TimeBlocks
        downsample (nn.Conv1d): kernel = stride = `stride`
        phoneme_head (nn.Linear or None)
        marker (nn.Parameter or None): [d_model], added at marked onsets
        time (SentenceTransformer)
        readout_before, readout_after (int): steps around an onset
        relative (nn.Embedding): one vector per readout step offset
        pool (AttentionPool)
        project (nn.Linear): d_model -> embed_dim, the t5 head
        project_second (nn.Linear or None): Variant K's second head
        project_tower (nn.Linear or None): d_model -> tower_dim
    """

    def __init__(
        self,
        n_channels,
        embed_dim=T5_WIDTH,
        sample_rate=50.0,
        clamp=5.0,
        baseline_seconds=0.5,
        spatial="merger",
        positions=None,
        sensor_types=None,
        merger_channels=270,
        merger_pos_dim=2048,
        merger_dropout=0.2,
        merger_type_bias=True,
        initial_linear=270,
        subject_layer=True,
        n_subjects=1,
        stem_kernels_ms=(500.0, 250.0, 125.0, 62.5),
        stem_channels=64,
        stem_depth=2,
        stem_kernel=7,
        stem_stride=2,
        d_model=256,
        time_depth=6,
        n_heads=8,
        ffn_expansion=4,
        attention_dropout=0.1,
        dropout=0.0,
        drop_path=0.0,
        onset_marker=True,
        readout_before_s=0.5,
        readout_after_s=1.5,
        second_head=False,
        second_embed_dim=T5_WIDTH,
        tower_dim=0,
        phoneme_head=False,
        n_phonemes=N_PHONEME_CLASSES,
    ):
        """
        ARGS:
            n_channels (int): 306
            embed_dim (int): the t5 target width
            sample_rate (float): the cache's rate
            clamp (float): +/- limit after the per-recording scaler
            baseline_seconds (float): per-item baseline span; 0 disables it
            spatial (str): "merger" or "linear"
            positions (array or None): [n_channels, 2], for the merger
            sensor_types (array or None): [n_channels] 0/1/2, for the offset
            merger_channels, merger_pos_dim, merger_dropout: the merger's
            merger_type_bias (bool): a learned score offset per sensor type
            initial_linear (int): 1x1 width after the spatial layer; 0 skips it
            subject_layer (bool), n_subjects (int): SubjectLayer
            stem_kernels_ms (sequence): the parallel convs' kernel durations
            stem_channels (int): channels per branch
            stem_depth (int): TimeBlocks before the stride
            stem_kernel (int): their depthwise kernel, odd
            stem_stride (int): samples per step
            d_model (int): the stem's output and the transformer's width
            time_depth, n_heads, ffn_expansion, attention_dropout, dropout,
                drop_path: the time transformer's
            onset_marker (bool): add a learned vector at marked onsets
            readout_before_s, readout_after_s (float): the readout window
            second_head (bool): Variant K's second head
            second_embed_dim (int): its width
            tower_dim (int): the word tower's space; 0 for no tower head
            phoneme_head (bool): the per-step phoneme classifier
            n_phonemes (int): its classes
        """
        super().__init__()
        self.n_channels = n_channels
        self.clamp = float(clamp)
        self.baseline_samples = int(round(baseline_seconds * sample_rate))
        self.stride = int(stem_stride)
        if self.stride < 1:
            raise ValueError(f"stem_stride must be >= 1, got {stem_stride}")

        #the spatial layer, with or without sensor positions
        if spatial == "merger":
            if positions is None:
                raise ValueError("spatial: merger needs sensor positions; the cache has none. Set spatial: linear.")
            self.spatial = ChannelMerger(
                positions, merger_channels, merger_pos_dim, merger_dropout,
                sensor_types=sensor_types if merger_type_bias else None,
            )
        elif spatial == "linear":
            self.spatial = LinearSpatial(n_channels, merger_channels)
        else:
            raise ValueError(f"spatial must be 'merger' or 'linear', got {spatial!r}")
        width = merger_channels

        #brainmagick's initial_linear: a 1x1 conv with no activation after it
        self.initial_linear = nn.Identity()
        if initial_linear:
            self.initial_linear = nn.Conv1d(width, initial_linear, 1)
            width = initial_linear
        self.subject = SubjectLayer(width, n_subjects) if subject_layer else nn.Identity()

        #MEGState's front end: several kernel lengths side by side
        kernels = [kernel_samples(ms, sample_rate) for ms in stem_kernels_ms]
        self.branches = nn.ModuleList(
            [nn.Conv1d(width, stem_channels, kernel, padding=kernel // 2) for kernel in kernels]
        )
        self.fuse = nn.Conv1d(stem_channels * len(kernels), d_model, 1)
        self.blocks = nn.ModuleList([TimeBlock(d_model, stem_kernel) for _ in range(stem_depth)])

        #non-overlapping, so step j is exactly samples [j s, (j + 1) s)
        self.downsample = nn.Conv1d(d_model, d_model, self.stride, stride=self.stride)
        self.phoneme_head = nn.Linear(d_model, n_phonemes) if phoneme_head else None
        self.marker = nn.Parameter(torch.zeros(d_model)) if onset_marker else None

        self.time = SentenceTransformer(
            d_model=d_model,
            depth=time_depth,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            attention_dropout=attention_dropout,
            dropout=dropout,
            drop_path=drop_path,
        )

        #the readout window in steps, and one learned vector per offset in it
        step_seconds = self.stride / float(sample_rate)
        self.readout_before = int(round(readout_before_s / step_seconds))
        self.readout_after = max(int(round(readout_after_s / step_seconds)), 1)
        self.relative = nn.Embedding(self.readout_before + self.readout_after, d_model)
        self.pool = AttentionPool(d_model)

        #the heads: t5 always; the others when asked
        self.project = nn.Linear(d_model, embed_dim)
        self.project_second = nn.Linear(d_model, second_embed_dim) if second_head else None
        self.project_tower = nn.Linear(d_model, tower_dim) if tower_dim else None

    def stem(self, signal, lengths, subject=None):
        """The signal to steps: everything before the transformer.

        ARGS:
            signal (torch.Tensor): [B, n_channels, T], zero past each length
            lengths (torch.Tensor): [B]
            subject (torch.Tensor or None): [B] subject layers; None is 0

        RETURNS:
            steps : [B, T', d_model], zero on invalid steps
            step_valid : [B, T'] bool
        """
        #the sensor axis is the one a wrong transpose would put time on
        n_items, n_channels, n_times = signal.shape
        assert n_channels == self.n_channels, f"expected {self.n_channels} channels, got {n_channels}"
        assert lengths.shape == (n_items,), f"lengths {tuple(lengths.shape)} for {n_items} items"

        #float32 for the clamp and the baseline, whatever the cache stored
        x = signal.float().clamp(-self.clamp, self.clamp)
        mask = sample_mask(lengths, n_times)

        #one baseline per item: the mean of its first real samples, per channel
        if self.baseline_samples > 0:
            head = mask[:, :, : self.baseline_samples]
            baseline = (x[:, :, : self.baseline_samples] * head).sum(dim=2, keepdim=True) / head.sum(dim=2, keepdim=True).clamp_min(1.0)
            x = (x - baseline) * mask

        #sensors -> virtual channels -> subject space, padding kept at zero
        x = self.spatial(x)
        x = self.initial_linear(x)
        x = (self.subject(x, subject) if isinstance(self.subject, SubjectLayer) else self.subject(x)) * mask

        #dim=1 is the channel axis: the branches side by side, then fused
        x = F.gelu(self.fuse(torch.cat([branch(x) for branch in self.branches], dim=1))) * mask
        for block in self.blocks:
            x = block(x, mask)

        #zeros to a whole number of steps, so the last partial cell -- and any
        #onset in it -- still gets a step: T' = ceil(T / s)
        remainder = x.shape[2] % self.stride
        if remainder:
            x = F.pad(x, (0, self.stride - remainder))

        #[B, d, T'] -> [B, T', d]; step j is valid when its first sample is real
        steps = self.downsample(x).transpose(1, 2)
        n_steps = steps.shape[1]
        step_valid = (torch.arange(n_steps, device=lengths.device).unsqueeze(0) * self.stride) < lengths.unsqueeze(1)
        return steps * step_valid.unsqueeze(2).to(steps.dtype), step_valid

    def readout(self, context, step_valid, onset, sequence):
        """One vector per word from the steps around its onset.

        ARGS:
            context (torch.Tensor): [B, T', d] the transformer's output
            step_valid (torch.Tensor): [B, T'] bool
            onset (torch.Tensor): [N] onset sample within the item
            sequence (torch.Tensor): [N] item of each word

        RETURNS:
            [N, d]
        """
        n_steps = context.shape[1]
        onset_step = sample_step(onset, self.stride)

        #[N, W]: the window's steps, before the onset first
        offsets = torch.arange(-self.readout_before, self.readout_after, device=onset.device)
        window = onset_step.unsqueeze(1) + offsets.unsqueeze(0)
        inside = (window >= 0) & (window < n_steps)
        clamped = window.clamp(0, n_steps - 1)
        valid = inside & step_valid[sequence.unsqueeze(1), clamped]

        #⚠️ every word's onset step lies inside its item, so no window is empty
        assert bool(valid.any(dim=1).all()), "a readout window has no valid step"

        #the step's features plus where it sits relative to the onset
        features = context[sequence.unsqueeze(1), clamped] + self.relative.weight.unsqueeze(0)
        return self.pool(features, valid)

    def forward(self, signal, lengths, onset, marker, sequence, subject=None):
        """
        ARGS:
            signal (torch.Tensor): [B, n_channels, T], zero-padded
            lengths (torch.Tensor): [B] real samples per item
            onset (torch.Tensor): [N] onset sample of each word within its item
            marker (torch.Tensor): [N] bool, which onsets get the marker
            sequence (torch.Tensor): [N] which item each word belongs to
            subject (torch.Tensor or None): [B] subject layers

        RETURNS:
            dict:
                "words" : [N, d_model] - the readout vectors
                "embedding" : [N, embed_dim] - the t5 head, compared by cosine
                "embedding_second" : [N, second_embed_dim], with second_head
                "brain_tower" : [N, tower_dim], with a tower head
                "phoneme_logits" : [B, T', n_phonemes], with the phoneme head
                "step_valid" : [B, T'] bool
        """
        #⚠️ every per-word array must describe the same words, in order
        n_words = onset.shape[0]
        assert sequence.shape == marker.shape == (n_words,), (
            f"sequence {tuple(sequence.shape)} and marker {tuple(marker.shape)} for {n_words} words"
        )
        steps, step_valid = self.stem(signal, lengths, subject)
        out = {"step_valid": step_valid}

        #the phoneme head reads the stem, before any word has met another
        if self.phoneme_head is not None:
            out["phoneme_logits"] = self.phoneme_head(steps)

        #a marked onset's step gets the marker once per marked word there
        if self.marker is not None and bool(marker.any()):
            counts = torch.zeros(steps.shape[:2], device=steps.device, dtype=steps.dtype)
            marked = marker.nonzero(as_tuple=True)[0]
            counts.index_put_(
                (sequence[marked], sample_step(onset[marked], self.stride)),
                torch.ones(len(marked), device=steps.device, dtype=steps.dtype),
                accumulate=True,
            )
            steps = steps + counts.unsqueeze(2) * self.marker.to(steps.dtype)

        context = self.time(steps, step_valid)
        words = self.readout(context, step_valid, onset, sequence)
        out["words"] = words
        out["embedding"] = self.project(words)
        if self.project_second is not None:
            out["embedding_second"] = self.project_second(words)
        if self.project_tower is not None:
            out["brain_tower"] = self.project_tower(words)
        return out
