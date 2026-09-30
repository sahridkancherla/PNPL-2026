"""The front end: per-window robust normalisation, then a fixed decimation.

Two fixed, non-learned operations that stand between the cache and the encoder.
Both are inside the MODEL rather than inside the cache builder, and that is
deliberate:

⚠️ The holdout is served as pre-cut 250-sample windows by `iter_windows()`. Any
transform the model depends on must be applied identically to a training window
and to a holdout window. A cache stage cannot do that -- a 200-tap anti-alias
filter is fine on a continuous recording and impossible on a 250-sample window.
A fixed layer inside the model is applied to both by construction.

⚠️ Everything here is **channels first**, `[batch, n_channels, n_times]`, which
is what convolutions want. The encoder transposes to channels-last once, at its
stem.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import firwin

#1.4826 turns a median absolute deviation into an estimate of sigma for
#Gaussian data; it is the robust-statistics constant, not a tuning knob
MAD_TO_SIGMA = 1.4826


class RobustInstanceNorm(nn.Module):
    """Normalise every channel of every window by its OWN median and MAD.

    The cheapest large win available, and the one trials 1-3 did not take. The
    2025 phoneme winner measured instance normalisation as worth **+200% on
    holdout** against BatchNorm, and the reason is specific to this evaluation:
    the holdout is a HELD-OUT RECORDING of the same subject. Head position
    drifts between recordings, SSS gain varies, and sessions differ in size by
    30x. Any statistic fitted on the training recordings is the wrong statistic
    at test time. A per-window statistic has no train/test distinction at all.

    Median and MAD rather than mean and standard deviation because a single
    eyeblink or SQUID jump moves a mean and does not move a median.

    ⚠️ This DESTROYS absolute amplitude, which is real information. `forward`
    therefore also returns `log(sigma)` per channel, and the caller is expected
    to feed it back in as a static bias. The scale stops setting the network's
    input range without being thrown away.

    ATTRS:
        clamp (float): after normalising, values are clipped to +/- this
        eps (float): floor on the MAD, so a dead channel does not divide by zero
    """

    def __init__(self, clamp=5.0, eps=1e-3):
        """
        ARGS:
            clamp (float): clip threshold in robust sigmas. The reference model
                clamps at 5; the cache already clipped at 20 raw sigmas
            eps (float): added to the MAD before dividing
        """
        super().__init__()
        self.clamp = clamp
        self.eps = eps

    def forward(self, windows, valid=None):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times]
            valid (torch.Tensor or None): [batch, n_times] bool, True on real
                samples. Trial 7's windows are zero-padded past their end,
                and padding inside a median drags it towards zero

        RETURNS:
            normalised : [batch, n_channels, n_times] - robustly standardised,
                zero wherever `valid` is False
            log_scale : [batch, n_channels] - log of the discarded scale
        """
        #⚠️ the statistics come from real samples only: padding becomes NaN,
        #which nanmedian skips. Every window has at least 1 s of real signal
        source = windows
        if valid is not None:
            source = windows.masked_fill(~valid.unsqueeze(1), float("nan"))

        #dim=2 is time. An implicit reduction here would collapse channels too,
        #which is exactly the normalisation this module exists to avoid
        centre = source.nanmedian(dim=2, keepdim=True).values
        centred = windows - centre

        #MAD is the median of the absolute deviations, not their mean
        deviation = (source - centre).abs().nanmedian(dim=2, keepdim=True).values
        sigma = deviation * MAD_TO_SIGMA + self.eps

        #clamped so one surviving artefact cannot dominate a batch gradient
        normalised = (centred / sigma).clamp(-self.clamp, self.clamp)

        #padding goes back to exactly zero, as the holdout's does
        if valid is not None:
            normalised = normalised * valid.unsqueeze(1)

        #squeeze(2) explicitly: a bare squeeze() would also drop a batch of 1
        return normalised, torch.log(sigma).squeeze(2)


class FixedLowpassDecimate(nn.Module):
    """Anti-alias lowpass, then drop samples. Not learned, not trainable.

    The cache is 250 Hz and the signal is bandpassed 0.1-125 Hz, right up to
    its own Nyquist. Reference pipelines for this task cut at 40 Hz and
    resample far lower -- d'Ascoli et al. run at 50 Hz. Decimating buys compute
    proportionally, and above ~45 Hz there is little word-decoding signal to
    lose.

    ⚠️ Decimation without a filter first is aliasing: content above the new
    Nyquist folds down and lands on top of real signal, and nothing warns you.
    `CLAUDE.md` records 9.97% of total power above 62.5 Hz, so this matters.

    The filter is a windowed-sinc FIR built once at construction and stored as
    a buffer. It is shared across channels, which is what makes it a *fixed*
    operation rather than a learned one: every channel gets the identical
    treatment, on training windows and holdout windows alike.

    ATTRS:
        factor (int): decimation factor. 2 takes 250 Hz to 125 Hz
        n_taps (int): filter length, odd so the delay is a whole sample
        kernel (torch.Tensor): [n_channels, 1, n_taps] buffer, not a parameter
    """

    def __init__(self, n_channels, sample_rate=250.0, lowpass_hz=45.0, n_taps=81, factor=2):
        """Design the filter and register it.

        ARGS:
            n_channels (int): C, preserved -- this is a depthwise operation
            sample_rate (float): input rate in Hz
            lowpass_hz (float): -6 dB cutoff of the FIR
            n_taps (int): filter length. 81 taps at 250 Hz is a 324 ms window
            factor (int): how many input samples become one output sample
        """
        super().__init__()

        #an even tap count has no centre sample, so the filter delays the
        #signal by half a sample and every latency in the window shifts
        if n_taps % 2 == 0:
            raise ValueError(f"n_taps must be odd for a whole-sample delay, got {n_taps}")

        #the new Nyquist after decimating; a cutoff above this aliases, which
        #is the entire failure this module exists to prevent
        new_nyquist = sample_rate / (2 * factor)
        if lowpass_hz >= new_nyquist:
            raise ValueError(
                f"lowpass_hz {lowpass_hz} is at or above the post-decimation "
                f"Nyquist {new_nyquist} (rate {sample_rate} / factor {factor}). "
                f"Content above {new_nyquist} Hz would fold down onto real signal."
            )

        self.factor = factor
        self.n_taps = n_taps
        self.n_channels = n_channels

        #firwin returns a unit-DC-gain lowpass, so the filter does not change
        #the scale of what passes through it
        taps = firwin(n_taps, lowpass_hz, fs=sample_rate).astype(np.float32)

        #one copy per channel, because conv1d with groups=C wants [C, 1, taps]
        kernel = torch.from_numpy(taps).view(1, 1, -1).repeat(n_channels, 1, 1)

        #a buffer, not a parameter: it must never receive a gradient, and it
        #must follow the module across devices and into the checkpoint
        self.register_buffer("kernel", kernel)

    def output_length(self, n_times):
        """How many frames come out for a given input length.

        ARGS:
            n_times (int): T, input samples

        RETURNS:
            int: output frames
        """
        #same arithmetic conv1d uses, kept here so callers can size a head
        padding = (self.n_taps - 1) // 2
        return (n_times + 2 * padding - self.n_taps) // self.factor + 1

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            decimated : [batch, n_channels, n_times // factor] approximately
        """
        #odd taps plus this padding keeps the output centred on the input
        padding = (self.n_taps - 1) // 2

        #groups=n_channels makes this depthwise: no mixing across sensors, the
        #front end's only job here is the time axis
        return nn.functional.conv1d(
            windows,
            self.kernel,
            stride=self.factor,
            padding=padding,
            groups=self.n_channels,
        )


class BaselineCorrect(nn.Module):
    """Subtract each channel's mean over an early slice of the window.

    Baseline correction removes the slow drift a window inherits from whatever
    the brain was doing before the word started, so two windows of the same
    word are compared on their RESPONSE rather than on their starting level.
    The reference model does this over `(0.0, 0.5)` s of a three-second
    sentence window.

    ⚠️ We cannot copy that interval, and the reason is a hard constraint rather
    than a preference. The reference's window opens at SENTENCE onset, so its
    first half-second genuinely precedes most of the words in it. Ours opens at
    WORD onset. Subtracting 0-0.5 s here would subtract the response itself.

    ⚠️ Nor can we use a pre-onset baseline, which is what the physiology
    actually wants. The cache carries `[-0.32, 1.28]` s and could supply one --
    but `iter_windows()` serves the holdout as pre-cut windows spanning exactly
    `[0.0, 1.0]` s, so at test time there is no pre-onset sample to be had. A
    model that needed one could be trained and could never be scored.

    What is left is the early post-onset slice: the first ~100 ms, before the
    lexical and semantic effects this task depends on have arrived. That is a
    compromise and it is ablatable, because `RobustInstanceNorm` has already
    subtracted the per-channel median over the whole window and this may be
    buying nothing on top of it.

    ATTRS:
        start (int): first sample of the baseline slice
        stop (int): one past the last sample of the slice
    """

    def __init__(self, sample_rate=250.0, baseline=(0.0, 0.1)):
        """
        ARGS:
            sample_rate (float): samples per second of the incoming window
            baseline (tuple): (start, stop) in seconds from window start
        """
        super().__init__()

        start_s, stop_s = baseline
        self.start = int(round(start_s * sample_rate))
        self.stop = int(round(stop_s * sample_rate))

        #an empty or inverted slice would make the mean a NaN, which then
        #propagates through every downstream layer without raising
        if self.stop <= self.start:
            raise ValueError(
                f"Baseline {baseline} s at {sample_rate} Hz gives samples "
                f"[{self.start}, {self.stop}), which is empty."
            )

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            corrected : [batch, n_channels, n_times] - same shape
        """
        #a window shorter than the baseline slice would silently average
        #whatever part of it exists, which is a different operation
        n_times = windows.shape[2]
        if n_times < self.stop:
            raise ValueError(
                f"Window has {n_times} samples but the baseline slice ends at "
                f"{self.stop}. Check sample_rate against the cropped window."
            )

        #dim=2 is time, keepdim so it broadcasts back over the whole window
        level = windows[:, :, self.start : self.stop].mean(dim=2, keepdim=True)
        return windows - level


class ComponentDropout(nn.Module):
    """Zero whole components for whole windows. Training only.

    The analogue of sensor dropout. SSS makes the array genuinely rank-limited,
    so any one component is partly predictable from the others and the model
    should not come to depend on a single one.

    ⚠️ This lives INSIDE the model, after `RobustInstanceNorm`, and that
    placement is the whole point: upstream, the zeroed values would enter the
    median and MAD the norm computes, on training windows only.

    ATTRS:
        p (float): probability of dropping a component, per window
    """

    def __init__(self, p=0.0):
        """
        ARGS:
            p (float): drop probability. 0 makes this module the identity
        """
        super().__init__()

        if not 0.0 <= p < 1.0:
            raise ValueError(f"ComponentDropout p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            same shape as `x`, whole channels zeroed
        """
        #identity at eval time and at p=0, with no random draw at all
        if self.p == 0.0 or not self.training:
            return x

        #one draw per (window, channel), broadcast over time -- a channel is
        #either present for the whole window or absent from it
        keep = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= self.p

        #⚠️ the rescale is only correct because this runs AFTER the instance
        #norm. Before it, the norm divides by each channel's own MAD and
        #cancels the factor entirely, making it dead arithmetic
        return x * keep / (1.0 - self.p)


class GaussianNoise(nn.Module):
    """Add small white noise to every value. Training only. Variant C.

    The cheapest regulariser there is. Trial 5's training closed-set loss fell
    only 0.25 nats from its start while validation stopped improving at epoch
    16, and noise is one of the four things Variant C adds against that.

    ⚠️ Inside the model and after `RobustInstanceNorm`, for the same reason as
    `ComponentDropout`: noise added upstream enters the median and MAD the
    norm computes, and the train and holdout windows stop being comparable.

    ATTRS:
        std (float): standard deviation, in normalised units
    """

    def __init__(self, std=0.0):
        """
        ARGS:
            std (float): 0 makes this module the identity. The streams it is
                applied to are unit-scale after normalisation, so 0.1 adds
                about 1% of their power
        """
        super().__init__()

        if std < 0:
            raise ValueError(f"GaussianNoise std must be >= 0, got {std}")
        self.std = std

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            same shape as `x`, plus noise when training
        """
        #identity at eval time and at std 0, with no random draw at all
        if self.std == 0.0 or not self.training:
            return x
        return x + torch.randn_like(x) * self.std
