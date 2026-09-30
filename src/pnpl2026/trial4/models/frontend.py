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

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            normalised : [batch, n_channels, n_times] - robustly standardised
            log_scale : [batch, n_channels] - log of the discarded scale
        """
        #dim=2 is time. An implicit reduction here would collapse channels too,
        #which is exactly the normalisation this module exists to avoid
        centre = windows.median(dim=2, keepdim=True).values
        centred = windows - centre

        #MAD is the median of the absolute deviations, not their mean
        deviation = centred.abs().median(dim=2, keepdim=True).values
        sigma = deviation * MAD_TO_SIGMA + self.eps

        #clamped so one surviving artefact cannot dominate a batch gradient
        normalised = (centred / sigma).clamp(-self.clamp, self.clamp)

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
