"""Stream B: slow amplitude envelopes of the 20-100 Hz band. Variant C.

High-frequency activity is kept the way ECoG speech decoders keep high gamma:
not as raw samples, which would need 250 frames and 25x the attention cost, but
as the slowly varying AMPLITUDE of a band. That amplitude is lowpassed at
20 Hz and decimated like stream A, so both streams arrive at 50 frames.

    [B, 73, 250]   whitened components, the RAW window -- not stream A's
                   normalised one, so nothing below 20 Hz leaks in through
                   the normaliser
      -> BandEnvelope          FFT band mask plus analytic signal, |.|
      -> FixedLowpassDecimate  20 Hz, / 5
      -> log                   gain drift becomes an offset, not a scale
      -> FittedProjection      PCA-whitened to ~32 components, TRAINING fit
    [B, 32, 50]

⚠️ The envelope is computed per window inside the model, never in the cache.
The holdout is served as pre-cut [0, 1] s windows, and whatever a training
window goes through, a holdout window must go through identically. Edge effects
at the window's ends are real and are the same on both.

⚠️ `FittedProjection` must be fitted before the first training step, on
training windows only -- see `fit_envelope_projections`. An unfitted one raises
rather than passing zeros through, because zeros train happily to chance.
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frontend import FixedLowpassDecimate

#width of each raised-cosine edge of the band mask, in Hz. A brick-wall mask
#rings across the whole window; 4 Hz keeps the band edges within ~2 Hz
TRANSITION_HZ = 4.0

#reflect padding either side before the FFT, so the circular convolution the
#FFT implies does not wrap the window's end onto its start
PAD_SECONDS = 0.25

#smallest amplitude the log is allowed to see; whitened components sit near
#unit scale, so this is ~4 orders of magnitude below a typical envelope
LOG_FLOOR = 1e-4

#whitened stream B is clipped here, matching RobustInstanceNorm's clamp on A
CLAMP = 5.0


def band_mask(n_fft, sample_rate, low_hz, high_hz, transition_hz=TRANSITION_HZ):
    """Weights that turn an FFT into a band-limited ANALYTIC signal.

    Positive frequencies inside the band get weight 2 and everything else 0,
    which is the Hilbert transform and the bandpass in one multiply: the
    inverse FFT is then complex, and its magnitude is the band's envelope.

    ARGS:
        n_fft (int): FFT length, the padded window
        sample_rate (float): Hz
        low_hz (float): lower -6 dB edge
        high_hz (float): upper -6 dB edge
        transition_hz (float): width of each raised-cosine edge

    RETURNS:
        mask : [n_fft] float32, ready to multiply a torch.fft.fft output
    """
    #fftfreq puts the negative frequencies in the second half; they get 0
    freqs = np.fft.fftfreq(n_fft, d=1.0 / sample_rate)

    #a rising and a falling raised-cosine edge, each centred on its cutoff
    half = transition_hz / 2
    rising = np.clip((freqs - (low_hz - half)) / transition_hz, 0.0, 1.0)
    falling = np.clip(((high_hz + half) - freqs) / transition_hz, 0.0, 1.0)
    gain = 0.5 * (1 - np.cos(np.pi * rising)) * 0.5 * (1 - np.cos(np.pi * falling))

    #x2 on positive frequencies is the analytic signal; negatives were zeroed
    #by the clips above, since freqs < 0 sit below low_hz - half
    return torch.from_numpy((2.0 * gain).astype(np.float32))


class BandEnvelope(nn.Module):
    """Hilbert amplitude of one frequency band, computed per window by FFT.

    ATTRS:
        pad (int): reflect-padded samples on each side
        n_times (int): window length the mask was built for
        mask (torch.Tensor): [n_fft] buffer from `band_mask`
    """

    def __init__(self, sample_rate, low_hz, high_hz, n_times):
        """
        ARGS:
            sample_rate (float): Hz
            low_hz (float): lower edge of the band
            high_hz (float): upper edge of the band
            n_times (int): samples in every window this will be given
        """
        super().__init__()

        #a band touching 0 Hz or Nyquist is not a band, and the analytic
        #signal's weights are wrong at both of those bins
        nyquist = sample_rate / 2
        if not TRANSITION_HZ / 2 < low_hz < high_hz < nyquist - TRANSITION_HZ / 2:
            raise ValueError(
                f"Band ({low_hz}, {high_hz}) Hz must sit strictly inside "
                f"({TRANSITION_HZ / 2}, {nyquist - TRANSITION_HZ / 2}) Hz."
            )

        #reflect padding must be shorter than the window it reflects
        self.pad = min(int(round(PAD_SECONDS * sample_rate)), n_times - 1)
        self.n_times = n_times
        self.register_buffer(
            "mask", band_mask(n_times + 2 * self.pad, sample_rate, low_hz, high_hz)
        )

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            amplitude : [batch, n_channels, n_times] - non-negative
        """
        #the mask was sized for one window length; another length would put
        #every frequency in the wrong bin without raising
        if windows.shape[2] != self.n_times:
            raise ValueError(
                f"BandEnvelope built for {self.n_times} samples, got {windows.shape[2]}."
            )

        padded = F.pad(windows, (self.pad, self.pad), mode="reflect")

        #dim=-1 is time in both transforms; the product is the band-limited
        #analytic signal, and its magnitude is the envelope
        spectrum = torch.fft.fft(padded, dim=-1)
        analytic = torch.fft.ifft(spectrum * self.mask, dim=-1)
        return analytic.abs()[..., self.pad : self.pad + self.n_times]


class FittedProjection(nn.Module):
    """A fixed PCA-whitening map, fitted once on training data.

    One per band, deliberately: a PCA shared across streams would spend its
    components on whichever stream has more variance.

    ATTRS:
        mean (torch.Tensor): [n_in] training mean, subtracted first
        components (torch.Tensor): [n_out, n_in] leading eigenvectors
        scale (torch.Tensor): [n_out] 1 / sqrt(eigenvalue), unit variance out
        is_fitted (torch.Tensor): scalar bool, persisted in the checkpoint
    """

    def __init__(self, n_in, n_out):
        """
        ARGS:
            n_in (int): width going in, the 73 components
            n_out (int): components kept
        """
        super().__init__()

        if not 0 < n_out <= n_in:
            raise ValueError(f"Cannot keep {n_out} of {n_in} components.")

        #buffers, not parameters: fitted, never learned, and they must travel
        #in the checkpoint so predict.py applies the identical projection
        self.register_buffer("mean", torch.zeros(n_in))
        self.register_buffer("components", torch.zeros(n_out, n_in))
        self.register_buffer("scale", torch.ones(n_out))
        self.register_buffer("is_fitted", torch.tensor(False))

    @torch.no_grad()
    def fit_from_moments(self, mean, covariance):
        """Set the projection from a mean and covariance.

        ARGS:
            mean (torch.Tensor): [n_in] float64
            covariance (torch.Tensor): [n_in, n_in] float64, symmetric

        RETURNS:
            float: fraction of the variance the kept components carry
        """
        #eigh returns ascending eigenvalues, so the leading ones come last
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        order = torch.argsort(eigenvalues, descending=True)[: self.components.shape[0]]
        kept = eigenvalues[order].clamp_min(1e-12)

        self.mean.copy_(mean.float())
        self.components.copy_(eigenvectors[:, order].T.float())
        self.scale.copy_(kept.rsqrt().float())
        self.is_fitted.fill_(True)

        return float(kept.sum() / eigenvalues.clamp_min(0).sum())

    def forward(self, x):
        """
        ARGS:
            x (torch.Tensor): [batch, n_in, n_frames]

        RETURNS:
            [batch, n_out, n_frames] - unit variance per component on the
            training data
        """
        #⚠️ an unfitted projection would pass all-zeros components through,
        #which trains to exactly chance and looks like a negative result
        if not bool(self.is_fitted):
            raise RuntimeError(
                "Envelope PCA used before it was fitted. cli.train fits it on a "
                "fresh start; a checkpoint restores it."
            )

        #dim 1 is the component axis; subtract the mean, rotate, rescale
        centred = x - self.mean.view(1, -1, 1)
        projected = torch.einsum("oi,bit->bot", self.components, centred)
        return projected * self.scale.view(1, -1, 1)


class EnvelopeStream(nn.Module):
    """One band of stream B: envelope, lowpass, decimate, log, PCA.

    ATTRS:
        band (tuple): (low_hz, high_hz)
        envelope (BandEnvelope)
        decimate (FixedLowpassDecimate): 20 Hz, same factor as stream A
        projection (FittedProjection)
        with_log (bool): log the amplitude before the PCA
    """

    def __init__(
        self,
        n_channels,
        sample_rate,
        n_times,
        band,
        n_components=32,
        lowpass_hz=20.0,
        n_taps=81,
        factor=5,
        with_log=True,
    ):
        """
        ARGS:
            n_channels (int): 73 whitened components
            sample_rate (float): Hz of the incoming window
            n_times (int): samples per window
            band (tuple): (low_hz, high_hz)
            n_components (int): PCA components kept, ~32 per the design
            lowpass_hz (float): smoothing of the envelope before decimation
            n_taps (int): FIR length, the same as stream A's
            factor (int): decimation, the same as stream A's
            with_log (bool): False keeps the raw Hilbert amplitude
        """
        super().__init__()

        low_hz, high_hz = band
        self.band = (float(low_hz), float(high_hz))
        self.with_log = with_log

        self.envelope = BandEnvelope(sample_rate, low_hz, high_hz, n_times)
        self.decimate = FixedLowpassDecimate(
            n_channels,
            sample_rate=sample_rate,
            lowpass_hz=lowpass_hz,
            n_taps=n_taps,
            factor=factor,
        )
        self.projection = FittedProjection(n_channels, n_components)

    @property
    def name(self):
        """A label for logs, e.g. '20-40 Hz'."""
        return f"{self.band[0]:g}-{self.band[1]:g} Hz"

    def features(self, windows):
        """Everything before the PCA -- what the PCA is fitted on.

        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times] raw components

        RETURNS:
            [batch, n_channels, n_frames]
        """
        amplitude = self.decimate(self.envelope(windows))

        #⚠️ floored before the log: the FIR can undershoot a hair below zero
        #where the envelope is nearly flat, and log(0) is -inf
        if self.with_log:
            amplitude = torch.log(amplitude.clamp_min(LOG_FLOOR))
        return amplitude

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times] raw components

        RETURNS:
            [batch, n_components, n_frames] - whitened, clipped at +/- CLAMP
        """
        return self.projection(self.features(windows)).clamp(-CLAMP, CLAMP)


@torch.no_grad()
def fit_envelope_projections(streams, dataset, n_windows, seed=0, batch_size=256):
    """Fit every stream's PCA on a random sample of TRAINING windows.

    Accumulates the mean and covariance in float64 over batches rather than
    holding every frame, so memory stays flat however many windows are used.
    Each FRAME is one observation: 16384 windows at 50 frames is ~800k.

    ARGS:
        streams (nn.ModuleList): the model's EnvelopeStreams
        dataset (CachedWindows): the TRAINING split, un-jittered
        n_windows (int): how many windows to sample
        seed (int): for the sample, so a rerun fits the same projection
        batch_size (int): windows per batch through the envelope front end

    RETURNS:
        dict: band name -> fraction of variance the kept components carry
    """
    if len(streams) == 0:
        return {}

    #sampled without replacement; the whole split when it is small
    generator = np.random.default_rng(seed)
    count = min(n_windows, len(dataset))
    indices = generator.choice(len(dataset), size=count, replace=False)

    #float64 accumulators, one per stream: frame count, sum, sum of outers
    n_channels = streams[0].projection.mean.shape[0]
    totals = [0] * len(streams)
    sums = [torch.zeros(n_channels, dtype=torch.float64) for _ in streams]
    outers = [torch.zeros(n_channels, n_channels, dtype=torch.float64) for _ in streams]

    for start in range(0, count, batch_size):
        #index 0 of each item is the window; label and session are not used
        windows = torch.stack(
            [dataset[int(index)][0] for index in indices[start : start + batch_size]]
        )

        for position, stream in enumerate(streams):
            features = stream.features(windows)

            #⚠️ the reshape below would fold a wrong channel count into rows of
            #the right width without raising, so the width is checked first
            if features.shape[1] != n_channels:
                raise ValueError(
                    f"{stream.name} features have {features.shape[1]} channels, "
                    f"the projection expects {n_channels}."
                )

            #[batch, C, F] -> [batch * F, C]: frames are the observations
            frames = features.transpose(1, 2).reshape(-1, n_channels).double()

            totals[position] += frames.shape[0]
            sums[position] += frames.sum(dim=0)
            outers[position] += frames.T @ frames

    report = {}
    for position, stream in enumerate(streams):
        total = totals[position]
        mean = sums[position] / total

        #unbiased covariance from the two running moments
        covariance = (outers[position] - total * torch.outer(mean, mean)) / (total - 1)
        report[stream.name] = stream.projection.fit_from_moments(mean, covariance)

    return report
