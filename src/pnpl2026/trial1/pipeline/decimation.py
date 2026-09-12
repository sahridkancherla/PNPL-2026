"""Anti-aliased decimation -- stage 1 of the trial-1 cache.

Recordings arrive at 250 Hz bandpassed 0.1-125 Hz, so they carry real content
right up to their own Nyquist: 9.97% of total power sits above 62.5 Hz. Halving
the sample rate therefore cannot be done by dropping samples. That 10% folds
back into the retained band, on top of signal, and no later stage can undo it.

Nothing here is fitted. These are hyperparameters, not statistics of the data,
so there is no train-only discipline to observe and nothing to persist beyond
the taps themselves as provenance.

The measured filter responses, the rejected alternatives and the arithmetic
behind the tap count are in `info/maths/decimation.md`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import firwin, kaiser_beta, resample_poly


# Where the passband ends, as a fraction of the NEW Nyquist. Strictly below 1
# so the whole transition band fits underneath the fold frequency.
PASSBAND_FRACTION = 0.9

# Stopband attenuation. scipy's own default filter for factor 2 reaches only
# -6 dB at the fold, passing half the amplitude of the band it exists to remove.
STOPBAND_ATTENUATION_DB = 80.0


@dataclass
class DecimatedRecording:
    """One decimated recording, carrying the facts a caller must not guess.

    ATTRS:
        data (np.ndarray): [n_channels, n_times_out] float32, decimated
        sample_rate (float): samples per second of `data`, NOT of the input.
            Bundled with the array because word onsets are held in seconds and
            turned into indices against this -- a stale rate here misplaces
            every window by a constant factor, with no error anywhere.
        invalid_edge_s (float): seconds at EACH end that the filter transient
            has corrupted. Windows overlapping either margin must be rejected.
    """

    data: np.ndarray
    sample_rate: float
    invalid_edge_s: float


def design_antialias_filter(
    sample_rate_in,
    factor,
    attenuation_db=STOPBAND_ATTENUATION_DB,
    passband_fraction=PASSBAND_FRACTION):
    """Design the linear-phase FIR lowpass that has to precede downsampling.

    A Kaiser-windowed FIR, sized by the standard Kaiser estimate so that the
    stopband begins at or below the new Nyquist rather than straddling it.
    Symmetric, hence linear phase, so the only delay is a constant one that
    `resample_poly` compensates exactly. That property is why an FIR is used
    and not an IIR: the evoked response's latency relative to word onset is
    the signal, and an IIR's frequency-dependent group delay smears it.

    ARGS:
        sample_rate_in (float): samples per second of the recording to filter
        factor (int): decimation factor, so the surviving band is
            0 to sample_rate_in / (2 * factor)
        attenuation_db (float): stopband attenuation, as positive dB
        passband_fraction (float): where the passband ends, as a fraction of
            the new Nyquist. The band between it and the new Nyquist is
            deliberately sacrificed rather than aliased.

    RETURNS:
        taps : [n_taps] - FIR coefficients, odd length, symmetric, unity gain
            in the passband
    """
    # Guards first, since every quantity below divides by one of these.
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    if not 0.0 < passband_fraction < 1.0:
        raise ValueError(
            f"passband_fraction must be in (0, 1), got {passband_fraction}"
        )
    if attenuation_db <= 0.0:
        raise ValueError(f"attenuation_db must be > 0, got {attenuation_db}")

    # The highest frequency that survives decimation; above it, content folds.
    nyquist_out = sample_rate_in / (2 * factor)

    # End the passband short of the fold so the transition fits below it.
    passband_edge_hz = passband_fraction * nyquist_out

    # Kaiser's tap estimate; the transition width is normalised to input Nyquist.
    transition_norm = (nyquist_out - passband_edge_hz) / (sample_rate_in / 2)
    n_taps = int(
        np.ceil((attenuation_db - 7.95) / (2.285 * np.pi * transition_norm))
    )

    # Force odd, so the kernel is symmetric about a sample and the delay is whole.
    n_taps += 1 - n_taps % 2

    # firwin places its cutoff at -6 dB, so centre it in the transition band.
    cutoff_hz = (passband_edge_hz + nyquist_out) / 2

    return firwin(
        n_taps,
        cutoff_hz,
        window=("kaiser", kaiser_beta(attenuation_db)),
        fs=sample_rate_in,
    )


def decimate_recording(recording, sample_rate_in=250.0, factor=2, taps=None):
    """Lowpass and downsample one continuous recording.

    Filters the recording WHOLE, before any windowing. The kernel is ~200
    samples and a stored window is 200, so filtering windows independently
    would give every one of them a transient across most of its length.

    ARGS:
        recording (np.ndarray): [n_channels, n_times] continuous, at
            `sample_rate_in`
        sample_rate_in (float): samples per second of `recording`
        factor (int): keep one sample in this many. 2 takes 250 Hz to 125 Hz.
        taps (np.ndarray or None): FIR coefficients, to reuse one design across
            recordings. None designs them from the module defaults. Every
            recording in a cache must share one filter or the shards are not
            comparable.

    RETURNS:
        a DecimatedRecording - [n_channels, ceil(n_times / factor)] float32,
            its new sample rate, and the corrupted margin at each end
    """
    # A 1D array would be filtered along the wrong axis and come back plausible.
    if recording.ndim != 2:
        raise ValueError(
            f"Expected [n_channels, n_times], got shape {recording.shape}"
        )
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")

    # A NaN hurts more here than downstream: convolution spreads it across the
    # full kernel width, so one bad sample corrupts a couple of hundred.
    if not np.isfinite(recording).all():
        raise ValueError("Recording contains non-finite samples.")

    # Factor 1 asks for no decimation, and filtering would only cost passband.
    if factor == 1:
        return DecimatedRecording(
            data=recording.astype(np.float32, copy=False),
            sample_rate=sample_rate_in,
            invalid_edge_s=0.0,
        )

    # Design once, unless the caller is sharing one filter across recordings.
    if taps is None:
        taps = design_antialias_filter(sample_rate_in, factor)

    # Polyphase: upfirdn computes only the samples that survive, so this costs
    # about half of filtering everything and then slicing it away.
    decimated = resample_poly(recording, up=1, down=factor, axis=-1, window=taps)

    # Half a kernel of context is missing at each end, measured in input samples.
    invalid_edge_s = (len(taps) // 2) / sample_rate_in

    # float32 out: the standardiser upcasts to float64 itself for the covariance,
    # so carrying float64 from here would double the cache for nothing.
    return DecimatedRecording(
        data=decimated.astype(np.float32, copy=False),
        sample_rate=sample_rate_in / factor,
        invalid_edge_s=invalid_edge_s,
    )
