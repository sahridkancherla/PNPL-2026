"""Fitted preprocessing for continuous MEG recordings.

Everything in this module is **fitted on the training recordings only** and
applied unchanged to validation and holdout. Fitting on all recordings leaks
each validation session's own statistics into the transform, which defeats the
one thing holding out whole sessions is meant to measure.

Transforms here operate on continuous `[n_channels, n_times]` recordings, not
on cut windows -- filtering and covariance estimation both need the unbroken
stream. Per-window and per-batch transforms (cropping, augmentation) happen
later, on the GPU, and do not belong here.

The maths behind each choice is recorded in `info/maths/`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np

# The interquartile range of a standard normal, 2 * 0.6745. Dividing an IQR by
# this makes the result agree with the standard deviation on clean Gaussian
# data, so the robust scale stays comparable to a familiar one.
NORMAL_IQR = 1.349


@dataclass
class ChannelStandardiser:
    """Per-channel centre and scale, robust to MEG artifacts.

    Magnetometers and gradiometers are different instruments in different
    units -- measured on this hardware, gradiometers are ~21.6x larger. Any
    later step that treats channels as commensurable is otherwise dominated by
    the gradiometers; in the covariance the ratio is squared to ~466x, which
    would push every eigenvector into the gradiometer subspace and discard the
    102 magnetometers by arithmetic accident rather than by decision.

    ATTRS:
        centre (np.ndarray): [n_channels] per-channel median
        scale (np.ndarray): [n_channels] per-channel IQR / NORMAL_IQR
    """

    centre: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,

        #iterate through every recording
        recordings: Iterable[np.ndarray],
        subsample: int = 50,
        seed: int = 0,
    ) -> Self:
        """Fit a centre and scale per channel over the training recordings.

        Median and IQR rather than mean and standard deviation: MEG retains
        blinks, movement and SQUID jumps even after SSS. Both the mean and the
        standard deviation are unbounded in a single point -- one jump of
        magnitude M inflates the standard deviation by roughly M / sqrt(T) --
        so an artifact would end up setting the scale meant to remove it. The
        median moves by at most one rank position however extreme the outlier.

        A mean streams in two numbers and a covariance streams in a fixed
        matrix, but a percentile needs every value at once in order to sort
        them, so the fit holds a thinned copy instead. Keeping one timepoint in
        50 leaves ~262k samples for ~320 MB and costs ~0.15% on the estimate,
        against 16 GB to keep all of them.

        Timepoints are drawn at RANDOM, not every nth. Strided sampling of a
        125 Hz recording at every 50th is a 2.5 Hz sampler, so any component at
        a multiple of 2.5 Hz lands on the same phase every time -- alpha at
        10 Hz above all. Measured on a synthetic 10 Hz oscillation, strided
        sampling underestimated the scale by 68% where random was within 2%.

        ARGS:
            recordings: iterable of [n_channel, times] arrays.
            subsample (int): keep one timepoint in this many. 1 keeps all
                of them and needs ~16 GB.
            seed (int): seed for the timepoint draw, so a rebuilt cache is
                reproducible.

        RETURNS:
            a fitted ChannelStandardiser
        """
        if subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {subsample}")

        # Thin each recording as it arrives rather than holding all of them.
        rng = np.random.default_rng(seed)
        thinned = []
        for recording in recordings:
            n_times = recording.shape[1]
            # At least one timepoint even from a recording shorter than the
            # subsample factor, so a short session is thinned and not lost.
            n_keep = max(1, n_times // subsample)
            keep = rng.choice(n_times, size=n_keep, replace=False)
            # Sorted indices read the recording front to back, which is much
            # kinder to the cache than gathering in scattered order.
            keep.sort()
            # Fancy indexing returns a copy, so nothing pins the parent array
            # alive -- a strided view would have kept all 16 GB resident.
            thinned.append(recording[:, keep])
        if not thinned:
            raise ValueError("No recordings were given to fit on.")

        # [n_channels, n_kept] -- one pooled cloud over every training session.
        pooled = np.concatenate(thinned, axis=1)

        # A single surviving timepoint gives an IQR of exactly zero, which the
        # flat-channel guard below would misreport as a dead sensor.
        if pooled.shape[1] < 2:
            raise ValueError(
                f"Only {pooled.shape[1]} timepoint(s) survived subsample="
                f"{subsample}; an IQR needs at least two. Lower subsample."
            )

        centre = np.median(pooled, axis=1)
        upper, lower = np.percentile(pooled, [75, 25], axis=1)
        scale = (upper - lower) / NORMAL_IQR

        # NaN slips silently through every comparison below, and no choice of
        # centre or scale repairs a channel whose samples are already NaN.
        if not (np.isfinite(centre).all() and np.isfinite(scale).all()):
            raise ValueError(
                "Non-finite centre or scale: channel(s) "
                f"{np.flatnonzero(~np.isfinite(centre) | ~np.isfinite(scale)).tolist()} "
                "contain NaN or inf. Fix the recording, not the statistic."
            )

        # A channel with no spread would divide by zero. Upstream bad-channel
        # repair should prevent this, so it means something is wrong: pass the
        # channel through as zeros rather than emitting infinities downstream.
        flat = scale <= 0
        if flat.any():
            print(
                f"[preproc] WARNING: {int(flat.sum())} channel(s) have zero "
                f"spread and will be zeroed: {np.flatnonzero(flat).tolist()}"
            )
            scale = np.where(flat, 1.0, scale)

        return cls(centre=centre, scale=scale)

    def apply(self, recording: np.ndarray) -> np.ndarray:
        """Standardise one recording.

        Returns float64 regardless of input dtype: the covariance accumulated
        from this in `Whitener.fit` is a sum over millions of timepoints, and
        float32 loses meaningful precision over that many additions.

        ARGS:
            recording (np.ndarray): [n_channels, n_times], continuous

        RETURNS:
            standardised : [n_channels, n_times] - float64, zero-centred,
                each channel with unit robust scale
        """
        n_channels = recording.shape[0]
        if n_channels != len(self.centre):
            raise ValueError(
                f"Expected {len(self.centre)} channels, got {n_channels}. "
                "The standardiser was fitted on a different sensor set."
            )

        # The None index broadcasts each per-channel constant along time.
        centred = recording.astype(np.float64, copy=False) - self.centre[:, None]
        return centred / self.scale[:, None]
