"""Fitted preprocessing for continuous MEG recordings.

Everything in this module is **fitted on the training recordings only** and
applied unchanged to validation and holdout. Fitting on all recordings leaks
each validation session's own statistics into the transform, which defeats the
one thing holding out whole sessions is meant to measure.

Transforms here operate on continuous `[n_channels, n_times]` recordings, not
on cut windows -- filtering and covariance estimation both need the unbroken
stream. Window-level transforms live in `transforms.py`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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
        recordings: Iterable[np.ndarray],
        subsample: int = 50,
    ) -> "ChannelStandardiser":
        """Fit a centre and scale per channel over the training recordings.

        Median and IQR rather than mean and standard deviation: MEG retains
        blinks, movement and SQUID jumps even after SSS. Both the mean and the
        standard deviation are unbounded in a single point -- one jump of
        magnitude M inflates the standard deviation by roughly M / sqrt(T) --
        so an artifact would end up setting the scale meant to remove it. The
        median moves by at most one rank position however extreme the outlier.

        Percentiles need the values themselves rather than running sums, so a
        thinned copy is held in memory. Every 50th timepoint of ~130 decimated
        recordings is ~262k samples for ~320 MB, and puts the percentile
        standard error near 1e-3 -- far below anything that matters here.

        ARGS:
            recordings: iterable of [n_channels, n_times] arrays, already
                decimated. Consumed once, so pass a fresh iterable per call.
            subsample (int): keep every nth timepoint when estimating
                percentiles. 1 uses every sample and needs ~16 GB.

        RETURNS:
            a fitted ChannelStandardiser
        """
        if subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {subsample}")

        # Thin each recording as it arrives rather than holding all of them.
        thinned = [recording[:, ::subsample] for recording in recordings]
        if not thinned:
            raise ValueError("No recordings were given to fit on.")

        # [n_channels, n_kept] -- one pooled cloud over every training session.
        pooled = np.concatenate(thinned, axis=1)

        centre = np.median(pooled, axis=1)
        upper, lower = np.percentile(pooled, [75, 25], axis=1)
        scale = (upper - lower) / NORMAL_IQR

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
