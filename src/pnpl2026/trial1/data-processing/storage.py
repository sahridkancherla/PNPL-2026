"""
sloppy v1 built with AI. reassess later
"""

import numpy as np


#clip threshold in standard deviations, applied to whitened windows
CLIP_SIGMA = 20.0


def clip_and_cast(windows, clip_sigma=CLIP_SIGMA):
    """

    ARGS:
        windows (np.ndarray): [n_windows, n_components, n_samples] whitened
        clip_sigma (float): clip to +/- this many standard deviations

    RETURNS:
        stored : [n_windows, n_components, n_samples] float16, halved in size
        clipped_fraction (float): fraction of values the clip actually moved,
            so the threshold can be measured rather than assumed
    """

    #check the window stack shape
    if windows.ndim != 3:
        raise ValueError(
            f"Expected [n_windows, n_components, n_samples], got {windows.shape}"
        )

    #a non-positive threshold would flatten every window to zero
    if clip_sigma <= 0:
        raise ValueError(f"clip_sigma must be > 0, got {clip_sigma}")

    #clipping cannot repair a NaN, and float16 keeps it, so stop here instead
    if not np.isfinite(windows).all():
        raise ValueError("Windows contain non-finite values.")

    #count what the clip touches, so the threshold is a measurement not a guess
    is_clipped = np.abs(windows) > clip_sigma
    clipped_fraction = float(is_clipped.mean())

    #bound the range before the cast, since float16 overflows to inf at 65504
    clipped = np.clip(windows, -clip_sigma, clip_sigma)

    return clipped.astype(np.float16), clipped_fraction
