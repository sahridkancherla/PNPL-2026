"""Input transforms applied to (306, 250) MEG windows."""

from __future__ import annotations


def normalize(x):
    """Per-sensor standardisation. TODO: fit stats on train only."""
    raise NotImplementedError


def augment(x):
    """Train-time augmentation (channel dropout, time shift, ...). TODO."""
    raise NotImplementedError
