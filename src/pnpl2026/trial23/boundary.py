"""Boundary features for the per-word sentence transformer: P6 of the 13 Oct research note.

36% of holdout rows are a sentence's first or last word, and at a sentence's
edges the context the transformer leans on is cut short by silence. The P6
specialist is told, per token, where it sits:

    is_first, is_last          its place in its item
    log gap before / after     to the neighbouring listed onsets, in units of
                               a typical 0.25 s gap, clipped to +/-3
    has gap before / after     0 where the edge has no neighbour

The same function serves training (`SequenceWindows`, from the cache's
onsets) and prediction (the holdout predictor, from `onset_s`), so the two
can never disagree.
"""

from __future__ import annotations

import numpy as np

#features per token, in the order above
BOUNDARY_WIDTH = 6

#a typical gap between neighbouring listed onsets, seconds
TYPICAL_GAP = 0.25

#log-gap features are clipped to this
LOG_GAP_CLIP = 3.0


def boundary_features(onsets_s):
    """Every token's boundary features within one item.

    ARGS:
        onsets_s (np.ndarray): [n_tokens] onsets in seconds, ascending

    RETURNS:
        [n_tokens, BOUNDARY_WIDTH] float32
    """
    onsets = np.asarray(onsets_s, dtype=np.float64)
    n_tokens = len(onsets)
    features = np.zeros((n_tokens, BOUNDARY_WIDTH), dtype=np.float32)
    if n_tokens == 0:
        return features

    #the item's edges
    features[0, 0] = 1.0
    features[-1, 1] = 1.0

    #gaps to the neighbours, on a log scale around a typical gap
    gaps = np.diff(onsets)
    log_gaps = np.clip(np.log(np.maximum(gaps, 1e-3) / TYPICAL_GAP), -LOG_GAP_CLIP, LOG_GAP_CLIP)
    features[1:, 2] = log_gaps
    features[:-1, 3] = log_gaps
    features[1:, 4] = 1.0
    features[:-1, 5] = 1.0
    return features
