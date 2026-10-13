"""One pipeline for the testbed and the holdout: shared components -> residual windows -> a vote.

`cli/testbed.py` (Sherlock chapters, labelled) and `cli/cross_graph.py` (the
holdout) call exactly these functions, so what the testbed measures is what
the holdout gets:

    shared_series     MCCA over every subject's aligned pieces (subject 0's
                      included in the fit, never in a vote), each piece
                      low-passed at 20 Hz first; each subject's own estimate
                      of the C shared components, piece by piece
    row_features      each row's window of a subject's series, flattened
    vote              P1 graph / SNF / P3 LDA / P8a product from the OTHER
                      subjects' residual features, seeded by the base mix
    blend             z(base) + w * z(vote) on the voting rows
"""

from __future__ import annotations

import numpy as np

from . import crossfeatures as cf
from .crosssubject import Whitener, fit_mcca
from .transduction import fit_temperature_confidence, softmax_rows

#the methods, by the names every report and settings file uses
METHODS = ("graph", "snf", "lda", "product")

#per-subject PCA before MCCA, below the SSS rank
N_PCS = 60

#the base mix's confidence after tempering, for the propagation's seeds
SEED_CONFIDENCE = 0.3


def fit_shared(pieces_by_subject, n_components, n_pcs=N_PCS):
    """The MCCA fit itself: each subject's whitener and block, after the 20 Hz low-pass.

    ARGS:
        pieces_by_subject (list): per subject, a list of [306, n_i] scaled
            50 Hz pieces, aligned piece for piece across subjects
        n_components (int): C shared components kept
        n_pcs (int)

    RETURNS:
        filtered pieces (same nesting), whiteners (list), blocks (list of
        [n_pcs, C]), the MCCA eigenvalues [C]
    """
    #every piece low-passed on its own, then stacked per subject
    filtered = [[cf.lowpass(piece) for piece in pieces] for pieces in pieces_by_subject]
    stacked = [np.concatenate(pieces, axis=1) for pieces in filtered]
    whiteners = [Whitener(data, n_pcs) for data in stacked]
    blocks, eigenvalues = fit_mcca([whitener(data) for whitener, data in zip(whiteners, stacked)], n_components)
    return filtered, whiteners, blocks, eigenvalues


def shared_series(pieces_by_subject, n_components, n_pcs=N_PCS):
    """Each subject's estimate of the shared components, piece by piece.

    ARGS:
        pieces_by_subject (list): per subject, a list of [306, n_i] scaled
            50 Hz pieces, aligned piece for piece across subjects
        n_components (int): C shared components kept
        n_pcs (int)

    RETURNS:
        list (per subject) of lists (per piece) of [C, n_i] float64;
        and the MCCA eigenvalues [C]
    """
    filtered, whiteners, blocks, eigenvalues = fit_shared(pieces_by_subject, n_components, n_pcs)

    #each subject's own projection, applied piece by piece
    series = [[block.T @ whitener(piece) for piece in pieces] for whitener, block, pieces in zip(whiteners, blocks, filtered)]
    return series, eigenvalues


def row_features(series, piece_of_row, onset_in_piece):
    """Every row's flattened window of one subject's shared series.

    ARGS:
        series (list): [C, n_i] per piece, from `shared_series`
        piece_of_row (np.ndarray): [n] which piece each row lies in
        onset_in_piece (np.ndarray): [n] the row's onset sample in its piece

    RETURNS:
        [n, C * WINDOW_LENGTH] float64
    """
    features = np.zeros((len(piece_of_row), series[0].shape[0] * cf.WINDOW_LENGTH))
    for piece in np.unique(piece_of_row):
        rows = np.flatnonzero(piece_of_row == piece)
        windows = cf.onset_windows(series[piece], onset_in_piece[rows])
        features[rows] = windows.reshape(len(rows), -1)
    return features


def seed_posteriors(base_scores):
    """The base mix's soft labels: its standardised scores, tempered by the confidence rule.

    ARGS:
        base_scores (np.ndarray): [n, K] standardised

    RETURNS:
        [n, K] float64
    """
    return softmax_rows(base_scores / fit_temperature_confidence(base_scores, SEED_CONFIDENCE))


def similarity_matrices(features_by_subject, valid):
    """Each subject's z-scored row-by-row correlation matrix.

    ARGS:
        features_by_subject (list): [n, D] residual features
        valid (np.ndarray): [n, n] bool

    RETURNS:
        list of [n, n] float64
    """
    return [cf.zscore_matrix(cf.row_correlations(features), valid) for features in features_by_subject]


def vote(method, features_by_subject, groups, base_scores, params, matrices=None):
    """The other subjects' vote on every row, by one method.

    ARGS:
        method (str): one of METHODS
        features_by_subject (list): [n, D] residual features, OTHER subjects only
        groups (np.ndarray): [n] sentence ids
        base_scores (np.ndarray): [n, K] standardised base mix
        params (dict): alpha, neighbours, power (graph, snf); temperature (product)
        matrices (list or None): precomputed `similarity_matrices`

    RETURNS:
        vote [n, K] float64 (row-standardised), and a diagnostics dict
    """
    valid = cf.valid_pairs(groups)
    seeds = seed_posteriors(base_scores)
    if method in ("graph", "snf"):
        if method == "graph":
            matrices = matrices if matrices is not None else similarity_matrices(features_by_subject, valid)
            weights = cf.reliability_weights(matrices, valid)
            similarity = cf.weighted_mean_matrix(matrices, weights)
            info = {"weights": weights}
        else:
            similarity = cf.snf_graph(features_by_subject, valid, neighbours=params.get("snf_neighbours", 20),
                                      iterations=params.get("snf_iterations", 20))
            info = {}
        graph = cf.knn_graph(similarity, valid, params.get("neighbours", 10), params.get("power", 3.0))
        propagated = cf.propagate(graph, seeds, params["alpha"])
        info["similarity"] = similarity
        return cf.zscore_rows(cf.posterior_over_share(propagated)), info

    #P3 and P8a: per-subject cross-fitted LDA, then averaged or multiplied
    scores = [cf.cross_fitted_lda(features, seeds, groups) for features in features_by_subject]
    if method == "lda":
        return cf.zscore_rows(np.mean([cf.zscore_rows(score) for score in scores], axis=0)), {}
    if method == "product":
        temperature = params.get("temperature", 1.0 / np.sqrt(len(scores)))
        return cf.zscore_rows(temperature * np.sum(scores, axis=0)), {}
    raise ValueError(f"Unknown method {method!r}; known: {METHODS}")


def blend(base_scores, vote_scores, weight):
    """The base mix with the vote added at a weight, row-standardised.

    ARGS:
        base_scores (np.ndarray): [n, K]
        vote_scores (np.ndarray): [n, K]
        weight (float)

    RETURNS:
        [n, K] float64
    """
    return cf.zscore_rows(cf.zscore_rows(base_scores) + weight * cf.zscore_rows(vote_scores))
