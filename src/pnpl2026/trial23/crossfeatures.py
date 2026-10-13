"""Cross-subject information without the timing echo: the 13 Oct research note's P1, P3 and P8a.

Trial 22's cross-subject votes echoed subject 0's picks (row-specific
r = 0.78): every subject's file shares the sentence timing, and a decoder fed
another brain re-votes that timing. The note's remedy, step by step:

    windows     -0.04 to +0.46 s around each word's own onset (25 samples at
                50 Hz), 20 Hz low-pass, in a shared MCCA space; the mean of
                the first 75 ms subtracted
    residuals   per subject, every feature ridge-regressed on a timing
                design identical in all files -- position, reversed
                position, sentence length, splines of the log gaps before
                and after, a silence flag -- and only the residual kept
    no pairs within a sentence: neighbouring windows overlap in time

and three ways to turn the other subjects' residuals into a vote:

    P1 graph    each subject's row-by-row correlation matrix, z-scored,
                averaged over subjects weighted by their agreement with the
                rest; the top-k neighbours, symmetrized and sharpened to a
                power (ZLaP); label propagation F = (1-a)(I - aS)^-1 Y from
                the mix's posteriors Y. SNF (Wang et al. 2014) fuses the
                subjects' graphs by cross-diffusion instead of averaging
    P3 LDA      per subject, 5 folds grouped by sentence: soft class means
                from the mix's posteriors on 4 folds, a Ledoit-Wolf shared
                covariance, LDA log-likelihood on the fifth; the subjects'
                z-scored scores averaged
    P8a product the same log-likelihoods SUMMED over subjects at a
                temperature -- a product of per-subject likelihoods

One pass only: a second round would feed each row's pseudo-label back into
it. Diagnostics: split-half reliability of the graph, the echo with subject
0's mix and with a timing-only voter, and (with labels) neighbour purity.
Deterministic throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import butter, filtfilt
from sklearn.covariance import LedoitWolf

#the features' rate and the low-pass the note asks for
RATE = 50.0
LOWPASS_HZ = 20.0

#the window: two samples before the onset (-0.04 s) to 22 after (+0.44 s)
WINDOW_START = -2
WINDOW_LENGTH = 25

#the baseline: the window's first four samples, about 75 ms
BASELINE_SAMPLES = 4

#a gap this long before a word counts as silence
SILENCE_SECONDS = 0.5

#a probability this small is zero inside logarithms
FLOOR = 1e-12


def lowpass(signal, rate=RATE, cutoff=LOWPASS_HZ, order=4):
    """Zero-phase Butterworth low-pass along time.

    ARGS:
        signal (np.ndarray): [..., n_times]
        rate (float), cutoff (float), order (int)

    RETURNS:
        [..., n_times] float64
    """
    numerator, denominator = butter(order, cutoff / (rate / 2.0))
    #filtfilt needs a few times the filter's length; short pieces are left as they are
    if signal.shape[-1] <= 3 * max(len(numerator), len(denominator)):
        return signal.astype(np.float64)
    return filtfilt(numerator, denominator, signal.astype(np.float64), axis=-1)


def onset_windows(series, onsets):
    """Each word's window of a shared-component series, baseline removed.

    ARGS:
        series (np.ndarray): [C, n_times] one epoch or recording, 50 Hz
        onsets (np.ndarray): [n_words] onset samples within it

    RETURNS:
        [n_words, C, WINDOW_LENGTH] float64; samples before the start are the
        first sample repeated
    """
    padded = np.pad(series, ((0, 0), (-WINDOW_START, WINDOW_LENGTH)), mode="edge")
    windows = np.stack([padded[:, onset : onset + WINDOW_LENGTH] for onset in np.asarray(onsets, dtype=np.int64)])

    #axis=2 is time: each component's mean over the first ~75 ms, removed
    return windows - windows[:, :, :BASELINE_SAMPLES].mean(axis=2, keepdims=True)


def spline_basis(values, n_knots=5, degree=3):
    """A cubic B-spline basis of a variable, knots at its quantiles.

    ARGS:
        values (np.ndarray): [n] with NaN where undefined
        n_knots (int): interior knots
        degree (int)

    RETURNS:
        [n, n_knots + degree + 1] float64, zero where `values` is NaN
    """
    finite = np.isfinite(values)
    basis = np.zeros((len(values), n_knots + degree + 1))
    if finite.sum() < n_knots + 2:
        return basis
    low, high = np.min(values[finite]), np.max(values[finite])
    interior = np.quantile(values[finite], np.linspace(0, 1, n_knots + 2)[1:-1])
    knots = np.r_[[low] * (degree + 1), interior, [high] * (degree + 1)]

    #the basis at the clipped values; a constant variable gets the first column
    clipped = np.clip(values[finite], low, high - 1e-9 * max(high - low, 1.0))
    if high > low:
        basis[finite] = BSpline.design_matrix(clipped, knots, degree).toarray()
    else:
        basis[finite, 0] = 1.0
    return basis


def sentence_structure(groups, onsets_s):
    """Each row's position, reversed position, sentence length and gaps.

    ARGS:
        groups (np.ndarray): [n] sentence of each row; rows of a sentence
            contiguous and in word order
        onsets_s (np.ndarray): [n] onsets in seconds

    RETURNS:
        dict of [n] arrays: position, reversed, length, gap_before (NaN for
        a first word), gap_after (NaN for a last word)
    """
    n_rows = len(groups)
    position, reversed_position, length = np.zeros(n_rows, int), np.zeros(n_rows, int), np.zeros(n_rows, int)
    gap_before, gap_after = np.full(n_rows, np.nan), np.full(n_rows, np.nan)
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], n_rows]
    for start, end in zip(starts, ends):
        #one sentence's rows: their places and the gaps between them
        size = end - start
        position[start:end] = np.arange(size)
        reversed_position[start:end] = np.arange(size)[::-1]
        length[start:end] = size
        gaps = np.diff(np.asarray(onsets_s[start:end], dtype=np.float64))
        gap_before[start + 1 : end] = gaps
        gap_after[start : end - 1] = gaps
    return {"position": position, "reversed": reversed_position, "length": length,
            "gap_before": gap_before, "gap_after": gap_after}


def timing_design(groups, onsets_s, max_position=8, n_knots=5):
    """The timing design every file shares: what a feature must not be explained by.

    ARGS:
        groups (np.ndarray): [n] sentence ids, rows contiguous per sentence
        onsets_s (np.ndarray): [n] onsets in seconds
        max_position (int): positions and lengths beyond are pooled
        n_knots (int): interior knots of each gap spline

    RETURNS:
        [n, n_columns] float64 with an intercept
    """
    structure = sentence_structure(groups, onsets_s)
    columns = [np.ones((len(groups), 1))]

    #one-hot position, reversed position and length, the tail pooled
    for name in ("position", "reversed"):
        values = np.minimum(structure[name], max_position - 1)
        columns.append(np.eye(max_position)[values])
    columns.append(np.eye(max_position + 1)[np.minimum(structure["length"], max_position)])

    #the gaps on a log scale, as splines; a missing gap has its own flag
    for name in ("gap_before", "gap_after"):
        log_gap = np.log(np.maximum(structure[name], 1e-3))
        columns.append(spline_basis(log_gap, n_knots))
        columns.append((~np.isfinite(structure[name]))[:, None].astype(np.float64))

    #silence before the word: the sentence's first word, or a long gap
    silence = ~np.isfinite(structure["gap_before"]) | (np.nan_to_num(structure["gap_before"], nan=0.0) > SILENCE_SECONDS)
    columns.append(silence[:, None].astype(np.float64))
    return np.concatenate(columns, axis=1)


def residual_projector(design, ridge=1e-3):
    """The matrix that keeps what the design cannot explain: I - D (D'D + rI)^-1 D'.

    ARGS:
        design (np.ndarray): [n, p]
        ridge (float): relative to the design's mean squared column norm

    RETURNS:
        [n, n] float64
    """
    gram = design.T @ design
    penalty = ridge * np.trace(gram) / max(gram.shape[0], 1)
    hat = design @ np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), design.T)
    return np.eye(len(design)) - hat


def residualize(features, projector):
    """One subject's features with the timing part removed.

    ARGS:
        features (np.ndarray): [n, D] flattened windows
        projector (np.ndarray): [n, n] from `residual_projector`

    RETURNS:
        [n, D] float64
    """
    return projector @ features


def valid_pairs(groups):
    """Which row pairs may be linked: different sentences only.

    ARGS:
        groups (np.ndarray): [n] sentence ids

    RETURNS:
        [n, n] bool
    """
    return groups[:, None] != groups[None, :]


def row_correlations(features):
    """Every pair of rows' correlation across features.

    ARGS:
        features (np.ndarray): [n, D]

    RETURNS:
        [n, n] float64
    """
    #axis=1: each row centred and scaled across its features
    centred = features - features.mean(axis=1, keepdims=True)
    unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), FLOOR)
    return unit @ unit.T


def zscore_matrix(matrix, valid):
    """A similarity matrix z-scored over its valid entries; invalid entries 0.

    ARGS:
        matrix (np.ndarray): [n, n]
        valid (np.ndarray): [n, n] bool

    RETURNS:
        [n, n] float64
    """
    values = matrix[valid]
    spread = values.std()
    out = np.zeros_like(matrix, dtype=np.float64)
    out[valid] = (values - values.mean()) / (spread if spread > 0 else 1.0)
    return out


def reliability_weights(matrices, valid):
    """Each subject's agreement with the rest: correlation of its matrix with the others' mean.

    ARGS:
        matrices (list): [n, n] z-scored similarity per subject
        valid (np.ndarray): [n, n] bool

    RETURNS:
        [n_subjects] float64, floored at 0
    """
    flat = np.stack([matrix[valid] for matrix in matrices])
    total = flat.sum(axis=0)
    weights = []
    for index in range(len(flat)):
        #leave-one-out: this subject against the mean of all the others
        others = (total - flat[index]) / max(len(flat) - 1, 1)
        weights.append(np.corrcoef(flat[index], others)[0, 1] if len(flat) > 1 else 1.0)
    return np.maximum(np.nan_to_num(np.array(weights)), 0.0)


def weighted_mean_matrix(matrices, weights):
    """The subjects' similarity matrices averaged by weight.

    ARGS:
        matrices (list): [n, n] each
        weights (np.ndarray): [n_subjects]

    RETURNS:
        [n, n] float64
    """
    total = weights.sum()
    if total <= 0:
        weights, total = np.ones(len(matrices)), float(len(matrices))
    return np.tensordot(weights / total, np.stack(matrices), axes=1)


def knn_graph(similarity, valid, neighbours=10, power=3.0):
    """Top-k neighbours, symmetrized, weights sharpened to a power (ZLaP).

    ARGS:
        similarity (np.ndarray): [n, n]
        valid (np.ndarray): [n, n] bool
        neighbours (int), power (float)

    RETURNS:
        [n, n] float64, non-negative, symmetric, zero off the kept pairs
    """
    masked = np.where(valid, similarity, -np.inf)
    keep = min(neighbours, int(valid.sum(axis=1).min()))
    graph = np.zeros_like(similarity, dtype=np.float64)
    if keep <= 0:
        return graph
    nearest = np.argpartition(-masked, keep - 1, axis=1)[:, :keep]
    rows = np.repeat(np.arange(len(similarity)), keep)
    weights = np.maximum(masked[rows, nearest.ravel()], 0.0) ** power
    graph[rows, nearest.ravel()] = weights
    return 0.5 * (graph + graph.T)


def propagate(graph, seeds, alpha):
    """Label propagation: F = (1 - a)(I - a S)^-1 Y with S the normalized graph.

    ARGS:
        graph (np.ndarray): [n, n] non-negative symmetric
        seeds (np.ndarray): [n, K] the mix's posteriors
        alpha (float): in [0, 1)

    RETURNS:
        [n, K] float64
    """
    degree = graph.sum(axis=1)
    scale = 1.0 / np.sqrt(np.maximum(degree, FLOOR))
    normalized = scale[:, None] * graph * scale[None, :]
    return (1.0 - alpha) * np.linalg.solve(np.eye(len(graph)) - alpha * normalized, seeds)


def posterior_over_share(scores):
    """Log of each word's propagated mass over its mean mass: balanced accuracy's ranking.

    ARGS:
        scores (np.ndarray): [n, K] non-negative

    RETURNS:
        [n, K] float64
    """
    #axis=0 is the row axis: a word's mean share over the rows, divided out
    share = np.maximum(scores.mean(axis=0, keepdims=True), FLOOR)
    return np.log(np.maximum(scores, FLOOR)) - np.log(share)


def snf_graph(features_by_subject, valid, neighbours=20, iterations=20, mu=0.5):
    """Similarity Network Fusion (Wang et al. 2014) across subjects' residual features.

    ARGS:
        features_by_subject (list): [n, D] each
        valid (np.ndarray): [n, n] bool
        neighbours (int): K, the local kNN of each view
        iterations (int): t, rounds of cross-diffusion
        mu (float): the kernel's scale

    RETURNS:
        [n, n] float64 fused similarity, zero off valid pairs
    """
    def affinity(features):
        #the scaled exponential kernel on row-normalized features
        centred = features - features.mean(axis=1, keepdims=True)
        unit = centred / np.maximum(np.linalg.norm(centred, axis=1, keepdims=True), FLOOR)
        distance = np.maximum(2.0 - 2.0 * unit @ unit.T, 0.0)
        sorted_distance = np.sort(np.where(valid, distance, np.inf), axis=1)[:, :neighbours]
        local = np.where(np.isfinite(sorted_distance), sorted_distance, 0.0).mean(axis=1)
        sigma = mu * (local[:, None] + local[None, :] + distance) / 3.0
        kernel = np.exp(-distance / np.maximum(sigma, FLOOR)) * valid
        return kernel

    def full_normalize(kernel):
        #off-diagonal rows sum to a half, the diagonal holds the other half
        off = kernel - np.diag(np.diag(kernel))
        rows = np.maximum(off.sum(axis=1, keepdims=True), FLOOR)
        normalized = off / (2.0 * rows)
        np.fill_diagonal(normalized, 0.5)
        return normalized

    def local_normalize(kernel):
        #each row keeps its K nearest, renormalized
        keep = np.argpartition(-kernel, neighbours - 1, axis=1)[:, :neighbours]
        local = np.zeros_like(kernel)
        rows = np.repeat(np.arange(len(kernel)), neighbours)
        local[rows, keep.ravel()] = kernel[rows, keep.ravel()]
        return local / np.maximum(local.sum(axis=1, keepdims=True), FLOOR)

    kernels = [affinity(features) for features in features_by_subject]
    full = [full_normalize(kernel) for kernel in kernels]
    local = [local_normalize(kernel) for kernel in kernels]
    for _ in range(iterations):
        #each view diffused through the mean of all the others
        total = np.sum(full, axis=0)
        full = [full_normalize(local[index] @ ((total - full[index]) / max(len(full) - 1, 1)) @ local[index].T)
                for index in range(len(full))]
    fused = np.mean(full, axis=0)
    return np.where(valid, 0.5 * (fused + fused.T), 0.0)


def sentence_folds(groups, n_folds=5):
    """Each row's fold, whole sentences together, deterministic.

    ARGS:
        groups (np.ndarray): [n] sentence ids
        n_folds (int)

    RETURNS:
        [n] int in 0..n_folds-1
    """
    #sentences in order of first appearance, dealt round the folds
    unique = list(dict.fromkeys(groups.tolist()))
    fold_of = {sentence: index % n_folds for index, sentence in enumerate(unique)}
    return np.array([fold_of[sentence] for sentence in groups.tolist()])


def cross_fitted_lda(features, posteriors, groups, n_folds=5, pca_dims=100, pseudo_count=1.0):
    """One subject's LDA log-likelihoods, each fold scored by prototypes fitted on the others.

    ARGS:
        features (np.ndarray): [n, D] residual features
        posteriors (np.ndarray): [n, K] the mix's soft labels
        groups (np.ndarray): [n] sentence ids, for the folds
        n_folds (int), pca_dims (int)
        pseudo_count (float): rows of the overall mean in every class mean,
            so a word with little soft mass is not a noisy point

    RETURNS:
        [n, K] float64 log-likelihoods, up to a per-row constant
    """
    #a within-subject PCA: the covariance is estimated in far fewer dimensions
    centred = features - features.mean(axis=0, keepdims=True)
    _, singular, components = np.linalg.svd(centred, full_matrices=False)
    keep = min(pca_dims, components.shape[0])
    reduced = centred @ components[:keep].T

    folds = sentence_folds(groups, n_folds)
    scores = np.zeros((len(features), posteriors.shape[1]))
    for fold in range(n_folds):
        train, test = folds != fold, folds == fold
        if not test.any() or not train.any():
            continue
        weights = posteriors[train]
        overall = reduced[train].mean(axis=0)

        #soft class means from the OTHER folds' rows only, shrunk to the overall mean
        mass = weights.sum(axis=0)
        means = (weights.T @ reduced[train] + pseudo_count * overall[None, :]) / (mass + pseudo_count)[:, None]

        #the shared within-class covariance, Ledoit-Wolf shrunk
        residual = reduced[train] - weights @ means
        precision = LedoitWolf().fit(residual).precision_
        projected = means @ precision
        scores[test] = reduced[test] @ projected.T - 0.5 * np.sum(projected * means, axis=1)[None, :]
    return scores


def zscore_rows(scores):
    """Each row centred and scaled across its words.

    ARGS:
        scores (np.ndarray): [n, K]

    RETURNS:
        [n, K] float64
    """
    #axis=1 is the word axis; a flat row stays flat
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def row_specific_correlation(first, second):
    """Agreement on each row's own part: every word's mean over rows removed first.

    ARGS:
        first, second (np.ndarray): [n, K]

    RETURNS:
        float: mean row correlation of the column-centred tables
    """
    #axis=0 is the row axis: each word's mean removed, then rows compared
    a = zscore_rows(first - first.mean(axis=0, keepdims=True))
    b = zscore_rows(second - second.mean(axis=0, keepdims=True))
    return float(np.mean(np.sum(a * b, axis=1) / a.shape[1]))


def neighbour_overlap(first, second, valid, neighbours=10):
    """How many of each row's top-k neighbours two similarity matrices share, against chance.

    ARGS:
        first, second (np.ndarray): [n, n]
        valid (np.ndarray): [n, n] bool
        neighbours (int)

    RETURNS:
        overlap (float, share of the k shared), chance (float)
    """
    keep = min(neighbours, int(valid.sum(axis=1).min()))
    top_first = np.argpartition(-np.where(valid, first, -np.inf), keep - 1, axis=1)[:, :keep]
    top_second = np.argpartition(-np.where(valid, second, -np.inf), keep - 1, axis=1)[:, :keep]
    overlap = np.mean([len(set(a) & set(b)) / keep for a, b in zip(top_first, top_second)])
    return float(overlap), float(keep / np.mean(valid.sum(axis=1)))


def split_half_reliability(matrices, valid, seed=0, neighbours=10):
    """The subjects split in two halves: how alike are the halves' graphs?

    ARGS:
        matrices (list): [n, n] z-scored similarity per subject
        valid (np.ndarray): [n, n] bool
        seed (int), neighbours (int)

    RETURNS:
        dict: correlation of the halves' mean matrices, its Spearman-Brown
        full-set value, neighbour overlap and its chance level
    """
    order = np.random.default_rng(seed).permutation(len(matrices))
    half = len(order) // 2
    first = np.mean([matrices[index] for index in order[:half]], axis=0)
    second = np.mean([matrices[index] for index in order[half:]], axis=0)
    correlation = float(np.corrcoef(first[valid], second[valid])[0, 1])
    overlap, chance = neighbour_overlap(first, second, valid, neighbours)
    return {"half_correlation": correlation, "spearman_brown": 2 * correlation / (1 + correlation) if correlation > -1 else float("nan"),
            "neighbour_overlap": overlap, "chance": chance}


def neighbour_purity(similarity, valid, labels, neighbours=10):
    """With labels: the share of each row's top-k neighbours that are the same word, and the base rate.

    ARGS:
        similarity (np.ndarray): [n, n]
        valid (np.ndarray): [n, n] bool
        labels (np.ndarray): [n] word of each row
        neighbours (int)

    RETURNS:
        purity (float), base rate (float): what random valid neighbours give
    """
    keep = min(neighbours, int(valid.sum(axis=1).min()))
    nearest = np.argpartition(-np.where(valid, similarity, -np.inf), keep - 1, axis=1)[:, :keep]
    purity = float(np.mean(labels[nearest] == labels[:, None]))
    same = (labels[:, None] == labels[None, :]) & valid
    base = float(np.mean(same.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)))
    return purity, base
