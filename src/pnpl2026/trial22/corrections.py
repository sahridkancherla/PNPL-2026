"""Per-word corrections beyond the column z-score: change 4 of the Route to 73.

Every word counts equally in BAcc@10, so a word the scores favour on every
row wastes top-10 places. The holdout-fitted z-score (`hubnorm.fit_zscore`,
+0.9 on the board) centres and scales each word's column over all rows.
The doc's variants, each a function from standardised scores [N, 50] to
corrected scores whose rankings are what change:

    hubw     the z-score with each row weighted by 1 - p(word): a row that
             probably IS the word no longer pulls the word's own correction
    emsk     the holdout's word shares estimated (BCTS calibration on
             validation, then Saerens EM -- "hard to beat" at label shift,
             Alexandari et al.), then Sinkhorn balancing of the scores to
             those shares, never to equal ones
    hubpos   the z-score fitted separately for first, middle and last words,
             shrunk towards the overall fit
    quota    entropic transport: every row has 10 slots, every word gets
             slots in proportion to its estimated share; each row's top 10
             is read off the plan -- aimed at top-10 recall directly
    qbnorm   QB-Norm's inverted softmax (Bogolin et al. 2022) with a frozen
             bank of validation rows: no holdout statistics at all

Deterministic throughout.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

#the metric's k and the vocabulary
K = 10
N_WORDS = 50

#a probability this small is zero inside logarithms
FLOOR = 1e-12


def softmax_rows(logits):
    """Row-wise softmax.

    ARGS:
        logits (np.ndarray): [N, K]

    RETURNS:
        [N, K] float64
    """
    #axis=1 is the word axis; shifted for stability
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def weighted_zscore(scores, weights):
    """Each word's column centred and scaled with per-row weights.

    ARGS:
        scores (np.ndarray): [N, 50]
        weights (np.ndarray): [N, 50] non-negative

    RETURNS:
        [N, 50] corrected
    """
    #axis=0 is the row axis: one weighted mean and spread per word
    mass = np.maximum(weights.sum(axis=0), FLOOR)
    mean = (weights * scores).sum(axis=0) / mass
    spread = np.sqrt((weights * (scores - mean[None, :]) ** 2).sum(axis=0) / mass)
    return (scores - mean[None, :]) / np.where(spread > 0, spread, 1.0)[None, :]


def hubw(scores, temperature):
    """The z-score leaving out each word's likely rows.

    ARGS:
        scores (np.ndarray): [N, 50] standardised
        temperature (float): turns scores into probabilities

    RETURNS:
        [N, 50]
    """
    #a row's weight for a word: how unlikely it is to BE that word
    probabilities = softmax_rows(scores / temperature)
    return weighted_zscore(scores, 1.0 - probabilities)


def fit_bcts(scores, labels):
    """Bias-corrected temperature scaling: one temperature and one bias per word.

    ARGS:
        scores (np.ndarray): [n, 50] labelled rows (validation)
        labels (np.ndarray): [n]

    RETURNS:
        temperature (float), biases [50]
    """
    def loss(parameters):
        #NLL of softmax(scores / T + b), and its gradient
        temperature, biases = np.exp(parameters[0]), parameters[1:]
        logits = scores / temperature + biases[None, :]
        log_probabilities = logits - logsumexp(logits, axis=1, keepdims=True)
        probabilities = np.exp(log_probabilities)
        one_hot = np.zeros_like(probabilities)
        one_hot[np.arange(len(labels)), labels] = 1.0
        residual = (probabilities - one_hot) / len(labels)
        gradient_log_t = -np.sum(residual * scores / temperature)
        return -np.mean(log_probabilities[np.arange(len(labels)), labels]), np.concatenate([[gradient_log_t], residual.sum(axis=0)])

    solution = minimize(loss, np.zeros(N_WORDS + 1), jac=True, method="L-BFGS-B")
    return float(np.exp(solution.x[0])), solution.x[1:] - solution.x[1:].mean()


def em_shares(probabilities, source_shares, floor=0.005, n_iterations=200, tolerance=1e-8):
    """Saerens EM: the target set's class shares, from calibrated probabilities.

    ARGS:
        probabilities (np.ndarray): [N, 50] calibrated under `source_shares`
        source_shares (np.ndarray): [50] the calibration set's shares
        floor (float): the least share any word may take
        n_iterations (int), tolerance (float)

    RETURNS:
        shares [50], adjusted probabilities [N, 50]
    """
    source = np.maximum(source_shares, FLOOR)
    shares = source.copy()
    adjusted = probabilities
    for _ in range(n_iterations):
        #posteriors re-weighted by the current shares over the source's
        adjusted = probabilities * (shares / source)[None, :]
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        updated = np.maximum(adjusted.mean(axis=0), floor)
        updated = updated / updated.sum()
        if np.abs(updated - shares).max() < tolerance:
            shares = updated
            break
        shares = updated
    return shares, adjusted


def sinkhorn(matrix, row_mass, column_mass, n_iterations=500, tolerance=1e-9):
    """Scale rows and columns of a positive matrix to the given sums.

    ARGS:
        matrix (np.ndarray): [N, 50] positive
        row_mass (np.ndarray): [N]
        column_mass (np.ndarray): [50], summing to row_mass's total

    RETURNS:
        [N, 50] with those row and column sums
    """
    balanced = matrix.copy()
    for _ in range(n_iterations):
        #columns, then rows, each scaled to its target
        balanced *= (column_mass / np.maximum(balanced.sum(axis=0), FLOOR))[None, :]
        balanced *= (row_mass / np.maximum(balanced.sum(axis=1), FLOOR))[:, None]
        if np.abs(balanced.sum(axis=0) - column_mass).max() < tolerance * column_mass.sum():
            break
    return balanced


def emsk(scores, temperature, biases, source_shares, floor=0.005):
    """Estimated shares, then Sinkhorn balancing of the scores to them.

    ARGS:
        scores (np.ndarray): [N, 50] standardised
        temperature (float), biases ([50]): from `fit_bcts` on validation
        source_shares (np.ndarray): [50] the validation shares
        floor (float)

    RETURNS:
        corrected [N, 50] (log of the balanced matrix), shares [50]
    """
    calibrated = softmax_rows(scores / temperature + biases[None, :])
    shares, _ = em_shares(calibrated, source_shares, floor)

    #the scores' own distribution, balanced to the estimated shares
    balanced = sinkhorn(softmax_rows(scores / temperature), np.ones(len(scores)), shares * len(scores))
    return np.log(np.maximum(balanced, FLOOR)), shares


def position_groups(sequence_ids):
    """First, middle or last word of its sentence, from rows in sentence order.

    ARGS:
        sequence_ids (np.ndarray): [N] sentence of each row, rows of one
            sentence contiguous and in word order

    RETURNS:
        [N] int: 0 first (or alone), 1 middle, 2 last
    """
    groups = np.ones(len(sequence_ids), dtype=np.int64)
    starts = np.r_[True, sequence_ids[1:] != sequence_ids[:-1]]
    ends = np.r_[sequence_ids[1:] != sequence_ids[:-1], True]
    groups[ends] = 2
    #a one-word sentence counts as a first word
    groups[starts] = 0
    return groups


def hubpos(scores, groups, shrink_rows=50.0):
    """The z-score per position group, shrunk towards the overall one.

    ARGS:
        scores (np.ndarray): [N, 50] standardised
        groups (np.ndarray): [N] from `position_groups`
        shrink_rows (float): pseudo-rows of the overall fit in each group's

    RETURNS:
        [N, 50]
    """
    #axis=0: the overall per-word statistics
    overall_mean, overall_variance = scores.mean(axis=0), scores.var(axis=0)
    corrected = np.empty_like(scores)
    for group in np.unique(groups):
        rows = groups == group
        count = rows.sum()

        #the group's statistics, pulled towards the overall ones
        mean = (count * scores[rows].mean(axis=0) + shrink_rows * overall_mean) / (count + shrink_rows)
        variance = (count * scores[rows].var(axis=0) + shrink_rows * overall_variance) / (count + shrink_rows)
        corrected[rows] = (scores[rows] - mean[None, :]) / np.sqrt(np.maximum(variance, FLOOR))[None, :]
    return corrected


def quota(scores, shares, epsilon=0.5, n_slots=K):
    """Top-10 lists read off a transport plan with per-word capacities.

    ARGS:
        scores (np.ndarray): [N, 50] standardised
        shares (np.ndarray): [50] estimated shares; capacities are n_slots * N * share
        epsilon (float): entropic smoothing, in score units
        n_slots (int): 10

    RETURNS:
        [N, 50]: each row's 10 planned words first, in the original order
        among themselves, then the rest
    """
    #the plan: rows supply n_slots each, words take their capacity
    plan = sinkhorn(np.exp((scores - scores.max()) / epsilon), np.full(len(scores), float(n_slots)), n_slots * len(scores) * shares)
    chosen = np.argpartition(-plan, n_slots - 1, axis=1)[:, :n_slots]

    #the chosen words lifted above every other, their own order kept
    lifted = scores.copy()
    lift = scores.max() - scores.min() + 1.0
    np.put_along_axis(lifted, chosen, np.take_along_axis(scores, chosen, axis=1) + lift, axis=1)
    return lifted


def qbnorm(scores, bank, beta=1.0):
    """QB-Norm's inverted softmax with a frozen bank.

    ARGS:
        scores (np.ndarray): [N, 50] standardised
        bank (np.ndarray): [B, 50] standardised training-side rows
        beta (float): the inverse temperature

    RETURNS:
        [N, 50]: beta * s - log sum over the bank of exp(beta * bank score)
    """
    #axis=0 is the bank: one frozen offset per word
    offsets = logsumexp(beta * bank, axis=0)
    return beta * scores - offsets[None, :]
