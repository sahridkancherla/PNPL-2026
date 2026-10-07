"""Hub correction: frozen per-word adjustments fitted on a bank of validation queries.

Every model retrieves its answer as the nearest of the 50 words' t5 vectors.
A few words sit near the centre of that space -- the HUBS -- and land in the
top 10 of most rows whatever the true word, crowding out the rest; the hub
check found one word in nearly every simulated uncertain top 10 and others
in none. Balanced top-10 accuracy pays for every slot a hub wastes.

Querybank normalisation (Bogolin et al. 2022, arXiv 2112.12777) corrects
hubs with statistics from a bank of TRAINING-SIDE queries, never from other
test queries. Its inverted softmax, in log space, subtracts a fixed amount
per candidate word -- so for 50 fixed words the whole correction is one
frozen number (or two) per word. Two ways of fitting it, both label-free on
the bank:

    zscore    each word's scores centred and scaled by that word's own mean
              and spread over the bank: the frozen version of column
              normalisation
    balance   per-word offsets, found by iteration, that give every word the
              fair top-10 share on the bank (10 of 50 = 20% of rows)

The bank is a validation partition's predictions: the holdout principle
(CLAUDE.md) allows anything fitted on validation data and then frozen, and
each holdout row is then corrected on its own.

Every score array here is in the space `cli.ensemble` reads a CSV in: each
row's log-probabilities (or logits) z-scored across the 50 words.
"""

from __future__ import annotations

import numpy as np

#BAcc@10: a word's fair share of top-10 places is 10 of 50
K = 10


def top_k_occupancy(scores, k=K):
    """How often each word reaches a row's top k.

    ARGS:
        scores (np.ndarray): [n_rows, n_words]
        k (int): 10

    RETURNS:
        [n_words] float64, the share of rows; a fair share is k / n_words
    """
    #axis=1 is the word axis: each row's k best words, in any order
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    return np.bincount(top.ravel(), minlength=scores.shape[1]) / len(scores)


def fit_zscore(bank):
    """Each word's mean and spread over the bank.

    ARGS:
        bank (np.ndarray): [n_bank, n_words] standardised scores

    RETURNS:
        dict: kind "zscore", mean [n_words], spread [n_words]
    """
    #axis=0 is the bank axis: one statistic per word, over every query
    mean = bank.mean(axis=0)
    spread = bank.std(axis=0)
    return {"kind": "zscore", "mean": mean, "spread": np.where(spread > 0, spread, 1.0)}


def fit_balance(bank, k=K, n_iterations=3000, step=0.5, tolerance=0.004):
    """Per-word offsets that give every word the fair top-k share on the bank.

    Subtracting a word's offset from every row's score for it moves that word
    down every ranking alike. Each iteration raises the offset of a word in
    more than its share of top-k lists and lowers one in fewer, by an amount
    proportional to the excess, with a step that shrinks so the discrete
    shares settle.

    ARGS:
        bank (np.ndarray): [n_bank, n_words] standardised scores
        k (int): 10
        n_iterations (int): cap
        step (float): first step, in score units per unit of relative excess
        tolerance (float): stop once every share is this close to fair

    RETURNS:
        dict: kind "balance", offsets [n_words] with mean 0, and the largest
        share error left on the bank
    """
    n_words = bank.shape[1]
    target = k / n_words
    offsets = np.zeros(n_words)
    error = np.inf
    for iteration in range(n_iterations):
        #each word's share of top-k places with the current offsets
        excess = top_k_occupancy(bank - offsets[None, :], k) - target
        error = float(np.abs(excess).max())
        if error < tolerance:
            break

        #proportional, shrinking steps; only differences between offsets rank
        offsets += step / (1.0 + iteration / 300.0) * excess / target
        offsets -= offsets.mean()
    return {"kind": "balance", "offsets": offsets, "share_error": error}


def apply_correction(scores, correction, strength=1.0):
    """One frozen correction applied to every row on its own.

    ARGS:
        scores (np.ndarray): [n_rows, n_words] standardised scores
        correction (dict): from `fit_zscore` or `fit_balance`
        strength (float): for "balance", the share of each offset applied;
            ignored by "zscore"

    RETURNS:
        [n_rows, n_words] float64 - corrected scores; rankings are what change
    """
    #[1, n_words] statistics against [n_rows, n_words] scores, word by word
    if correction["kind"] == "zscore":
        return (scores - correction["mean"][None, :]) / correction["spread"][None, :]
    if correction["kind"] == "balance":
        return scores - strength * correction["offsets"][None, :]
    raise ValueError(f"Unknown correction kind {correction['kind']!r}")
