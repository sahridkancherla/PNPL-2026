"""P5 of the 13 Oct research note: a skip-aware word-sequence prior, applied by forward-backward.

A holdout sentence's scored rows are vocabulary words in order, and about 30%
of neighbouring pairs skip one non-vocabulary word. Two priors give the next
vocabulary word from the last:

    lm    a general language model's own text (GPT-2 medium, sampled
          unconditionally -- never the stimulus), restricted to the 50
          words: counts of adjacent and of one-skip transitions, kept apart
    em    the evaluated rows' own posteriors: soft counts over neighbouring
          rows, ONE cross-fitted pass -- a chain's prior never comes from its
          own rows (the note: multi-round EM learns its own errors)

Each becomes pointwise mutual information -- a transition's lift over the
next word's base rate -- so the prior adds context, never frequency; the
balanced metric punishes a frequency prior. A pair's lm transition mixes the
adjacent and one-skip forms by that pair's skip probability, read from its
onset gap (`fit_skip_model`).

Rescoring (`rescore`): emissions are the mix's z-scores times a calibrated
temperature, transitions exp(weight * PMI) with the note's weight <= 0.1,
and each row's forward-backward posterior is its new score. Weight 0 gives
back the base ranking exactly; rows in no chain (lone words) keep it always.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.special import logsumexp

#curly apostrophes made straight, as the tutorial unifies them
APOSTROPHES = {"\u2019": "'", "\u2018": "'"}

#a spoken word in written text: letters, with at most one inner apostrophe
WORD_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")

#a chain never crosses a sentence end or a line break
SENTENCE_PATTERN = re.compile(r"[.!?]+|\n+")

#pseudo-counts shrinking each conditional toward the next word's base rate
KAPPA = 50.0

#onset gaps below this are clipped before the log
MIN_GAP_SECONDS = 1e-3


def normalise_text(text):
    """Lower case with straight apostrophes, as words are matched.

    ARGS:
        text (str)

    RETURNS:
        str
    """
    for curly, straight in APOSTROPHES.items():
        text = text.replace(curly, straight)
    return text.lower()


def count_transitions(texts, words):
    """Adjacent and one-skip transitions between vocabulary words in running text.

    A pair is adjacent when the two words follow each other, one-skip when
    exactly one other word lies between them -- the two forms a holdout
    sentence's neighbouring rows take. Sentences end at . ! ? and line breaks.

    ARGS:
        texts (iterable of str)
        words (list): the vocabulary, in scored order

    RETURNS:
        adjacent [50, 50], skip [50, 50], unigram [50] counts (float64), and
        the number of words read (int)
    """
    index = {normalise_text(word).strip(): position for position, word in enumerate(words)}
    n_words = len(words)
    adjacent = np.zeros((n_words, n_words))
    skip = np.zeros((n_words, n_words))
    unigram = np.zeros(n_words)
    total = 0
    for text in texts:
        for sentence in SENTENCE_PATTERN.split(normalise_text(text)):
            #each word's vocabulary index, -1 for any other word
            ids = np.array([index.get(token, -1) for token in WORD_PATTERN.findall(sentence)], dtype=np.int64)
            total += len(ids)
            if len(ids) == 0:
                continue
            unigram += np.bincount(ids[ids >= 0], minlength=n_words)

            #adjacent: two vocabulary words in a row
            last, following = ids[:-1], ids[1:]
            both = (last >= 0) & (following >= 0)
            np.add.at(adjacent, (last[both], following[both]), 1.0)

            #one skip: a vocabulary word, one other word, a vocabulary word
            if len(ids) >= 3:
                start, middle, end = ids[:-2], ids[1:-1], ids[2:]
                hop = (start >= 0) & (middle < 0) & (end >= 0)
                np.add.at(skip, (start[hop], end[hop]), 1.0)
    return adjacent, skip, unigram, total


def conditional(counts, kappa=KAPPA):
    """P(next | last) from transition counts, each row shrunk toward the base rate.

    ARGS:
        counts (np.ndarray): [50, 50], last word x next word
        kappa (float): pseudo-counts toward the next word's base rate

    RETURNS:
        conditional [50, 50] (rows sum to 1), base_rate [50]
    """
    #the next word's base rate, with one count each so no word is impossible
    base_rate = (counts.sum(axis=0) + 1.0) / (counts.sum() + counts.shape[1])
    totals = counts.sum(axis=1, keepdims=True)
    return (counts + kappa * base_rate[None, :]) / (totals + kappa), base_rate


def pmi(conditional_and_base):
    """A transition's log lift over the next word's base rate.

    ARGS:
        conditional_and_base (tuple): from `conditional`

    RETURNS:
        [50, 50]
    """
    probabilities, base_rate = conditional_and_base
    return np.log(probabilities) - np.log(base_rate)[None, :]


def mixed_pmi(adjacent, skip, skip_probability):
    """One pair's PMI, its adjacent and one-skip forms mixed by its skip probability.

    ARGS:
        adjacent (tuple): `conditional` of the adjacent counts
        skip (tuple): `conditional` of the one-skip counts
        skip_probability (float): this pair's chance of a skipped word

    RETURNS:
        [50, 50]
    """
    (adjacent_next, adjacent_base), (skip_next, skip_base) = adjacent, skip
    transition = (1.0 - skip_probability) * adjacent_next + skip_probability * skip_next
    base_rate = (1.0 - skip_probability) * adjacent_base + skip_probability * skip_base
    return np.log(transition) - np.log(base_rate)[None, :]


def zscore_rows(scores):
    """Each row z-scored across its words; a flat row stays flat.

    ARGS:
        scores (np.ndarray): [n, 50]

    RETURNS:
        [n, 50] float64
    """
    #axis=1 is the word axis
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def forward_backward(emissions, transitions):
    """Each row's posterior over words within one chain.

    ARGS:
        emissions (np.ndarray): [n, 50] log-scale evidence per row
        transitions (list): n - 1 arrays [50, 50], log-scale, last word x next

    RETURNS:
        [n, 50] log posteriors, each row normalised
    """
    n_rows = len(emissions)
    forward = np.zeros_like(emissions, dtype=np.float64)
    backward = np.zeros_like(emissions, dtype=np.float64)

    #forward messages, normalised at every step
    forward[0] = emissions[0] - logsumexp(emissions[0])
    for row in range(1, n_rows):
        step = logsumexp(forward[row - 1][:, None] + transitions[row - 1], axis=0) + emissions[row]
        forward[row] = step - logsumexp(step)

    #backward messages: everything after a row, given its word
    for row in range(n_rows - 2, -1, -1):
        step = logsumexp(transitions[row] + (emissions[row + 1] + backward[row + 1])[None, :], axis=1)
        backward[row] = step - logsumexp(step)

    #axis=1 is the word axis
    posterior = forward + backward
    return posterior - logsumexp(posterior, axis=1, keepdims=True)


def chain_order(groups, onsets):
    """Rows grouped into chains, each in onset order.

    ARGS:
        groups (np.ndarray): [n] chain id per row
        onsets (np.ndarray): [n] onset per row, any unit

    RETURNS:
        list of int arrays: each chain's rows, first word first
    """
    #one stable sort by chain, then onset; chains in order of first appearance
    _, first_seen = np.unique(groups, return_index=True)
    rank = np.empty(len(first_seen), dtype=np.int64)
    rank[np.argsort(first_seen, kind="stable")] = np.arange(len(first_seen))
    chain_rank = rank[np.unique(groups, return_inverse=True)[1]]
    order = np.lexsort((onsets, chain_rank))
    boundaries = np.flatnonzero(np.diff(chain_rank[order])) + 1
    return np.split(order, boundaries)


def chain_gap_features(onsets_s, chain):
    """One chain's neighbouring pairs' log gaps: raw, and against the chain's median.

    The second feature takes out the speaking rate, which differs by task.

    ARGS:
        onsets_s (np.ndarray): [n] onsets in seconds, every row
        chain (np.ndarray): the chain's rows, in order

    RETURNS:
        [len(chain) - 1, 2]
    """
    log_gaps = np.log(np.maximum(np.diff(onsets_s[chain]), MIN_GAP_SECONDS))
    if len(log_gaps) == 0:
        return np.zeros((0, 2))
    return np.stack([log_gaps, log_gaps - np.median(log_gaps)], axis=1)


def pair_gap_features(onsets_s, chains):
    """Every chain's pair features, stacked in chain order.

    ARGS:
        onsets_s (np.ndarray): [n] onsets in seconds
        chains (list): from `chain_order`

    RETURNS:
        [n_pairs, 2]
    """
    pieces = [chain_gap_features(onsets_s, chain) for chain in chains]
    return np.concatenate(pieces) if pieces else np.zeros((0, 2))


def pair_steps(table_index, chains):
    """How far apart in the word table each neighbouring pair is: 1 adjacent, 2 one skip.

    ARGS:
        table_index (np.ndarray): [n] each row's place in its recording's table
        chains (list)

    RETURNS:
        [n_pairs] int, in chain order
    """
    pieces = [np.diff(table_index[chain]) for chain in chains]
    return np.concatenate(pieces).astype(np.int64) if pieces else np.zeros(0, dtype=np.int64)


def fit_skip_model(features, is_skip):
    """A logistic model of 'this pair skips a word' from its gap features.

    ARGS:
        features (np.ndarray): [n_pairs, 2], from `pair_gap_features`
        is_skip (np.ndarray): [n_pairs] bool, from the word tables

    RETURNS:
        dict: coefficients, intercept and the skip share it was fitted on
    """
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0).fit(features, is_skip.astype(np.int64))
    return {"coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]),
            "skip_share": float(is_skip.mean())}


def skip_probability(features, skip_model):
    """Each pair's chance of a skipped word, from a fitted skip model.

    ARGS:
        features (np.ndarray): [n_pairs, 2]
        skip_model (dict): from `fit_skip_model`

    RETURNS:
        [n_pairs] in (0, 1)
    """
    logits = features @ np.asarray(skip_model["coefficients"]) + skip_model["intercept"]
    return 1.0 / (1.0 + np.exp(-logits))


def chain_skip_probabilities(onsets_s, chains, skip_model):
    """Each chain's pairs' skip probabilities, chain by chain.

    ARGS:
        onsets_s (np.ndarray): [n] onsets in seconds
        chains (list)
        skip_model (dict)

    RETURNS:
        list of [len(chain) - 1] arrays
    """
    return [skip_probability(chain_gap_features(onsets_s, chain), skip_model) for chain in chains]


def calibrate_temperature(scores, labels, grid=None):
    """The temperature whose softmax(t * z) has the lowest NLL on labelled rows.

    ARGS:
        scores (np.ndarray): [n, 50] z-scored
        labels (np.ndarray): [n] true columns
        grid (np.ndarray or None): temperatures tried

    RETURNS:
        float
    """
    grid = np.geomspace(0.25, 8.0, 41) if grid is None else grid
    losses = []
    for temperature in grid:
        #axis=1 is the word axis
        log_probabilities = temperature * scores - logsumexp(temperature * scores, axis=1, keepdims=True)
        losses.append(-log_probabilities[np.arange(len(labels)), labels].mean())
    return float(grid[int(np.argmin(losses))])


def chain_folds(n_chains, n_folds, seed=0):
    """Each chain's cross-fitting fold, drawn once.

    ARGS:
        n_chains (int), n_folds (int), seed (int)

    RETURNS:
        [n_chains] int
    """
    return np.random.default_rng(seed).permutation(n_chains) % n_folds


def em_conditionals_by_fold(posteriors, chains, folds, chain_skips, kappa=KAPPA):
    """The em prior: each fold's adjacent and one-skip conditionals from the OTHER folds' rows.

    One pass, so no chain's prior is shaped by its own rows' posteriors. Each
    pair's soft count -- the outer product of its two rows' posteriors -- is
    split between the adjacent and one-skip counts by its skip probability.

    ARGS:
        posteriors (np.ndarray): [n, 50] the base's calibrated posteriors
        chains (list): from `chain_order`
        folds (np.ndarray): [n_chains] from `chain_folds`
        chain_skips (list): from `chain_skip_probabilities`
        kappa (float)

    RETURNS:
        dict: fold -> (adjacent, skip), each a `conditional` tuple
    """
    n_words = posteriors.shape[1]
    adjacent = {fold: np.zeros((n_words, n_words)) for fold in np.unique(folds)}
    skip = {fold: np.zeros((n_words, n_words)) for fold in np.unique(folds)}
    for chain, fold, skips in zip(chains, folds, chain_skips):
        if len(chain) < 2:
            continue
        #[pairs, 50] each: the last row's posterior weighted by each form's share
        last, following = posteriors[chain[:-1]], posteriors[chain[1:]]
        adjacent[fold] += (last * (1.0 - skips)[:, None]).T @ following
        skip[fold] += (last * skips[:, None]).T @ following

    #each fold's conditionals from every other fold's counts
    all_adjacent, all_skip = sum(adjacent.values()), sum(skip.values())
    return {fold: (conditional(all_adjacent - adjacent[fold], kappa), conditional(all_skip - skip[fold], kappa))
            for fold in adjacent}


def rescore(base_scores, chains, temperature, weight, pair_pmi):
    """Every chain's rows rescored by forward-backward; other rows keep the base.

    ARGS:
        base_scores (np.ndarray): [n, 50]
        chains (list): from `chain_order`
        temperature (float): emissions are temperature * z(base)
        weight (float): transitions are exp(weight * PMI); 0 is the base
        pair_pmi (callable): (chain index, pair position) -> [50, 50] PMI

    RETURNS:
        [n, 50] z-scored
    """
    rescored = zscore_rows(base_scores.astype(np.float64))
    for chain_index, chain in enumerate(chains):
        if len(chain) < 2 or weight == 0:
            continue
        emissions = temperature * rescored[chain]
        transitions = [weight * pair_pmi(chain_index, position) for position in range(len(chain) - 1)]
        rescored[chain] = zscore_rows(forward_backward(emissions, transitions))
    return rescored
