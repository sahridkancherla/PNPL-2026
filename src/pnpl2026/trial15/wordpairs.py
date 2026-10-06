"""A word-pair model of the 50 scored words, and the fusion that uses it.

Every holdout "sentence" is a run of CONSECUTIVE words that are all in the
50-word list (lengths 4-8 almost always). No model trained so far knows that:
they trained on narrative, where most neighbours are unscored words. If word 3
of a run is "the", word 4 can only be one of the few list words that follow
"the" directly.

    pairs[a, b]    how often scored word b directly follows scored word a in
                   one sentence of subject 0's transcripts (the cache's
                   `words_<key>.npz` tables)
    transitions    pairs, each row smoothed toward the overall next-word
                   distribution, as P(next = b | this = a)
    stationary     the chain's stationary distribution: the prior of a word at
                   any position of a long run

`fuse_run` combines a submission's scores with the chain along one run by
forward-backward. Each word's score becomes

    own scores / temperature  +  strength * log P(the other words' MEG | this word)

-- its own evidence plus what the neighbours' evidence says about it, as a
likelihood ratio. The prior is divided out (log forward - log stationary), so
frequent words gain nothing for being frequent: the metric weights every word
equally. With the neighbours' evidence flat, the context term is constant and
the ranking is the input's.

`temperature` turns z-scored log-probabilities into calibrated
log-likelihoods; `cli.fuse_check` fits it on the validation partition.
`strength` 1 is exact Bayes under the chain; below 1 trusts it less.
"""

from __future__ import annotations

import numpy as np

#the scored vocabulary's size; columns are load_vocabulary("primary") order
N_WORDS = 50

#onsets further apart than this (at 50 Hz) do not pair when a task has no
#sentence ids; 99.4% of the holdout's gaps are under 0.8 s
MAX_GAP_SAMPLES = 40


def count_word_pairs(tables, n_words=N_WORDS, max_gap_samples=MAX_GAP_SAMPLES):
    """Count directly adjacent scored-word pairs, and their onset gaps.

    Two words are adjacent when they are consecutive rows of one recording's
    word table and share a sentence id; rows with no sentence id (-1) pair
    when their onsets are at most `max_gap_samples` apart.

    ARGS:
        tables (iterable of dict): word tables with `primary` [n] (column or
            -1), `sentence` [n] and `onset_sample` [n] at 50 Hz, onset order
        n_words (int): 50
        max_gap_samples (int): the pairing limit where sentences are unknown

    RETURNS:
        dict:
            pairs : [n_words, n_words] float64, row = first word
            unigram : [n_words] float64, every scored token
            adjacent_gaps : [n_pairs] int64, onset gaps of the counted pairs
            skip_gaps : [n] int64, gaps across ONE unscored word between two
                scored ones, for comparison with the holdout's gaps
    """
    pairs = np.zeros((n_words, n_words), dtype=np.float64)
    unigram = np.zeros(n_words, dtype=np.float64)
    adjacent_gaps, skip_gaps = [], []

    for table in tables:
        primary = np.asarray(table["primary"]).astype(np.int64)
        sentence = np.asarray(table["sentence"]).astype(np.int64)
        onset = np.asarray(table["onset_sample"]).astype(np.int64)

        #every scored token, whatever its neighbours
        unigram += np.bincount(primary[primary >= 0], minlength=n_words)
        if len(primary) < 2:
            continue

        #rows i and i+1 belong together: one sentence, or close in time
        gap = onset[1:] - onset[:-1]
        same_sentence = (sentence[1:] == sentence[:-1]) & (sentence[1:] >= 0)
        no_sentence = (sentence[1:] < 0) & (sentence[:-1] < 0) & (gap <= max_gap_samples)
        together = same_sentence | no_sentence

        #both scored: one count in the first word's row, the second's column
        first, second = primary[:-1], primary[1:]
        counted = together & (first >= 0) & (second >= 0)
        np.add.at(pairs, (first[counted], second[counted]), 1.0)
        adjacent_gaps.append(gap[counted])

        #scored, unscored, scored, all together: what a skipped word looks like
        if len(primary) >= 3:
            spans = together[:-1] & together[1:]
            skipped = spans & (primary[:-2] >= 0) & (primary[1:-1] < 0) & (primary[2:] >= 0)
            skip_gaps.append((onset[2:] - onset[:-2])[skipped])

    #empty arrays rather than a failed concatenate when nothing was found
    return {
        "pairs": pairs,
        "unigram": unigram,
        "adjacent_gaps": np.concatenate(adjacent_gaps) if adjacent_gaps else np.zeros(0, dtype=np.int64),
        "skip_gaps": np.concatenate(skip_gaps) if skip_gaps else np.zeros(0, dtype=np.int64),
    }


def transition_matrix(pairs, smoothing):
    """P(next scored word | this scored word), each row smoothed.

    A row with few counts leans on the overall next-word distribution (the
    column totals, plus one each): (counts + smoothing * base) / (total +
    smoothing).

    ARGS:
        pairs (np.ndarray): [n_words, n_words] counts
        smoothing (float): pseudo-counts per row, > 0

    RETURNS:
        [n_words, n_words] float64, every row summing to 1, every entry > 0
    """
    if smoothing <= 0:
        raise ValueError(f"smoothing must be > 0, got {smoothing}")

    #axis=0 sums over first words: how often each word is the SECOND of a pair
    base = pairs.sum(axis=0) + 1.0
    base = base / base.sum(axis=0)

    #axis=1 is the next word: each row becomes a distribution
    smoothed = pairs + smoothing * base[None, :]
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def pair_log_likelihood(transitions, pairs):
    """Mean log-probability of held-out pairs under a transition matrix.

    ARGS:
        transitions (np.ndarray): [n_words, n_words]
        pairs (np.ndarray): [n_words, n_words] held-out counts

    RETURNS:
        float: nats per pair; nan when there are no pairs
    """
    total = pairs.sum()
    if total == 0:
        return float("nan")
    return float((pairs * np.log(transitions)).sum() / total)


def choose_smoothing(train_pairs, held_out_pairs, candidates=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0)):
    """The smoothing that best predicts held-out pairs.

    ARGS:
        train_pairs (np.ndarray): [n_words, n_words] counts the table is built from
        held_out_pairs (np.ndarray): [n_words, n_words] counts it is judged on
        candidates (tuple): smoothing values to try

    RETURNS:
        best : float
        scores : dict, smoothing -> held-out nats per pair
    """
    scores = {value: pair_log_likelihood(transition_matrix(train_pairs, value), held_out_pairs) for value in candidates}

    #no held-out pairs: nothing to choose by, so the middle of the range
    if all(np.isnan(score) for score in scores.values()):
        return float(candidates[len(candidates) // 2]), scores
    return float(max(scores, key=lambda value: scores[value])), scores


def stationary_distribution(transitions, n_iterations=10000, tolerance=1e-13):
    """The distribution a long run of the chain settles into.

    ARGS:
        transitions (np.ndarray): [n_words, n_words], rows summing to 1
        n_iterations (int): power-iteration cap
        tolerance (float): stop once no entry moves more than this

    RETURNS:
        [n_words] float64 summing to 1
    """
    n_words = transitions.shape[0]
    distribution = np.full(n_words, 1.0 / n_words)

    #power iteration: a distribution times the chain, until it stops moving
    for _ in range(n_iterations):
        updated = distribution @ transitions
        updated = updated / updated.sum(axis=0)
        if np.abs(updated - distribution).max() < tolerance:
            return updated
        distribution = updated
    return distribution


def fuse_run(standardised, transitions, stationary, temperature, strength=1.0):
    """Combine one run's scores with the word-pair chain by forward-backward.

    ARGS:
        standardised (np.ndarray): [run_length, n_words], each row's
            log-probabilities (or logits) z-scored across the words, in the
            run's word order
        transitions (np.ndarray): [n_words, n_words]
        stationary (np.ndarray): [n_words], the chain's stationary distribution
        temperature (float): z-scores / temperature are taken as
            log-likelihoods
        strength (float): weight of the neighbours' evidence; 1 is exact Bayes

    RETURNS:
        [run_length, n_words] float64 - fused log-scores; a run of one word
        comes back as standardised / temperature
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    run_length, n_words = standardised.shape
    own = np.asarray(standardised, dtype=np.float64) / temperature

    #each word's own likelihood, scaled per row so exp never overflows; axis=1
    #is the word axis
    emissions = np.exp(own - own.max(axis=1, keepdims=True))

    #forward[t]: P(word t | the MEG of words before t), normalised each step
    forward = np.empty((run_length, n_words))
    forward[0] = stationary
    for position in range(1, run_length):
        message = (forward[position - 1] * emissions[position - 1]) @ transitions
        forward[position] = message / message.sum(axis=0)

    #backward[t]: P(the MEG of words after t | word t), up to a constant
    backward = np.empty((run_length, n_words))
    backward[run_length - 1] = 1.0
    for position in range(run_length - 2, -1, -1):
        message = transitions @ (emissions[position + 1] * backward[position + 1])
        backward[position] = message / message.sum(axis=0)

    #the neighbours' evidence as a likelihood ratio: forward over the prior,
    #times backward. The prior is divided out, so frequency earns nothing
    context = np.log(forward) - np.log(stationary)[None, :] + np.log(backward)
    return own + strength * context


def scored_runs(sequence, primary):
    """Maximal runs of consecutive scored tokens within each sequence.

    In a scored dump every token of a sequence is the next word of the
    sentence, so consecutive scored tokens are directly adjacent words.

    ARGS:
        sequence (np.ndarray): [n_tokens] sequence id of each token, in order
        primary (np.ndarray): [n_tokens] scored column, -1 when unscored

    RETURNS:
        list of np.ndarray: token indices of each run, in order, length >= 1
    """
    runs, current = [], []
    for token in range(len(sequence)):
        #a new sequence or an unscored word ends the current run
        if current and (sequence[token] != sequence[current[-1]] or primary[token] < 0):
            runs.append(np.array(current))
            current = []
        if primary[token] >= 0:
            current.append(token)
    if current:
        runs.append(np.array(current))
    return runs


def save_table(path, counts, smoothing, partitions, smoothing_scores):
    """Write a word-pair table, with what built it.

    ARGS:
        path (Path): the .npz to write
        counts (dict): from `count_word_pairs`
        smoothing (float): the chosen pseudo-count
        partitions (list): pnpl partitions the counts came from
        smoothing_scores (dict): smoothing -> held-out nats per pair
    """
    transitions = transition_matrix(counts["pairs"], smoothing)
    np.savez(
        path,
        pairs=counts["pairs"],
        unigram=counts["unigram"],
        smoothing=np.float64(smoothing),
        transitions=transitions,
        stationary=stationary_distribution(transitions),
        partitions=np.array(partitions),
        smoothing_candidates=np.array(list(smoothing_scores), dtype=np.float64),
        smoothing_nats=np.array(list(smoothing_scores.values()), dtype=np.float64),
    )


def load_table(path):
    """Read a word-pair table written by `save_table`.

    ARGS:
        path (Path)

    RETURNS:
        transitions : [N_WORDS, N_WORDS]
        stationary : [N_WORDS]
        info : dict - pairs, smoothing, partitions
    """
    with np.load(path) as stored:
        transitions = stored["transitions"]
        stationary = stored["stationary"]
        info = {
            "pairs": stored["pairs"],
            "smoothing": float(stored["smoothing"]),
            "partitions": [str(name) for name in stored["partitions"]],
        }

    #a table from another vocabulary size would fuse the wrong columns
    if transitions.shape != (N_WORDS, N_WORDS) or stationary.shape != (N_WORDS,):
        raise ValueError(f"{path}: expected a {N_WORDS}-word table, got {transitions.shape}")
    return transitions, stationary, info
