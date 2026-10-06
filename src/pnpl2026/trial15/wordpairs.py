"""A word-pair model of the 50 scored words, and the fusion that uses it.

Every holdout "sentence" is a run of 4-9 scored words. Most neighbours in a
run are directly adjacent in the speech, but not all: inside the holdout's
runs, onset gaps are 0.21 s at the median against 0.14 s for adjacent scored
pairs in training, and the upper quarter looks like a pair with ONE unscored
word between (0.28-1.16 s in training). No model trained so far knows either
relation; they trained on narrative, where most neighbours are unscored.

    pairs[a, b]        how often scored word b directly follows scored word a
                       in one sentence of subject 0's transcripts (the cache's
                       `words_<key>.npz` tables)
    skip_pairs[a, b]   how often b is the next scored word after a with exactly
                       one unscored word between them
    transitions        each, row-smoothed toward the overall next-word
                       distribution, as P(next = b | this = a)
    stationary         the adjacent chain's stationary distribution
    gap models         log-normal fits of the onset gaps of the two kinds of
                       pair, so a holdout pair's gap says which kind it likely is

`fuse_run` combines a submission's scores with the chain along one run by
forward-backward; each step may have its own transition matrix. Each word's
score becomes

    own scores / temperature  +  strength * log P(the other words' MEG | this word)

-- its own evidence plus what the neighbours' evidence says about it, as a
likelihood ratio. The prior is divided out (log forward - log prior), so
frequent words gain nothing for being frequent: the metric weights every word
equally. With the neighbours' evidence flat, the context term is constant and
the ranking is the input's.

The gap-aware chain (`step_transitions`): each step mixes the adjacent and
skip-one matrices by P(adjacent | that pair's onset gap), from the two gap
models, a global speech-rate shift and the share of adjacent pairs -- both
fitted on the holdout's own gaps by `fit_gap_mixture` (inputs, never labels).
The adjacency-only chain (every step adjacent) lost 0.8-2.3 leaderboard
points on 2026-10-06.

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

#the cache's onsets are 50 Hz samples
SAMPLE_RATE = 50.0

#gaps are clamped to this before the log: two onsets can share a 50 Hz sample
MIN_GAP_SECONDS = 0.01

#the speech-rate shifts (log units, x0.61 to x1.65) and adjacent shares the
#gap mixture is fitted over
GAP_SHIFTS = np.linspace(-0.5, 0.5, 101)
GAP_SHARES = np.linspace(0.01, 0.99, 99)


def count_word_pairs(tables, n_words=N_WORDS, max_gap_samples=MAX_GAP_SAMPLES):
    """Count adjacent and skip-one scored-word pairs, and their onset gaps.

    Two words are together when they are consecutive rows of one recording's
    word table and share a sentence id; rows with no sentence id (-1) are
    together when their onsets are at most `max_gap_samples` apart. A skip-one
    pair is scored, unscored, scored, all three together.

    ARGS:
        tables (iterable of dict): word tables with `primary` [n] (column or
            -1), `sentence` [n] and `onset_sample` [n] at 50 Hz, onset order
        n_words (int): 50
        max_gap_samples (int): the pairing limit where sentences are unknown

    RETURNS:
        dict:
            pairs : [n_words, n_words] float64, row = first word
            skip_pairs : [n_words, n_words] float64, one unscored word between
            unigram : [n_words] float64, every scored token
            adjacent_gaps : [n_pairs] int64, onset gaps (samples) of `pairs`
            skip_gaps : [n_skip_pairs] int64, onset gaps of `skip_pairs`
    """
    pairs = np.zeros((n_words, n_words), dtype=np.float64)
    skip_pairs = np.zeros((n_words, n_words), dtype=np.float64)
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

        #scored, unscored, scored, all together: a pair with one word skipped
        if len(primary) >= 3:
            spans = together[:-1] & together[1:]
            skipped = spans & (primary[:-2] >= 0) & (primary[1:-1] < 0) & (primary[2:] >= 0)
            np.add.at(skip_pairs, (primary[:-2][skipped], primary[2:][skipped]), 1.0)
            skip_gaps.append((onset[2:] - onset[:-2])[skipped])

    #empty arrays rather than a failed concatenate when nothing was found
    return {
        "pairs": pairs,
        "skip_pairs": skip_pairs,
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


def log_gap_model(gaps_seconds):
    """A log-normal fit of onset gaps: mean and spread of log seconds.

    ARGS:
        gaps_seconds (np.ndarray): [n] gaps, > 0

    RETURNS:
        [2] float64: mean, standard deviation of log(gap)
    """
    if len(gaps_seconds) < 2:
        raise ValueError("A gap model needs at least two gaps.")

    #axis=0 is the only axis: one mean and one spread over every gap
    logs = np.log(np.maximum(gaps_seconds, MIN_GAP_SECONDS))
    return np.array([logs.mean(axis=0), max(float(logs.std(axis=0)), 1e-3)])


def gap_log_densities(gaps_seconds, gap_models, shift):
    """Log density of each gap under the adjacent and the skip-one model.

    ARGS:
        gaps_seconds (np.ndarray): [n]
        gap_models (dict): "adjacent", "skip" -> [mean, sd] of log gap
        shift (float): added to both means, a global speech-rate change

    RETURNS:
        [n, 2] float64 - column 0 adjacent, column 1 skip-one
    """
    logs = np.log(np.maximum(np.asarray(gaps_seconds, dtype=np.float64), MIN_GAP_SECONDS))
    columns = []
    for kind in ("adjacent", "skip"):
        mean, spread = gap_models[kind]

        #a normal density in log-gap space, the same Jacobian for both kinds
        z = (logs - mean - shift) / spread
        columns.append(-0.5 * z**2 - np.log(spread) - 0.5 * np.log(2 * np.pi))
    return np.stack(columns, axis=1)


def fit_gap_mixture(gaps_seconds, gap_models, shifts=GAP_SHIFTS, shares=GAP_SHARES):
    """The speech-rate shift and adjacent share that best explain some gaps.

    Maximum likelihood over a grid: each gap is adjacent with probability
    `share`, else skip-one, both models shifted by `shift` in log seconds.

    ARGS:
        gaps_seconds (np.ndarray): [n] gaps between neighbours in runs
        gap_models (dict): from the table
        shifts (np.ndarray): candidate shifts
        shares (np.ndarray): candidate adjacent shares

    RETURNS:
        shift : float
        share : float
    """
    best_total, best_shift, best_share = -np.inf, 0.0, 0.5
    for shift in shifts:
        densities = gap_log_densities(gaps_seconds, gap_models, shift)

        #every share at once: [n, 1] densities against [1, n_shares] shares
        mixed = np.logaddexp(
            np.log(shares)[None, :] + densities[:, :1],
            np.log(1.0 - shares)[None, :] + densities[:, 1:],
        )
        totals = mixed.sum(axis=0)
        if totals.max() > best_total:
            best_total, best_shift, best_share = float(totals.max()), float(shift), float(shares[totals.argmax()])
    return best_shift, best_share


def adjacent_weight(gaps_seconds, gap_models, shift, share):
    """P(a pair is directly adjacent | its onset gap).

    ARGS:
        gaps_seconds (np.ndarray): [n]
        gap_models (dict): from the table
        shift (float), share (float): from `fit_gap_mixture`

    RETURNS:
        [n] float64 in (0, 1)
    """
    densities = gap_log_densities(gaps_seconds, gap_models, shift)
    adjacent = np.log(share) + densities[:, 0]
    skip = np.log(1.0 - share) + densities[:, 1]
    return np.exp(adjacent - np.logaddexp(adjacent, skip))


def step_transitions(weights, transitions, skip_transitions):
    """One transition matrix per step of a run, adjacent and skip-one mixed.

    ARGS:
        weights (np.ndarray): [n_steps] P(adjacent) of each neighbouring pair
        transitions (np.ndarray): [n_words, n_words] adjacent chain
        skip_transitions (np.ndarray): [n_words, n_words] skip-one chain

    RETURNS:
        [n_steps, n_words, n_words] float64, every row summing to 1
    """
    #[n_steps, 1, 1] weights against [1, n_words, n_words] matrices
    weights = np.asarray(weights, dtype=np.float64)[:, None, None]
    return weights * transitions[None, :, :] + (1.0 - weights) * skip_transitions[None, :, :]


def fuse_run(standardised, transitions, start, temperature, strength=1.0):
    """Combine one run's scores with the word-pair chain by forward-backward.

    ARGS:
        standardised (np.ndarray): [run_length, n_words], each row's
            log-probabilities (or logits) z-scored across the words, in the
            run's word order
        transitions (np.ndarray): [n_words, n_words] for every step, or
            [run_length - 1, n_words, n_words], step t from word t to t+1
        start (np.ndarray): [n_words], the prior of the run's first word
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
    n_steps = max(run_length - 1, 0)

    #one matrix per step, whether given once or per step
    steps = np.broadcast_to(transitions, (n_steps, n_words, n_words)) if transitions.ndim == 2 else transitions
    if steps.shape != (n_steps, n_words, n_words):
        raise ValueError(f"transitions {transitions.shape} for a run of {run_length} words")
    own = np.asarray(standardised, dtype=np.float64) / temperature

    #each word's own likelihood, scaled per row so exp never overflows; axis=1
    #is the word axis
    emissions = np.exp(own - own.max(axis=1, keepdims=True))

    #prior[t]: P(word t) before any MEG; forward[t]: P(word t | MEG before t)
    prior = np.empty((run_length, n_words))
    forward = np.empty((run_length, n_words))
    prior[0] = forward[0] = start
    for position in range(1, run_length):
        prior[position] = prior[position - 1] @ steps[position - 1]
        message = (forward[position - 1] * emissions[position - 1]) @ steps[position - 1]
        forward[position] = message / message.sum(axis=0)

    #backward[t]: P(the MEG of words after t | word t), up to a constant
    backward = np.empty((run_length, n_words))
    backward[run_length - 1] = 1.0
    for position in range(run_length - 2, -1, -1):
        message = steps[position] @ (emissions[position + 1] * backward[position + 1])
        backward[position] = message / message.sum(axis=0)

    #the neighbours' evidence as a likelihood ratio: forward over the prior,
    #times backward. The prior is divided out, so frequency earns nothing
    context = np.log(forward) - np.log(prior) + np.log(backward)
    return own + strength * context


def scored_chains(sequence, primary, max_skipped=0):
    """Chains of scored tokens within each sequence, and what lies between them.

    In a scored dump every token of a sequence is the next word of the
    sentence. A chain continues across at most `max_skipped` unscored tokens.

    ARGS:
        sequence (np.ndarray): [n_tokens] sequence id of each token, in order
        primary (np.ndarray): [n_tokens] scored column, -1 when unscored
        max_skipped (int): 0 for directly adjacent words only

    RETURNS:
        list of (tokens, skips): tokens [length] token indices in order,
        skips [length - 1] unscored tokens between each neighbouring pair
    """
    chains = []
    tokens, skips, pending = [], [], 0

    def close():
        """Keep the chain built so far, if any, and start a new one."""
        if tokens:
            chains.append((np.array(tokens), np.array(skips, dtype=np.int64)))
        tokens.clear()
        skips.clear()

    for token in range(len(sequence)):
        #a new sequence always ends the chain
        if tokens and sequence[token] != sequence[tokens[-1]]:
            close()
            pending = 0

        #an unscored token is skipped, until there are too many in a row
        if primary[token] < 0:
            if tokens:
                pending += 1
                if pending > max_skipped:
                    close()
                    pending = 0
            continue

        #a scored token joins the chain, noting what it skipped to get here
        if tokens:
            skips.append(pending)
        tokens.append(token)
        pending = 0
    close()
    return chains


def scored_runs(sequence, primary):
    """Maximal runs of directly adjacent scored tokens within each sequence.

    ARGS:
        sequence (np.ndarray): [n_tokens] sequence id of each token, in order
        primary (np.ndarray): [n_tokens] scored column, -1 when unscored

    RETURNS:
        list of np.ndarray: token indices of each run, in order, length >= 1
    """
    return [tokens for tokens, _ in scored_chains(sequence, primary, max_skipped=0)]


def save_table(path, counts, held_out, partitions):
    """Choose both smoothings, fit both gap models, and write the table.

    ARGS:
        path (Path): the .npz to write
        counts (dict): `count_word_pairs` on the counted partitions
        held_out (dict): `count_word_pairs` on the held-out partition
        partitions (list): pnpl partitions the counts came from

    RETURNS:
        dict: smoothing, skip_smoothing, and the held-out nats per pair of
        every candidate, for the report
    """
    smoothing, smoothing_scores = choose_smoothing(counts["pairs"], held_out["pairs"])
    skip_smoothing, skip_scores = choose_smoothing(counts["skip_pairs"], held_out["skip_pairs"])
    transitions = transition_matrix(counts["pairs"], smoothing)

    #gaps in seconds: the holdout's onsets are seconds, the cache's samples
    adjacent_model = log_gap_model(counts["adjacent_gaps"] / SAMPLE_RATE)
    skip_model = log_gap_model(counts["skip_gaps"] / SAMPLE_RATE)
    np.savez(
        path,
        pairs=counts["pairs"],
        skip_pairs=counts["skip_pairs"],
        unigram=counts["unigram"],
        smoothing=np.float64(smoothing),
        skip_smoothing=np.float64(skip_smoothing),
        transitions=transitions,
        skip_transitions=transition_matrix(counts["skip_pairs"], skip_smoothing),
        stationary=stationary_distribution(transitions),
        adjacent_log_gap=adjacent_model,
        skip_log_gap=skip_model,
        partitions=np.array(partitions),
    )
    return {"smoothing": smoothing, "skip_smoothing": skip_smoothing,
            "smoothing_scores": smoothing_scores, "skip_scores": skip_scores}


def load_table(path):
    """Read a word-pair table written by `save_table`.

    ARGS:
        path (Path)

    RETURNS:
        dict: transitions, skip_transitions [N_WORDS, N_WORDS]; stationary
        [N_WORDS]; gap_models ("adjacent", "skip" -> [mean, sd]); pairs,
        smoothing, skip_smoothing, partitions
    """
    with np.load(path) as stored:
        #a table from before the skip-one model cannot fuse gap-aware
        if "skip_transitions" not in stored.files:
            raise SystemExit(f"{path} predates the skip-one model; rebuild it with cli.wordpairs.")
        table = {
            "transitions": stored["transitions"],
            "skip_transitions": stored["skip_transitions"],
            "stationary": stored["stationary"],
            "gap_models": {"adjacent": stored["adjacent_log_gap"], "skip": stored["skip_log_gap"]},
            "pairs": stored["pairs"],
            "smoothing": float(stored["smoothing"]),
            "skip_smoothing": float(stored["skip_smoothing"]),
            "partitions": [str(name) for name in stored["partitions"]],
        }

    #a table from another vocabulary size would fuse the wrong columns
    for name in ("transitions", "skip_transitions"):
        if table[name].shape != (N_WORDS, N_WORDS):
            raise ValueError(f"{path}: expected a {N_WORDS}-word {name}, got {table[name].shape}")
    return table
