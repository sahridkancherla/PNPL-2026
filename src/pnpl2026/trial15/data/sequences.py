"""Sentences as token sequences, under three protocols.

    REFERENCE   d'Ascoli et al. and the competition paper's 73.23%: every word
                of a sentence is a token, every window a full 3 s from onset.
                Long sentences are cut into near-equal runs of `max_tokens`.
                What trial 10 trained and was judged on
    FRAGMENT    what the competition holdout actually serves, measured on its
                files 2026-10-02: runs of 4-8 CONSECUTIVE vocabulary words
                (median onset gap 0.21 s), the MEG starting at the first
                word's onset and stopping 1.13-1.68 s after the last, so
                windows are median 1.94 s; plus 92 of 960 words alone with 1 s
    SPARSE      trial 10's "holdout" protocol, kept for continuity only: every
                vocabulary word of a sentence as tokens, the unscored words
                between them skipped. It read ~27 for every run where the
                leaderboard read 41-58, because the real holdout is dense runs

Trial 13 selects checkpoints on FRAGMENT and reports all three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sequence:
    """One sentence, or one run of a long one, as tokens.

    ATTRS:
        recording (int): index into the partition's recording list
        onsets (np.ndarray): [n_tokens] onset samples, ascending
        targets (np.ndarray): [n_tokens] t5 rows, -1 when a word has none
        primary (np.ndarray): [n_tokens] competition columns, -1 unscored
        moses (np.ndarray): [n_tokens] Moses columns, -1 unscored
        isolated (bool): a single word with the isolated window
        tail (int or None): samples kept after the last onset. None is the
            reference protocol: every window is full
    """

    recording: int
    onsets: np.ndarray
    targets: np.ndarray
    primary: np.ndarray
    moses: np.ndarray
    isolated: bool = False
    tail: int = None


def chunk_sizes(n_tokens, max_tokens):
    """Split n tokens into the fewest runs of at most max_tokens, near-equal.

    ARGS:
        n_tokens (int)
        max_tokens (int)

    RETURNS:
        list of int, summing to n_tokens

    >>> chunk_sizes(70, 64)
    [35, 35]
    >>> chunk_sizes(10, 9)
    [5, 5]
    """
    n_chunks = -(-n_tokens // max_tokens)
    base, extra = divmod(n_tokens, n_chunks)
    return [base + 1 if index < extra else base for index in range(n_chunks)]


def sentence_groups(sentence_ids):
    """Positions grouped into consecutive runs of one sentence id.

    Consecutive runs, not a global group-by: an id that reappears later in a
    recording would otherwise join words minutes apart.

    ARGS:
        sentence_ids (np.ndarray): [n] sentence id per word, -1 for none

    RETURNS:
        list of np.ndarray of positions
    """
    if len(sentence_ids) == 0:
        return []

    #a boundary wherever the id changes, and around every -1 on its own
    boundary = np.flatnonzero(
        (sentence_ids[1:] != sentence_ids[:-1]) | (sentence_ids[1:] < 0) | (sentence_ids[:-1] < 0)
    ) + 1
    return np.split(np.arange(len(sentence_ids)), boundary)


def reference_sequences(tables, n_times, target_rows, window_samples, max_tokens, token_set="all"):
    """Every sentence as tokens with full windows: the reference protocol.

    ARGS:
        tables (list): word tables, one per recording
        n_times (list): samples per recording
        target_rows (list): per recording, [n_words] t5 rows
        window_samples (int): the full window, 150 at 50 Hz
        max_tokens (int): longest run
        token_set (str): "all" or "invocab"

    RETURNS:
        list of Sequence
    """
    sequences = []
    for recording, (table, length, rows) in enumerate(zip(tables, n_times, target_rows)):
        #a word whose full window runs off the recording is dropped, as an
        #epoch past the end would be
        onsets = table["onset_sample"]
        keep = (onsets >= 0) & (onsets + window_samples <= length)
        if token_set == "invocab":
            keep &= table["primary"] >= 0

        positions = np.flatnonzero(keep)
        for group in sentence_groups(table["sentence"][positions]):
            members = positions[group]
            start = 0
            for size in chunk_sizes(len(members), max_tokens):
                chosen = members[start : start + size]
                start += size
                sequences.append(
                    Sequence(
                        recording,
                        onsets[chosen],
                        rows[chosen],
                        table["primary"][chosen].astype(np.int64),
                        table["moses"][chosen].astype(np.int64),
                    )
                )
    return sequences


def vocabulary_runs(table, length, max_gap_samples, room_samples):
    """Maximal runs of consecutive vocabulary words, as the holdout cuts them.

    Consecutive means adjacent rows of the word table -- no unscored word in
    between -- in one sentence, each onset within `max_gap_samples` of the
    previous one. A word without `room_samples` after its onset ends a run.

    ARGS:
        table (dict): one recording's word table
        length (int): samples in the recording
        max_gap_samples (int): a longer onset gap ends a run
        room_samples (int): MEG needed after a word's onset

    RETURNS:
        list of np.ndarray, each the table rows of one run, in onset order
    """
    onsets = table["onset_sample"]
    usable = (table["primary"] >= 0) & (onsets >= 0) & (onsets + room_samples <= length)

    runs, current = [], []
    for row in range(len(onsets)):
        #a row continues the run only if it directly follows the last one
        joins = (
            current
            and usable[row]
            and row == current[-1] + 1
            and table["sentence"][row] == table["sentence"][current[-1]]
            and table["sentence"][row] >= 0
            and onsets[row] - onsets[current[-1]] <= max_gap_samples
        )
        if joins:
            current.append(row)
            continue

        #anything else closes the run and may open a new one here
        if current:
            runs.append(np.array(current))
        current = [row] if usable[row] else []
    if current:
        runs.append(np.array(current))
    return runs


def fragment_sequences(
    tables,
    n_times,
    target_rows,
    sample_rate,
    max_tokens,
    min_words,
    max_gap_seconds,
    isolated_fraction,
    isolated_seconds,
    tail_seconds,
    seed,
):
    """Holdout-shaped sequences: runs of consecutive vocabulary words.

    Runs shorter than `min_words` are dropped (the holdout's are almost all
    4-8 words); longer than `max_tokens` are cut into near-equal pieces. Each
    keeps its MEG to the last onset plus a tail drawn from the holdout's range.
    Then, as the holdout's 92 of 960, a matching share of OTHER vocabulary
    words is served alone with exactly `isolated_seconds`.

    ARGS:
        tables (list): word tables, one per recording
        n_times (list): samples per recording
        target_rows (list): per recording, [n_words] t5 rows
        sample_rate (float): Hz of the cache
        max_tokens (int): longest fragment, 9
        min_words (int): shortest fragment kept
        max_gap_seconds (float): a longer onset gap ends a run
        isolated_fraction (float): share of all scored words served alone
        isolated_seconds (float): their window
        tail_seconds (tuple): (low, high) MEG kept after a fragment's last onset
        seed (int): one draw for tails and isolated words, so every run is
            scored on identical sequences

    RETURNS:
        list of Sequence
    """
    generator = np.random.default_rng(seed)
    isolated_samples = int(round(isolated_seconds * sample_rate))
    max_gap = int(round(max_gap_seconds * sample_rate))
    low, high = (int(round(value * sample_rate)) for value in tail_seconds)

    sequences = []
    for recording, (table, length, rows) in enumerate(zip(tables, n_times, target_rows)):
        onsets = table["onset_sample"]
        in_fragment = np.zeros(len(onsets), dtype=bool)

        #every word of a fragment needs room for the longest tail after it
        for run in vocabulary_runs(table, length, max_gap, high):
            if len(run) < min_words:
                continue
            start = 0
            for size in chunk_sizes(len(run), max_tokens):
                chosen = run[start : start + size]
                start += size
                in_fragment[chosen] = True
                sequences.append(
                    Sequence(
                        recording,
                        onsets[chosen],
                        rows[chosen],
                        table["primary"][chosen].astype(np.int64),
                        table["moses"][chosen].astype(np.int64),
                        False,
                        int(generator.integers(low, high + 1)),
                    )
                )

        #isolated words: other vocabulary words, in the holdout's proportion
        n_fragment = int(in_fragment.sum())
        n_alone = int(round(n_fragment * isolated_fraction / (1.0 - isolated_fraction)))
        candidates = np.flatnonzero(
            (table["primary"] >= 0) & ~in_fragment & (onsets >= 0) & (onsets + isolated_samples <= length)
        )
        if n_alone == 0 or len(candidates) == 0:
            continue
        for position in np.sort(generator.choice(candidates, size=min(n_alone, len(candidates)), replace=False)):
            chosen = np.array([position])
            sequences.append(
                Sequence(
                    recording,
                    onsets[chosen],
                    rows[chosen],
                    table["primary"][chosen].astype(np.int64),
                    table["moses"][chosen].astype(np.int64),
                    True,
                    isolated_samples,
                )
            )
    return sequences


def sparse_sequences(
    tables,
    n_times,
    target_rows,
    sample_rate,
    max_tokens,
    isolated_fraction,
    isolated_seconds,
    tail_seconds,
    seed,
):
    """Trial 10's mis-specified holdout protocol, kept for comparison only.

    Every vocabulary word of a sentence becomes a token, the unscored words
    between them skipped -- sparse tokens over long spans, which the real
    holdout never serves. Use `fragment_sequences` for anything that matters.

    ARGS:
        tables (list): word tables, one per recording
        n_times (list): samples per recording
        target_rows (list): per recording, [n_words] t5 rows
        sample_rate (float): Hz of the cache
        max_tokens (int): longest run, 9 in the holdout
        isolated_fraction (float): share of words served alone
        isolated_seconds (float): their window
        tail_seconds (tuple): (low, high) MEG kept after the last word
        seed (int): one draw for isolation and tails, so every run is scored
            on identical sequences

    RETURNS:
        list of Sequence
    """
    generator = np.random.default_rng(seed)
    isolated_samples = int(round(isolated_seconds * sample_rate))
    low, high = (int(round(value * sample_rate)) for value in tail_seconds)

    sequences = []
    for recording, (table, length, rows) in enumerate(zip(tables, n_times, target_rows)):
        #scored words only, each with room for the longest tail
        onsets = table["onset_sample"]
        keep = (table["primary"] >= 0) & (onsets >= 0) & (onsets + high <= length)
        positions = np.flatnonzero(keep)

        alone = generator.random(len(positions)) < isolated_fraction
        for position in positions[alone]:
            chosen = np.array([position])
            sequences.append(
                Sequence(
                    recording,
                    onsets[chosen],
                    rows[chosen],
                    table["primary"][chosen].astype(np.int64),
                    table["moses"][chosen].astype(np.int64),
                    True,
                    isolated_samples,
                )
            )

        in_context = positions[~alone]
        for group in sentence_groups(table["sentence"][in_context]):
            members = in_context[group]
            start = 0
            for size in chunk_sizes(len(members), max_tokens):
                chosen = members[start : start + size]
                start += size
                sequences.append(
                    Sequence(
                        recording,
                        onsets[chosen],
                        rows[chosen],
                        table["primary"][chosen].astype(np.int64),
                        table["moses"][chosen].astype(np.int64),
                        False,
                        int(generator.integers(low, high + 1)),
                    )
                )
    return sequences
