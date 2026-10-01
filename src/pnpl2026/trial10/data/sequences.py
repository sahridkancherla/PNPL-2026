"""Sentences as token sequences, under two protocols.

    REFERENCE   d'Ascoli et al. and the competition paper's 73.23%: every word
                of a sentence is a token, every window a full 3 s from onset.
                Long sentences are cut into near-equal runs of `max_tokens`.
                F2 keeps only scored words as tokens, still with full windows
    HOLDOUT     what the competition holdout serves (trial 7's measurements):
                scored words only, runs of at most 9, the MEG stopping
                1.1-1.7 s after the last word, 9.6% of words alone with 1 s

Trial 10 trains under the reference protocol and is scored under both, so the
gap between them is measured rather than argued.
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


def holdout_sequences(
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
    """Holdout-shaped sequences: trial 7's protocol, every tail fixed.

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
