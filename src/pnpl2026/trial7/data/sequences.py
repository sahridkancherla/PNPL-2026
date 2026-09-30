"""Turn a recording's words into sequences shaped like the holdout's.

The holdout decides the shape, measured 2026-09-30 on subject 0:

    only scored words have onsets    -> tokens are in-vocabulary words only;
                                        out-of-vocabulary words are never
                                        tokens, though their MEG stays inside
                                        every window it overlaps
    1-9 scored words per sentence    -> a sentence's tokens are cut into runs
                                        of at most `max_tokens`, near-equal
    MEG ends 1.1-1.7 s after the     -> each sequence stops at its last onset
    last scored word                    plus a tail drawn from that range
    92 of 960 words stand alone      -> `isolated_fraction` of words become
                                        one-word sequences with a 1 s window

⚠️ A validation sequence is fixed once, tail included; a training sequence
draws a fresh tail every time it is read. Isolation is fixed in both, so a
word is never both alone and in a sentence within one split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sequence:
    """One holdout-shaped training or validation example.

    ATTRS:
        recording (int): index into the split's recording list
        onsets (np.ndarray): [n_tokens] onset samples, ascending
        labels (np.ndarray): [n_tokens] competition class ids
        isolated (bool): a single word served with the isolated window
        tail (int or None): samples kept after the last onset; None draws one
            per read
    """

    recording: int
    onsets: np.ndarray
    labels: np.ndarray
    isolated: bool
    tail: int = None


def chunk_sizes(n_tokens, max_tokens):
    """Split n tokens into the fewest runs of at most max_tokens, near-equal.

    Near-equal rather than greedy, so a 10-word sentence becomes 5 + 5 and not
    9 + 1: the holdout's sentences are mostly 4-7 words, and a trailing single
    word would be an isolated word in all but name.

    ARGS:
        n_tokens (int): tokens in the sentence
        max_tokens (int): longest run allowed

    RETURNS:
        list of int, summing to n_tokens

    >>> chunk_sizes(10, 9)
    [5, 5]
    >>> chunk_sizes(19, 9)
    [7, 6, 6]
    """
    n_chunks = -(-n_tokens // max_tokens)
    base, extra = divmod(n_tokens, n_chunks)
    return [base + 1 if index < extra else base for index in range(n_chunks)]


def build_sequences(
    tables,
    n_times,
    sample_rate,
    max_tokens,
    isolated_fraction,
    isolated_seconds,
    tail_seconds,
    fixed_tails,
    seed,
):
    """Holdout-shaped sequences for every recording of one split.

    ARGS:
        tables (list): one word table per recording -- dicts with
            onset_sample, sentence, label
        n_times (list): samples per recording, parallel to `tables`
        sample_rate (float): Hz
        max_tokens (int): longest sequence
        isolated_fraction (float): share of words served alone
        isolated_seconds (float): the isolated window
        tail_seconds (tuple): (low, high) MEG kept after the last onset
        fixed_tails (bool): draw each tail once here (validation) rather than
            per read (training)
        seed (int): for the isolation draw and any fixed tails

    RETURNS:
        list of Sequence
    """
    generator = np.random.default_rng(seed)
    isolated_samples = int(round(isolated_seconds * sample_rate))
    low, high = (int(round(value * sample_rate)) for value in tail_seconds)

    sequences = []
    for recording, (table, length) in enumerate(zip(tables, n_times)):
        #in-vocabulary words only, and only those with a full isolated window:
        #the holdout guarantees at least that much after every scored onset
        keep = (table["label"] >= 0) & (table["onset_sample"] + high <= length)
        keep &= table["onset_sample"] >= 0
        onsets = table["onset_sample"][keep]
        labels = table["label"][keep].astype(np.int64)
        sentences = table["sentence"][keep]

        #a fixed share of words stands alone, drawn once for the split
        alone = generator.random(len(onsets)) < isolated_fraction
        for index in np.flatnonzero(alone):
            sequences.append(
                Sequence(recording, onsets[index : index + 1], labels[index : index + 1], True, isolated_samples)
            )

        #the rest keep their sentence; a word with no sentence id is its own
        in_context = np.flatnonzero(~alone)
        for group in _sentence_groups(sentences[in_context]):
            members = in_context[group]
            start = 0
            for size in chunk_sizes(len(members), max_tokens):
                chosen = members[start : start + size]
                start += size
                tail = int(generator.integers(low, high + 1)) if fixed_tails else None
                sequences.append(Sequence(recording, onsets[chosen], labels[chosen], False, tail))

    return sequences


def _sentence_groups(sentence_ids):
    """Positions grouped into runs of one sentence id, in order.

    Consecutive runs, not a global group-by: the tables are in onset order, and
    a sentence id that reappears later would otherwise join words minutes apart.

    ARGS:
        sentence_ids (np.ndarray): [n] sentence id per word, -1 for none

    RETURNS:
        list of np.ndarray, each a run of positions into `sentence_ids`
    """
    if len(sentence_ids) == 0:
        return []

    #a boundary wherever the id changes, and around every -1 on its own
    boundary = np.flatnonzero(
        (sentence_ids[1:] != sentence_ids[:-1]) | (sentence_ids[1:] < 0) | (sentence_ids[:-1] < 0)
    ) + 1
    return np.split(np.arange(len(sentence_ids)), boundary)
