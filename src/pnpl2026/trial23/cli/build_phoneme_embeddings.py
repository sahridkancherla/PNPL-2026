"""Build Variant P's phonological target table: every word's sound shape as a vector.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.build_phoneme_embeddings

The research note's acoustic target (w2v-bert-2.0 on the stimulus audio)
cannot be built here: pnpl's LibriBrain100 ships no audio. Its idea -- a
second, non-semantic target space beside t5's -- can: the event tables align
every word to its phonemes. Each word's vector is HDND's form embedding
(arXiv 2609.24095) with phonemes in place of letters:

    the word's most frequent pronunciation, as ARPAbet symbols
    -> boundary-marked n-grams, n = 1..3:  <, DH, AH, >, <DH, DHAH, AH>, ...
    -> each n-gram a fixed Gaussian vector, seeded by a stable hash
    -> summed, L2-normalised                                  [1024]

Words that sound alike land close together, whatever they mean: the errors
it invites are not t5's. A word no event table spells out (only the two
vocabularies can add one) falls back to its LETTER n-grams, so no row is
zero -- `TargetTable` refuses zero rows.

⚠️ The rows are the t5 table's words in the t5 table's order: a second head
indexes both tables with the same targets, and `lit.py` checks the lists
are equal.

Writes data/word_embeddings_phon.npz (words, embeddings). Light: reads every
subject-0 event table once; minutes on a login node, no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from ...config import DATA_DIR, REPO_ROOT
from ..data.words import Vocabulary, normalise_word
from ..data_processing.build_reference import (
    EVENTS_KIND_COLUMN,
    EVENTS_SEGMENT_COLUMN,
    EVENTS_SENTENCE_COLUMN,
    WORD_KIND,
    discover_recordings,
)

#the table's width: t5-large's, so a second head maps into the same size
DEFAULT_DIM = 1024

#n-gram orders, as HDND's form embedding
MAX_ORDER = 3

#the events' other link columns, beside the sentence index
EVENTS_WORD_COLUMN = "wordidx"
EVENTS_PHONEME_COLUMN = "phonemeidx"
PHONEME_KIND = "phoneme"

#silence carries no sound shape of the word
SILENCE = "SIL"


def parse_args():
    """Command-line arguments for the table build."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root holding the task folders")
    parser.add_argument("--table", default="data/word_embeddings_t5_all.npz", help="repo-relative; its words and order")
    parser.add_argument("--out", default="data/word_embeddings_phon.npz", help="repo-relative output")
    parser.add_argument("--dim", type=int, default=DEFAULT_DIM, help="vector width")
    return parser.parse_args()


def phoneme_symbol(label, mapping):
    """One event label as an ARPAbet symbol, or None for silence.

    ARGS:
        label (object): the phoneme row's segment, e.g. "ah", "AH0", "bcl"
        mapping (dict): pnpl's TIMIT -> ARPAbet table

    RETURNS:
        str or None
    """
    #stress digits carry no identity here: "ah0" and "ah1" are one sound
    text = re.sub(r"\d", "", str(label).strip().lower())

    #LibriBrain tags each phoneme's place in its word ("b_B", "uh_I", "k_E"):
    #the sound is the part before the tag
    text = re.sub(r"_[a-z]+$", "", text)
    if not text or text == "nan":
        return None
    symbol = mapping.get(text, text.upper())
    return None if symbol == SILENCE else symbol


def pronunciations(events_path, mapping):
    """Every word token's phoneme sequence in one event table.

    ARGS:
        events_path (Path): one recording's events TSV
        mapping (dict): pnpl's TIMIT -> ARPAbet table

    RETURNS:
        list of (normalised word, tuple of symbols)
    """
    events = pd.read_csv(events_path, sep="\t")
    link = [EVENTS_SENTENCE_COLUMN, EVENTS_WORD_COLUMN]

    #phonemes grouped by the word they belong to, in their own order
    phonemes = events[events[EVENTS_KIND_COLUMN] == PHONEME_KIND].dropna(subset=link)
    phonemes = phonemes.sort_values(link + [EVENTS_PHONEME_COLUMN], kind="stable")
    sounds = defaultdict(list)
    for sentence, word, label in zip(
        phonemes[EVENTS_SENTENCE_COLUMN], phonemes[EVENTS_WORD_COLUMN], phonemes[EVENTS_SEGMENT_COLUMN]
    ):
        symbol = phoneme_symbol(label, mapping)
        if symbol is not None:
            sounds[(int(sentence), int(word))].append(symbol)

    #each word row paired with its phonemes; a word without any is skipped
    found = []
    words = events[events[EVENTS_KIND_COLUMN] == WORD_KIND].dropna(subset=link)
    for sentence, word, segment in zip(words[EVENTS_SENTENCE_COLUMN], words[EVENTS_WORD_COLUMN], words[EVENTS_SEGMENT_COLUMN]):
        spelled = normalise_word(segment)
        symbols = sounds.get((int(sentence), int(word)))
        if spelled and symbols:
            found.append((spelled, tuple(symbols)))
    return found


def ngram_vector(gram, dim):
    """The fixed Gaussian vector of one n-gram, the same on every machine.

    ARGS:
        gram (str): the n-gram, joined with "|" and prefixed by its kind
        dim (int): width

    RETURNS:
        [dim] float64, unit expected norm
    """
    #md5, not hash(): Python's own hash changes from one process to the next
    seed = int(hashlib.md5(gram.encode("utf-8")).hexdigest()[:16], 16)
    return np.random.default_rng(seed).standard_normal(dim) / np.sqrt(dim)


def form_vector(symbols, kind, dim):
    """HDND's form embedding of one symbol sequence.

    ARGS:
        symbols (sequence): phonemes, or letters for the fallback
        kind (str): "phoneme" or "letter"; the two never share n-grams
        dim (int): width

    RETURNS:
        [dim] float32, unit length
    """
    #boundary marks, so a word's first and last sounds are their own n-grams
    marked = ["<"] + list(symbols) + [">"]
    total = np.zeros(dim)
    for order in range(1, MAX_ORDER + 1):
        for start in range(len(marked) - order + 1):
            total += ngram_vector(f"{kind}:" + "|".join(marked[start : start + order]), dim)
    return (total / np.linalg.norm(total)).astype(np.float32)


def main():
    """Read every pronunciation, embed every word of the t5 table, write the table."""
    args = parse_args()
    from pnpl.datasets.constants import TIMIT_TO_ARPABET

    #the t5 table fixes which words, and in what order
    with np.load(REPO_ROOT / args.table, allow_pickle=False) as stored:
        stored_words = stored["words"]
    words = [normalise_word(word) for word in stored_words]

    #every token's pronunciation, counted per word
    heard = defaultdict(Counter)
    recordings = discover_recordings(args.data_dir)
    n_tokens = 0
    for found in recordings:
        for spelled, symbols in pronunciations(found["events_path"], TIMIT_TO_ARPABET):
            heard[spelled][symbols] += 1
            n_tokens += 1
    print(f"[phon] {n_tokens:,} word tokens with phonemes from {len(recordings)} event tables; "
          f"{len(heard):,} distinct words")

    #each word's commonest pronunciation; letters where none was heard
    rows, fallback, ambiguous = [], [], 0
    for word in words:
        if word in heard:
            counts = heard[word]
            ambiguous += len(counts) > 1
            rows.append(form_vector(counts.most_common(1)[0][0], "phoneme", args.dim))
        else:
            fallback.append(word)
            rows.append(form_vector(word, "letter", args.dim))
    table = np.stack(rows, axis=0)
    print(f"[phon] {len(words):,} rows: {len(words) - len(fallback):,} from phonemes "
          f"({ambiguous:,} with more than one pronunciation, the commonest kept), "
          f"{len(fallback):,} from letters: {fallback[:10]}")

    #the 50 scored words: how separable their sound shapes are
    primary = Vocabulary("primary").words
    index = {word: row for row, word in enumerate(words)}
    missing = [word for word in primary if word not in index]
    if missing:
        raise SystemExit(f"Scored words missing from {args.table}: {missing}")
    vectors = table[[index[word] for word in primary]]
    cosines = vectors @ vectors.T
    pairs = sorted(
        ((float(cosines[a, b]), primary[a], primary[b]) for a in range(len(primary)) for b in range(a + 1, len(primary))),
        reverse=True,
    )
    print("[phon] closest scored pairs: " + ", ".join(f"{a}/{b} {value:.2f}" for value, a, b in pairs[:8]))
    identical = sum(pair[0] > 0.999 for pair in pairs)
    print(f"[phon] scored pairs' cosine: median {np.median([pair[0] for pair in pairs]):.2f}, identical {identical}")

    #true homophones are a handful at most; many more means unlinked phonemes
    if identical > 3:
        print(f"[phon] WARNING: {identical} scored pairs sound identical -- check that phoneme rows "
              f"carry their word's sentenceidx and wordidx before training on this table")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, words=stored_words, embeddings=table)
    print(f"[phon] wrote {out_path} ({table.shape[0]:,} x {table.shape[1]})")


if __name__ == "__main__":
    main()
