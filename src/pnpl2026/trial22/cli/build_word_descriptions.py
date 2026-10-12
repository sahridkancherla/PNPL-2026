"""Build the word tower's input: several fixed descriptions of every word, side by side.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.build_word_descriptions

Rows are the t5 table's words in its order (`data/word_embeddings_t5_all.npz`),
so a batch's targets index this table and the t5 table alike. Blocks, each
L2-normalised per row, laid end to end:

    t5_12     the main table (t5-large's middle layer)       1024
    t5_3, t5_6, t5_18   trial 16's other layers, if built   1024 each
    phon      the sound shape (trial 18's P table), if built 1024
    letters   letter 1- to 3-grams, as the P builder's fallback  1024
    duration  the word's median spoken length, z-scored     1

Optional blocks that are missing are skipped with a note; the manifest in
the file (`blocks`, `widths`) tells the tower what it got. Writes
`data/word_descriptions.npz`. Seconds on a login node.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

from ...config import DATA_DIR, REPO_ROOT
from ..data.words import normalise_word
from ..data_processing.build_reference import (
    EVENTS_KIND_COLUMN,
    EVENTS_SEGMENT_COLUMN,
    WORD_KIND,
    discover_recordings,
)
from .build_phoneme_embeddings import form_vector
from .build_phoneme_labels import EVENTS_DURATION_COLUMN

#the optional tables: block name -> repo-relative file
OPTIONAL_TABLES = {
    "t5_3": "data/word_embeddings_t5_layer3.npz",
    "t5_6": "data/word_embeddings_t5_layer6.npz",
    "t5_18": "data/word_embeddings_t5_layer18.npz",
    "phon": "data/word_embeddings_phon.npz",
}

#the letter block's width
LETTER_DIM = 1024


def parse_args():
    """Command-line arguments for the description build."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root, for word durations")
    parser.add_argument("--table", default="data/word_embeddings_t5_all.npz", help="repo-relative; its words and order")
    parser.add_argument("--out", default="data/word_descriptions.npz", help="repo-relative output")
    parser.add_argument("--skip-durations", action="store_true", help="no event tables: a zero duration block")
    return parser.parse_args()


def unit_rows(table):
    """Each row to unit length; a zero row stays zero.

    ARGS:
        table (np.ndarray): [n, d]

    RETURNS:
        [n, d] float32
    """
    #axis=1 is the feature axis
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    return (table / np.where(norms > 0, norms, 1.0)).astype(np.float32)


def load_table(path, words):
    """One embedding table, checked against the word list.

    ARGS:
        path (Path): an .npz with `words` and `embeddings`
        words (list): the normalised words expected, in order

    RETURNS:
        [n_words, d] float32
    """
    with np.load(path, allow_pickle=False) as stored:
        these = [normalise_word(word) for word in stored["words"]]
        if these != words:
            raise SystemExit(f"{path} lists other words, or another order, than the main table.")
        return np.asarray(stored["embeddings"], dtype=np.float32)


def word_durations(data_dir, words):
    """Each word's median spoken duration in seconds, the global median where unheard.

    ARGS:
        data_dir (Path or str): the download root
        words (list): normalised words

    RETURNS:
        [n_words] float32, and how many words were heard
    """
    heard = defaultdict(list)
    for found in discover_recordings(data_dir):
        events = pd.read_csv(found["events_path"], sep="\t")
        rows = events[events[EVENTS_KIND_COLUMN] == WORD_KIND]
        for segment, duration in zip(rows[EVENTS_SEGMENT_COLUMN], rows[EVENTS_DURATION_COLUMN]):
            if np.isfinite(duration):
                heard[normalise_word(segment)].append(float(duration))
    fallback = float(np.median([value for values in heard.values() for value in values])) if heard else 0.3
    durations = np.array([np.median(heard[word]) if word in heard else fallback for word in words], dtype=np.float32)
    return durations, sum(word in heard for word in words)


def main():
    """Assemble the blocks and write the table."""
    args = parse_args()
    main_path = REPO_ROOT / args.table
    with np.load(main_path, allow_pickle=False) as stored:
        stored_words = stored["words"]
        main_table = np.asarray(stored["embeddings"], dtype=np.float32)
    words = [normalise_word(word) for word in stored_words]

    blocks = [("t5_12", unit_rows(main_table))]
    for name, relative in OPTIONAL_TABLES.items():
        path = REPO_ROOT / relative
        if path.exists():
            blocks.append((name, unit_rows(load_table(path, words))))
        else:
            print(f"[descriptions] {name}: {relative} not built; skipped")

    #letters: deterministic n-gram vectors, as the P builder's fallback
    blocks.append(("letters", np.stack([form_vector(word, "letter", LETTER_DIM) for word in words])))

    #duration: one z-scored number per word
    if args.skip_durations:
        durations, heard = np.zeros(len(words), dtype=np.float32), 0
    else:
        durations, heard = word_durations(args.data_dir, words)
    spread = float(durations.std()) or 1.0
    blocks.append(("duration", ((durations - durations.mean()) / spread).reshape(-1, 1).astype(np.float32)))

    table = np.concatenate([values for _, values in blocks], axis=1)
    names = [name for name, _ in blocks]
    widths = np.array([values.shape[1] for _, values in blocks], dtype=np.int64)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, words=stored_words, descriptions=table, blocks=np.array(names), widths=widths)
    print(f"[descriptions] {table.shape[0]:,} words x {table.shape[1]:,}: " + ", ".join(f"{n} {w}" for n, w in zip(names, widths))
          + f"; durations known for {heard:,} words -> {out_path}")


if __name__ == "__main__":
    main()
