"""Load the fixed table of target word embeddings.

The semantic head regresses onto a language model's representation of the word
rather than onto a one-hot class. That table is built once, offline, by
`cli.build_word_embeddings`, and loaded here.

⚠️ Row order is the whole contract. Row `i` must be the embedding of
`vocab.PRIMARY[i]`, the competition's own frequency-ordered vocabulary. If the
table is built in a different order the model trains, the loss descends,
validation looks plausible, and every prediction names the wrong word -- the
same silent failure `data/vocab.py` exists to prevent, one layer further out.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data import vocab


def mean_off_diagonal_cosine(table):
    """Average cosine between distinct rows -- how separable the targets are.

    ARGS:
        table (np.ndarray): [n_rows, embed_dim]

    RETURNS:
        float: mean cosine over every ordered pair of distinct rows
    """
    unit = table / np.linalg.norm(table, axis=1, keepdims=True)
    similarity = unit @ unit.T
    return float(similarity[~np.eye(len(table), dtype=bool)].mean())


def load_target_embeddings(path, n_classes=None, centre=True):
    """Read the embedding table and check it against the scored vocabulary.

    ⚠️ `centre` is on by default and it matters more than it looks. Language
    model hidden states are strongly anisotropic -- they occupy a narrow cone,
    not the whole sphere. Measured on t5-large's middle layer for these 50
    words, the mean cosine between two DIFFERENT words is **0.709**, and the
    closest pair sits at 0.927. A contrastive loss into that space is being
    asked to separate targets that are already nearly parallel, and most of the
    gradient goes into reproducing the shared mean direction, which carries no
    information about which word it was.

    Subtracting the mean row fixes it: the shared direction is exactly what the
    mean is, and removing it leaves only what distinguishes the words. This is
    the standard treatment for LM embedding anisotropy.

    The reference model did not need this because it trained on an OPEN
    vocabulary of thousands of word types, where the cone is populated densely
    enough for the structure to be informative. With 50 fixed targets it is not.

    ARGS:
        path (Path or str): the .npz written by `cli.build_word_embeddings`
        n_classes (int or None): expected row count. None skips that check
        centre (bool): subtract the mean row. False is the ablation

    RETURNS:
        table : [n_classes, embed_dim] float32 tensor, rows in vocabulary order
    """
    #say how to make the file, rather than leaving a bare FileNotFoundError
    try:
        stored = np.load(path, allow_pickle=False)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"No word-embedding table at {path}. Build it first:\n"
            f"    python -m pnpl2026.trial6.cli.build_word_embeddings\n"
            f"Or set semantic_weight: 0.0 to train the closed-set head alone."
        ) from error

    for key in ("words", "embeddings"):
        if key not in stored:
            raise KeyError(
                f"{path} has no {key!r} array; found {sorted(stored.files)}. "
                f"Rebuild it with cli.build_word_embeddings."
            )

    words = [str(w) for w in stored["words"]]
    table = np.asarray(stored["embeddings"], dtype=np.float32)

    #the row count and the word list must agree before either is trusted
    if len(words) != len(table):
        raise ValueError(
            f"{path}: {len(words)} words but {len(table)} embedding rows."
        )

    if n_classes is not None and len(table) != n_classes:
        raise ValueError(
            f"{path} holds {len(table)} rows, expected {n_classes}."
        )

    #⚠️ the check that makes the contract real: row i IS vocab.PRIMARY[i]
    expected = [w.lower() for w in vocab.PRIMARY]
    if [w.lower() for w in words] != expected[: len(words)]:
        mismatch = next(
            index
            for index, (got, want) in enumerate(zip(words, expected))
            if got.lower() != want
        )
        raise ValueError(
            f"{path} row {mismatch} is {words[mismatch]!r}, but the competition "
            f"vocabulary has {expected[mismatch]!r} there. The table is in the "
            f"wrong order and every prediction would name the wrong word."
        )

    #a zero row would make its cosine similarity undefined, and the class would
    #be unreachable rather than merely badly represented
    norms = np.linalg.norm(table, axis=1)
    if not np.all(norms > 0):
        dead = [words[i] for i in np.flatnonzero(norms == 0)]
        raise ValueError(f"{path} has zero-norm embeddings for: {dead}")

    before = mean_off_diagonal_cosine(table)

    if centre:
        #axis=0 is the WORD axis: this is the mean word, the shared direction
        #every row has in common, and it is the part that carries no information
        table = table - table.mean(axis=0, keepdims=True)

        #centring can only produce a zero row if two words were identical
        norms = np.linalg.norm(table, axis=1)
        if not np.all(norms > 0):
            dead = [words[i] for i in np.flatnonzero(norms == 0)]
            raise ValueError(f"Centring collapsed these words to zero: {dead}")

    after = mean_off_diagonal_cosine(table)
    print(
        f"[embed] {len(table)} targets, dim {table.shape[1]}, mean off-diagonal "
        f"cosine {before:.3f}"
        + (f" -> {after:.3f} after centring" if centre else " (uncentred)")
    )

    return torch.from_numpy(np.ascontiguousarray(table, dtype=np.float32))
