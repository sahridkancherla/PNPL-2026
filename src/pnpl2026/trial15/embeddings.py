"""The all-word t5 table: every word's target, and the two scored vocabularies.

d'Ascoli et al. regress onto t5-large's middle layer for EVERY word -- each
encoded alone, sub-tokens averaged -- and score by retrieval: the predicted
embedding's cosine against the candidates' embeddings. Here the candidates are
the 50 competition words (BAcc@10) and the 50 Moses words (the secondary).

⚠️ Rows are looked up by `normalise_word`, the same function the cache builder
labels with. Two spellings of one word must land on one row.
"""

from __future__ import annotations

import numpy as np
import torch

from .data.words import Vocabulary, normalise_word


class TargetTable:
    """Word -> t5 row, plus the scored vocabularies' rows in column order.

    ATTRS:
        words (list): normalised words, one per row
        table (torch.Tensor): [n_words, embed_dim] float32
        index (dict): normalised word -> row
        primary_rows (torch.Tensor): [50] rows of the competition words
        moses_rows (torch.Tensor): [50] rows of the Moses words
    """

    def __init__(self, path, centre=False):
        """
        ARGS:
            path (Path or str): the .npz from `cli.build_word_embeddings`
            centre (bool): subtract the mean row; the paper does not
        """
        try:
            stored = np.load(path, allow_pickle=False)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"No all-word table at {path}. Build it first:\n"
                f"    python -m pnpl2026.trial10.cli.build_word_embeddings --config ..."
            ) from error

        self.words = [normalise_word(word) for word in stored["words"]]
        table = np.asarray(stored["embeddings"], dtype=np.float32)

        #the row count and the word list must agree before either is trusted
        if len(self.words) != len(table):
            raise ValueError(f"{path}: {len(self.words)} words but {len(table)} rows.")
        if len(set(self.words)) != len(self.words):
            raise ValueError(f"{path} lists a word twice after normalising.")

        #a zero row has no direction, so its cosine with anything is undefined
        if not np.all(np.linalg.norm(table, axis=1) > 0):
            raise ValueError(f"{path} has zero-norm rows.")

        if centre:
            table = table - table.mean(axis=0, keepdims=True)

        self.table = torch.from_numpy(np.ascontiguousarray(table))
        self.index = {word: row for row, word in enumerate(self.words)}

        #⚠️ every scored word must have a row, or retrieval has a hole in it
        self.primary_rows = self._vocabulary_rows(Vocabulary("primary"))
        self.moses_rows = self._vocabulary_rows(Vocabulary("moses"))

    def _vocabulary_rows(self, vocabulary):
        """Rows of a vocabulary's words, in its column order.

        ARGS:
            vocabulary (Vocabulary)

        RETURNS:
            [len(vocabulary)] int64 tensor
        """
        missing = [word for word in vocabulary.words if word not in self.index]
        if missing:
            raise KeyError(f"The t5 table lacks {vocabulary.name} words {missing}; rebuild it.")
        return torch.tensor([self.index[word] for word in vocabulary.words], dtype=torch.int64)

    def rows(self, words):
        """Rows of many normalised words, -1 where a word has no row.

        ARGS:
            words (np.ndarray): [n] normalised word strings

        RETURNS:
            [n] int64 numpy array
        """
        return np.array([self.index.get(str(word), -1) for word in words], dtype=np.int64)

    @property
    def embed_dim(self):
        """The target width, 1024 for t5-large."""
        return self.table.shape[1]
