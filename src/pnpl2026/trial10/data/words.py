"""Word strings: one normalisation, and the two scored vocabularies.

Trial 10 trains on EVERY word, so word strings matter more than in any trial
before it: each one is looked up in the t5 table, and every in-vocabulary one
also needs its competition column.

⚠️ One normalisation for both sides, always. The competition's primary
vocabulary spells `it’s` with a CURLY apostrophe. The shared `data/vocab.py`
only lowercases, so a transcript writing `it's` with a straight one never
matches it -- the tutorial unifies the two, and so does `normalise_word`.

⚠️ Columns stay in `load_vocabulary` order. Normalising a vocabulary word
changes its spelling, never its index: column i is still PRIMARY[i].
"""

from __future__ import annotations

from pnpl.competition import load_vocabulary

#apostrophe-like characters a transcript or the vocabulary may use for '
APOSTROPHES = ("’", "‘", "ʼ", "`")

#stripped from either END of a word, never from inside: "it's" keeps its '
EDGE_PUNCTUATION = "\"'.,;:!?()[]{}<>-_*~"


def normalise_word(word):
    """The one spelling a word is compared and looked up in.

    ARGS:
        word (object): a transcript segment or a vocabulary entry

    RETURNS:
        str: lowercased, apostrophes unified, edge punctuation stripped. May
        be empty for a segment that was punctuation only
    """
    text = str(word).strip().lower()
    for apostrophe in APOSTROPHES:
        text = text.replace(apostrophe, "'")
    return text.strip(EDGE_PUNCTUATION).strip()


class Vocabulary:
    """A scored 50-word vocabulary, as normalised word -> column.

    ATTRS:
        name (str): "primary" or "moses"
        words (list): normalised words in column order
        index (dict): normalised word -> column
    """

    def __init__(self, name):
        """
        ARGS:
            name (str): pnpl's vocabulary name, "primary" or "moses"
        """
        self.name = name
        self.words = [normalise_word(word) for word in load_vocabulary(name)]

        #a collision after normalising would silently merge two columns
        if len(set(self.words)) != len(self.words):
            raise ValueError(f"Normalising the {name} vocabulary merges words: {self.words}")
        self.index = {word: column for column, word in enumerate(self.words)}

    def label(self, word):
        """The column of a normalised word, or -1 when it is not scored.

        ARGS:
            word (str): already normalised

        RETURNS:
            int
        """
        return self.index.get(word, -1)

    def __len__(self):
        """Words in the vocabulary, 50 for both."""
        return len(self.words)
