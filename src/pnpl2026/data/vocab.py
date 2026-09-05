"""The competition's 50-word label space.

`WordClassification` numbers its classes as `sorted(set(every word in the
dataset))` -- thousands of classes, alphabetical. The competition scores 50
classes in `PRIMARY_VOCAB` order, which is frequency-ordered, not alphabetical.

Training on the task's native ids and submitting those columns fails silently:
the model trains, validation looks fine, and the submission scores at chance
because column 0 means different words to you and to the scorer. Every target
in this project comes from `to_id` below, never from `task.label_info`.
"""

from __future__ import annotations

from pnpl.competition import load_vocabulary

#: 50 words in competition column order.
PRIMARY: list[str] = load_vocabulary("primary")
N_CLASSES: int = len(PRIMARY)

WORD_TO_ID: dict[str, int] = {w.lower(): i for i, w in enumerate(PRIMARY)}


def is_target(word: object) -> bool:
    """True if `word` is one of the 50 scored words."""
    return isinstance(word, str) and word.lower() in WORD_TO_ID


def to_id(word: str) -> int:
    """Map a word to its competition column index."""
    return WORD_TO_ID[word.lower()]
