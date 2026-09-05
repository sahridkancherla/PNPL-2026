"""Dataset wrapper restricting LibriBrain to the competition label space."""

from __future__ import annotations

from torch.utils.data import Dataset

from .vocab import is_target, to_id

# Index positions in a LibriBrain sample tuple:
# (subject, session, task, run, onset, word, sent_idx, word_idx)
_SESSION = 1
_WORD = 5


class CompetitionWordDataset(Dataset):
    """Keep only samples whose word is scored, and relabel to competition ids.

    Wraps rather than mutates the underlying dataset so the base object stays
    reusable and the filtering is inspectable.
    """

    def __init__(self, base):
        self.base = base
        self.indices: list[int] = [
            i for i, s in enumerate(base.samples) if is_target(s[_WORD])
        ]
        if not self.indices:
            raise ValueError(
                "No samples matched the competition vocabulary -- check that the "
                "underlying task emits raw words in sample[5]."
            )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        j = self.indices[i]
        data, _ = self.base[j]
        return data, to_id(self.base.samples[j][_WORD])

    def sessions(self) -> list[str]:
        """Session id per kept sample, aligned with __getitem__ order."""
        return [str(self.base.samples[j][_SESSION]) for j in self.indices]

    def labels(self) -> list[int]:
        """Competition label per kept sample, aligned with __getitem__ order."""
        return [to_id(self.base.samples[j][_WORD]) for j in self.indices]
