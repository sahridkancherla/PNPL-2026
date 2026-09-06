"""Validation splits.

Deep track is within-subject decoding on subj00, so *all* data comes from one
subject and the only meaningful held-out unit is the session. A random window
split lets neighbouring windows from the same session land in both halves;
validation then reads far above the leaderboard. Split by whole session.
"""

from __future__ import annotations


def _natural(session: str):
    """Sort key ordering numeric session ids numerically.

    LibriBrain session ids are numeric strings, so plain lexicographic order
    runs 0, 1, 10, 11, ... 2, 20 -- making "the last session" mean '9' rather
    than '28'. Fall back to string order for non-numeric ids.
    """
    return (0, int(session)) if session.isdigit() else (1, session)


def session_split(
    sessions: list[str], val_sessions: list[str] | None = None
) -> tuple[list[int], list[int]]:
    """Return `(train_idx, val_idx)` splitting on whole sessions.

    `val_sessions` names the sessions to hold out; when omitted the
    last session in sorted order is used, which keeps the split
    deterministic without needing the config to know session names.
    """
    unique = sorted(set(sessions), key=_natural)
    if len(unique) < 2:
        raise ValueError(
            f"Need >=2 sessions to split, found {unique!r}. "
            "Widen the data selection before training."
        )

    held = set(val_sessions) if val_sessions else {unique[-1]}
    unknown = held - set(unique)
    if unknown:
        raise ValueError(f"Unknown val_sessions {sorted(unknown)}; have {unique}")

    train_idx = [i for i, s in enumerate(sessions) if s not in held]
    val_idx = [i for i, s in enumerate(sessions) if s in held]
    if not train_idx or not val_idx:
        raise ValueError(f"Empty split with val_sessions={sorted(held)}")
    return train_idx, val_idx
