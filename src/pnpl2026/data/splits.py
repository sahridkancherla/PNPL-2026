"""Validation splits.

Deep track is within-subject decoding on subj00, so *all* data comes from one
subject and the only meaningful held-out unit is the session. A random window
split lets neighbouring windows from the same session land in both halves;
validation then reads far above the leaderboard. Split by whole session.
"""

from __future__ import annotations

from collections import Counter


def _natural(session: str):
    """Sort key ordering numeric session ids numerically.

    LibriBrain session ids are numeric strings, so plain lexicographic order
    runs 0, 1, 10, 11, ... 2, 20 -- making "the last session" mean '9' rather
    than '28'. Fall back to string order for non-numeric ids.
    """
    return (0, int(session)) if session.isdigit() else (1, session)


def session_split(
    sessions: list[str],
    val_sessions: list[str] | None = None,
    val_fraction: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Return `(train_idx, val_idx)` splitting on whole sessions.

    `val_sessions` names the sessions to hold out explicitly. Otherwise whole
    sessions are taken from the end (natural order) until they account for at
    least `val_fraction` of the samples.

    Sessions vary enormously in size -- subj00 session '28' holds 863 windows
    against session '9's 26k -- so holding out a single fixed session gives a
    validation set whose size and class coverage are pure luck.
    """
    counts = Counter(sessions)
    unique = sorted(counts, key=_natural)
    if len(unique) < 2:
        raise ValueError(
            f"Need >=2 sessions to split, found {unique!r}. "
            "Widen the data selection before training."
        )

    if val_sessions:
        held = set(val_sessions)
        unknown = held - set(unique)
        if unknown:
            raise ValueError(f"Unknown val_sessions {sorted(unknown)}; have {unique}")
    else:
        if not 0 < val_fraction < 1:
            raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
        target = val_fraction * len(sessions)
        held, running = set(), 0
        # Take from the end so held-out sessions are contiguous and later,
        # rather than an arbitrary scatter.
        for s in reversed(unique):
            if running >= target:
                break
            held.add(s)
            running += counts[s]
        if len(held) == len(unique):
            raise ValueError(
                f"val_fraction={val_fraction} consumed every session; lower it."
            )

    train_idx = [i for i, s in enumerate(sessions) if s not in held]
    val_idx = [i for i, s in enumerate(sessions) if s in held]
    if not train_idx or not val_idx:
        raise ValueError(f"Empty split with val_sessions={sorted(held)}")
    return train_idx, val_idx
