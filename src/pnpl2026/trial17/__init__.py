"""Trial 17: Variant L -- G2 trained on holdout-matched runs.

Trial 13's G2 (isolated-word training) is the parent; L changes one thing.
Half of each epoch's training sentences are served as HOLDOUT RUNS, built the
way the competition holdout builds its runs:

    tokens       scored (list) words only -- the holdout gives no token for an
                 unscored word
    neighbours   directly adjacent, or with ONE unscored word between them
                 (about a third of the holdout's pairs are), and never further
                 apart than the holdout's longest gaps
    length       drawn from the holdout's run-length histogram
    MEG          stopping 1.13-1.68 s after the last onset, as the holdout's

Every token keeps its t5 target, so no supervision is lost. The public
leaderboard is made of these runs only (it holds no lone words), so this is
where public points are.

A copy of trial 16's code (compliant prediction, onset-shift TTA, lone-word
ensembling, the second-head switch -- off here) without its layer check.
Reads trial 10's cache and t5 table; loads trial 10/12/13/15/16 checkpoints.
Nothing here imports from another trial. See `info/CONTEXT.md`.
"""
