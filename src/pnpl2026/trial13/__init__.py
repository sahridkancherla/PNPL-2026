"""Trial 13: Variant G -- trial 10's F1, trained and judged under holdout conditions.

The competition holdout serves runs of 4-8 consecutive vocabulary words, each
epoch starting at the first word's onset and ending ~1.4 s after the last, plus
92 words alone with 1 s each. Trial 10 trained on whole sentences with full
3 s windows and was judged the same way, which predicted the leaderboard
poorly. Trial 13 adds a local protocol shaped like the holdout, trains with
holdout-shaped items mixed in, and ensembles seeds. A copy of trial 10's code;
it reads trial 10's cache and t5 table and can load trial 10/12 checkpoints.
Nothing here imports from another trial. See `info/CONTEXT.md`.
"""
