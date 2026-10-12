"""Trial 22: the Route to 73, with the holdout open to any use.

The changes of `handoff-docs/route-to-73-2026-10-12.md`, built after the
organisers confirmed (12 Oct) that the holdout data may be used in any way:

    1   other subjects' recordings of the same sentences: a repeat check
        (`cli/repeat_check.py`), votes from the multi-subject model
        (`cli/predict.py --subject`), MCCA alignment into subject 0's sensor
        space (`cli/align.py`), combined by `cli/cross_votes.py`
    2   transductive inference over the holdout rows (`cli/transduce.py`),
        an embedding dump (`cli/embed.py`) and a kNN memory voter
        (`cli/knn_vote.py`)
    4   per-word correction variants (`cli/correct.py`)
    6   an onset-gap voter (`cli/gap_voter.py`)

Changes 3 and 5 reuse existing tools. A copy of trial 21's code (both
families, prediction, ensembling, hub correction, lone-word tools).
Nothing here imports from another trial. See `info/CONTEXT.md`.
"""
