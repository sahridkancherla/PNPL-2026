"""Trial 16: Variant K -- G2 with per-window normalisation and a second target head.

Trial 13's G2 (isolated-word training) is the parent. K changes five things:

    per-window normalisation   trial 15's `input_norm: instance`
    no baseline                the 0.5 s post-onset baseline removed
    onset jitter               +/-40 ms in training only
    weight EMA                 0.999, for validation, checkpoints and prediction
    second head                a second 256 -> 1024 map trained against an
                               early t5-large layer; predictions average the
                               two heads' row-standardised scores

K0 is K without the second head, so the two isolate its effect.

A copy of trial 15's code (2026-10-07: compliant prediction, onset-shift TTA,
lone-word ensembling) minus the word-pair fusion, which stays trial 15's. Reads
trial 10's cache and t5 tables; loads trial 10/12/13/15 checkpoints. Nothing
here imports from another trial. See `info/CONTEXT.md`.
"""
