"""Trial 15: Variant I -- per-window input normalisation, and a word-pair model.

Two independent changes on top of trial 13's G1 and G2, whose ensembles (with
F1 and w1024) scored 66.39 on the leaderboard:

    per-window normalisation   each word's window standardised by its own
                               statistics before the encoder, so the holdout's
                               unseen recordings arrive on the training scale
    word-pair fusion           every holdout run is consecutive words that are
                               ALL in the 50-word list; a word-pair table from
                               subject 0's transcripts, combined with any
                               submission's scores along each run

A copy of trial 13's code; it reads trial 10's cache and t5 table and can load
trial 10/12/13 checkpoints. Nothing here imports from another trial. See
`info/CONTEXT.md`.
"""
