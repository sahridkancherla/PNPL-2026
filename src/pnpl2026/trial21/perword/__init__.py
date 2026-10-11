"""The per-word family (trials 10-18), embedded in trial 21.

A copy of trial 18's `config.py`, `lit.py`, `losses.py`, `embeddings.py`,
`models/` and `data/` (2026-10-11), with two import depths changed and
nothing else, so every trial 10-18 checkpoint -- G1, G2, Kn, K2, K18, N, F1,
w1024 -- loads here exactly as it does in trial 18. `holdout.py` is trial
18's `cli/predict.py` without its command line.

Trial 21 uses it to distil those checkpoints on the holdout epochs
(`..cli.distil`), to predict them (`..cli.predict`), and to warm-start the
hybrid stem from one of them (`..lit.warm_start_encoder`).
"""
