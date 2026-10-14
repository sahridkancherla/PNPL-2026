"""The per-word family (trials 10-18), embedded in trial 21.

A copy of trial 18's `config.py`, `lit.py`, `losses.py`, `embeddings.py`,
`models/` and `data/` (2026-10-11), with two import depths changed and
nothing else, so every trial 10-18 checkpoint -- G1, G2, Kn, K2, K18, N, F1,
w1024 -- loads here exactly as it does in trial 18. `holdout.py` is trial
18's `cli/predict.py` without its command line.

Trial 21 uses it to distil those checkpoints on the holdout epochs
(`..cli.distil`), to predict them (`..cli.predict`), and to warm-start the
hybrid stem from one of them (`..lit.warm_start_encoder`).

Trial 23 adds, all off by default so every earlier checkpoint behaves as
before: the anneal's knobs in `config.py` and `lit.py` (`schedule`,
`freeze_batchnorm`, `freeze_backbone`, `sam_rho`), `sam.py`, a second head
of its own width (`models/net.py`), and `evaluation.py` -- trial 18's
`cli/evaluate.py` as functions, which `..cli.evaluate` and `..cli.anneal` call.
"""
