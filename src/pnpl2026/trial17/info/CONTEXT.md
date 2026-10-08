# Trial 17 — router

Variant L: trial 13's G2 trained with half its sentences served as holdout
runs -- scored words only, one skipped word allowed, the MEG cut after the
last onset. A copy of trial 16's code (2026-10-07) minus its layer check.

| want | go to |
|---|---|
| why, the one change, the decision rule, how to run, what was checked | `variant-l.md` |
| finding a sentence's holdout-like chains | `../data/sequences.py` (`holdout_chains`) |
| serving a holdout run | `../data/dataset.py` (`Shaping.holdout_run`, `Shaping.choose`) |
| the new settings | `../config.py` (`run_p`, `run_max_skipped`) |
| compliant prediction, onset shifts, lone-word ensembling | `../cli/predict.py`, `../cli/training_scaler.py`, `../cli/ensemble.py` |
| hub correction (querybank normalisation, no training) | `hub-correction.md`, `../hubnorm.py`, `../cli/hubnorm.py` |
| hub correction fitted on the holdout's own rows (transductive) | `hub-correction.md`, `../cli/hubnorm_holdout.py` |
| which recipe fills the lone-word rows (private board only; local comparison) | `../cli/lone_check.py`; results in `hub-correction.md` |
| write the chosen lone-word recipe into a submission | `../cli/lone_fill.py` |
| screen models before submitting: holdout vs local agreement with G2 | `../cli/screen.py` |
| configs and the queue | `configs/trial17/`, `pbs/trial17.pbs` |

Everything else (network, cache, protocols) is trial 16's, unchanged; see
`src/pnpl2026/trial16/info/variant-k.md`.
