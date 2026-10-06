# Trial 15 — router

Trial 13's code plus per-window input normalisation (I1/I2), a lone-word
specialist (I3), compliant prediction (fixed training scaling, onset shifts),
and the abandoned word-pair fusion.

| want | go to |
|---|---|
| why, the evidence, the runs, how to run, what was checked, results | `variant-i.md` |
| compliant prediction: fixed scaler, onset shifts | `../cli/training_scaler.py`, `../cli/predict.py` |
| the lone-word specialist: isolated protocol, routing | `../data/sequences.py`, `../config.py`, `../cli/ensemble.py` |
| the normalisation change | `../models/encoder.py` (`standardise_window`) |
| the word-pair fusion (abandoned) | `../wordpairs.py`, `../cli/wordpairs.py`, `../cli/fuse_check.py`, `../cli/fuse.py` |
| train, score, collate | `../cli/` (trial 13's, renamed) |
| configs and the queue | `configs/trial15/`, `pbs/trial15.pbs` |

The holdout principle these tools follow is in CLAUDE.md. Everything else
(data, cache, Lightning module) is trial 13's; see
`src/pnpl2026/trial13/info/variant-g.md`.
