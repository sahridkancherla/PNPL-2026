# Trial 15 — router

Trial 13's code plus two independent changes: per-window input
normalisation (retrained G1/G2 as I1/I2) and a word-pair model fused with any
submission along the holdout's runs.

| want | go to |
|---|---|
| why, the evidence, the runs, how to run, what was checked | `variant-i.md` |
| the word-pair counts and the fusion maths | `../wordpairs.py` |
| the normalisation change | `../models/encoder.py` (`standardise_window`) |
| build the table / measure the fusion / fuse a CSV | `../cli/wordpairs.py`, `../cli/fuse_check.py`, `../cli/fuse.py` |
| train, score, predict, collate, ensemble | `../cli/` (trial 13's, renamed) |
| configs and the queue | `configs/trial15/`, `pbs/trial15.pbs` |

Everything else (data, cache, protocols, Lightning module) is trial 13's,
unchanged; see `src/pnpl2026/trial13/info/variant-g.md`.
