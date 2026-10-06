# Trial 16 — router

Variant K: trial 13's G2 with per-window normalisation, no baseline, onset
jitter, weight EMA and a second output head trained against an early t5
layer. K0 is K without the second head. A copy of trial 15's code
(2026-10-07) minus the word-pair fusion, which stays in trial 15.

| want | go to |
|---|---|
| why, the runs, the decision rule, how to run, what was checked | `variant-k.md` |
| the run configs and the queue | `configs/trial16/`, `pbs/trial16.pbs` |
| the second head | `../models/net.py` (`project_second`), `../lit.py` (`candidate_scores`, `combine_heads`) |
| building and choosing the second table | `pbs/trial16_embed.pbs`, `../cli/hub_check.py` |
| compliant prediction, onset shifts, lone-word ensembling | `../cli/predict.py`, `../cli/training_scaler.py`, `../cli/ensemble.py` (trial 15's) |
| train, score, collate | `../cli/` |

Everything else (data, cache, protocols, encoder) is trial 13's and 15's,
unchanged; see `src/pnpl2026/trial15/info/variant-i.md`.
