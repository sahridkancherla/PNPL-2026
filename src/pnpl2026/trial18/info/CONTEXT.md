# Trial 18 — router

Variant M: HDND's refinement (arXiv 2609.24095) trained on a frozen
checkpoint, G2's first. A copy of trial 17's code (2026-10-08); with
`refine_base` unset it runs exactly as trial 17.

| want | go to |
|---|---|
| why, the design, the departures from the paper, the decision rule, how to run, what was checked | `variant-m.md` |
| the refiner: pyramid pooling, query, bounded residual | `../models/refine.py` |
| where it sits in the network; the conv-layer taps | `../models/net.py`, `../models/encoder.py` (`features`) |
| the extra losses, freezing the base, calibration, the strength curve | `../lit.py` |
| a refinement run's settings: the base's, plus the run's own fields | `../config.py` (`REFINE_RUN_FIELDS`, `refinement_config`) |
| training on a frozen base; the strength written into best.ckpt | `../cli/train.py` |
| predicting, and `--refine-strength` | `../cli/predict.py` |
| configs and the queue | `configs/trial18/`, `pbs/trial18.pbs` |

Everything else (hub correction, lone words, the screen) is trial 17's,
copied; see `src/pnpl2026/trial17/info/CONTEXT.md`.
