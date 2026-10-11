# Trial 21 — router

Train on the holdout's own session and format. Part A distils trained
members of either family on the holdout epochs with the best mix as teacher;
part B is the hybrid, the continuous family with the per-word encoder as its
stem, warm-started from G2. Built 2026-10-11 from
`handoff-docs/architecture-review-2026-10-10.md`, on a copy of trial 20's
code plus trial 18's per-word family under `perword/`.

| want | go to |
|---|---|
| why, what each part does, the decision rules, the commands, what was checked, results | `holdout-adaptation.md` |
| the distillation loop | `../cli/distil.py` |
| the hybrid stem (`stem: dascoli`) and its warm start | `../models/continuous.py`, `../lit.py` (`warm_start_encoder`) |
| which family a checkpoint is, one loader for both | `../families.py` |
| prediction for either family, `--bn-adapt` for both | `../cli/predict.py`, `../perword/holdout.py` |
| the per-word family's code (trial 18's, two import depths changed) | `../perword/` |
| random marker dropout | `../data/dataset.py` (`marker_drop_mode`) |
| configs, the run list, the student list | `configs/trial21/`, `pbs/trial21.pbs`, `pbs/trial21_distil.pbs` |

Everything else -- the megstate stem, the readout, the tower, the phoneme
head, the data, the ensemble and hub-correction tools -- is trial 20's: see
`src/pnpl2026/trial20/info/CONTEXT.md`.
