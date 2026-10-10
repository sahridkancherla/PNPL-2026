# Trial 20 — router

The continuous family from its best member, SPW (trial 19's +0.68), with one
change per run: a word transformer after the readout (H), a 2.5 s readout
(R), time masking (M), and SPW's seeds 3-5 (SPW+). Plus two tools that need
no GPU: family-weighted mixes and a by-corpus calibration of local scores
against the board. Built 2026-10-11 from the second architecture note, on a
copy of trial 19's code.

| want | go to |
|---|---|
| the note, claim by claim; each run; the tools; the decision rule; commands; checks; results | `variants.md` |
| the word transformer (`word_depth`, `word_grid`) | `../models/continuous.py` |
| family-weighted ensembles (`--group`, `--weights`) | `../cli/ensemble.py` |
| local scores against the board, corpus by corpus | `../cli/calibrate_tasks.py` |
| configs and the queue | `configs/trial20/`, `pbs/trial20.pbs` |

Everything else -- the stem, the readout, the tower, the phoneme head, the
data, prediction -- is trial 19's: see `src/pnpl2026/trial19/info/CONTEXT.md`.
