# trial13/ — Variant G: F1, trained and judged under holdout conditions

Trial 10's F1 with a holdout-shaped local protocol, holdout-shaped training
items, masked BatchNorm, and tools to ensemble and calibrate. A copy of trial
10's code; it reads trial 10's cache and t5 table and loads trial 10/12
checkpoints.

## Router

| where | what |
|---|---|
| `variant-g.md` | **start here** — why local and leaderboard disagree, what was built, the queue, how to run it |
| `../data/sequences.py` | the three protocols: reference, fragment (the holdout's shape), sparse (old) |
| `../data/dataset.py` | training-time shaping: fragments and isolated words |
| `../models/encoder.py` | masked BatchNorm; otherwise trial 10's encoder |
| `../config.py` | `Trial13Config` (a superset of trial 10's); `PROTOCOL_FIELDS` |
| `../cli/` | `train`, `evaluate`, `collate`, `calibrate`, `ensemble`, `predict` |
| `../../../../configs/trial13/` | `trial13.yaml` is G0; one file per run; `runs.txt`; `leaderboard.csv` |
| `../../../../pbs/trial13*.pbs` | the 12-line array; the calibration job |
| `../../trial10/info/` | the reference pipeline this trial builds on |

## The order

```
ensemble (now) -> qsub trial13_calibrate -> qsub -J 1-12 trial13 -> collate -> predict + ensemble
```
