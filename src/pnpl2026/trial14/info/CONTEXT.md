# trial14/ — Variant H: G2 plus a trained 50-word head, on holdout-shaped chunks

Trial 13's G2 with a 50-way head blended into retrieval, fine-tuned from G2's
checkpoints, and a new local protocol -- chunked -- that scores every test word
in the holdout's geometry. A copy of trial 13's code; it reads trial 10's cache
and t5 table and loads trial 10/12/13 checkpoints.

## Router

| where | what |
|---|---|
| `variant-h.md` | **start here** — both parts, the queue, how to run it, what was checked |
| `../lit.py` | the head, its logit-adjusted loss, frozen mode, loading from G2 |
| `../losses.py` | `logit_adjusted_cross_entropy`, `blend_scores` |
| `../data/sequences.py` | the four protocols; `chunked_sequences` is new |
| `../config.py` | `Trial14Config` (a superset of trial 13's); `PROTOCOL_FIELDS` |
| `../cli/` | `train`, `evaluate`, `collate`, `calibrate`, `ensemble`, `predict` |
| `../../../../configs/trial14/` | `trial14.yaml` is H1; one file per run; `runs.txt`; `ensembles.csv` |
| `../../../../pbs/trial14*.pbs` | the 9-line array; the calibration job |
| `../../trial13/info/variant-g.md` | G2, and why the fragment protocol was too small |

## The order

```
qsub trial14_calibrate + qsub -J 1-9 trial14 -> collate -> predict -> ensemble
```
