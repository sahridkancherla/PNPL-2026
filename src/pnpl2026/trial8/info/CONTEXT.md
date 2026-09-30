# trial8/ — trial 7's Variant D, with an attention reach per sentence block

Trial 7's D1 network, plain SigLIP, no word-token masking, and one new knob:
how many words either side each sentence block may read. Self-contained; it
reads trial 7's continuous cache and has no builder of its own.

## Router

| where | what |
|---|---|
| `reach.md` | **start here** — the idea, the queue, parameters, what was checked |
| `architecture.md` | the network drawn end to end, the masks, how reach compounds |
| `../models/` | the network. Its own `info/CONTEXT.md` |
| `../data/` | unchanged from trial 7: holdout-shaped sequences, flat token batches |
| `../losses.py`, `../lit.py` | unchanged from trial 7 |
| `../config.py` | `Trial8Config`; checks `sentence_reach` against the depth at load |
| `../cli/` | `train`, `collate` (every seed against E0), `predict` (two modes) |
| `../../../../configs/trial8/` | `trial8.yaml` is E0's base; one file per run; `runs.txt` |
| `../../../../pbs/trial8.pbs` | the array job, 12 lines |
| `../../trial7/info/` | the pipeline, shapes and rules question — all still true here |

## The order

```
trial 7's cache on RDS -> qsub -J 1-12 pbs/trial8.pbs -> cli.collate -> cli.predict
```

⚠️ As in trial 7, the default `predict` mode uses holdout MEG outside each
word's 1 s window, which the rules do not settle. Reach only matters there:
under `--window-only` every word is alone and all four runs are identical.
