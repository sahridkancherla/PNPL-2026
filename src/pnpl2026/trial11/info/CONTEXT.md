# trial11/ — one stream per sentence, a Zipformer-style multi-rate encoder

Each sentence stays one timeline: trial 7's front end and trunk run over it
once, six stacks at 50/25/12.5/6.25/12.5/25 Hz work across the whole sentence,
and every word is read off the shared timeline at the end. Self-contained;
reads trial 7's continuous cache and has no builder.

## Router

| where | what |
|---|---|
| `stream-zipformer.md` | **start here** — the idea, the network drawn, the queue, parameters, what was checked |
| `../models/` | the network. Its own `info/CONTEXT.md` |
| `../data/` | sentences cut by count AND onset span; one padded stream per sequence |
| `../losses.py`, `../lit.py` | trial 8 E0's losses; `lit.py` passes streams instead of windows |
| `../config.py` | `Trial11Config`; checks span + tail fit the stream, and every stack factor tiles it |
| `../cli/` | `train`, `collate` (Z1 against Z0), `predict` (sentences as streams; `--window-only`) |
| `../../../../configs/trial11/` | `trial11.yaml` is Z1's base; `z1_multirate.yaml`, `z0_flat.yaml`; `runs.txt` |
| `../../../../pbs/trial11.pbs` | the array job, 6 lines |
| `../../trial7/info/` | the holdout measurements and the rules question — still true here |

## The order

```
trial 7's cache on RDS -> qsub -J 1-6 pbs/trial11.pbs -> cli.collate -> cli.predict
```

⚠️ As in trials 7–9, the default `predict` mode reads holdout MEG outside
each word's 1 s window; `--window-only` is the rules-safe mode.
