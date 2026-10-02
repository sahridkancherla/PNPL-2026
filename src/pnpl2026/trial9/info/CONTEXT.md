# trial9/ — trial 7's Variant D, with a 0.5 Hz high-pass on every window

Trial 8's E0 network (trial 7 D1, plain SigLIP, no word masking) with one
fixed addition at the very front: a zero-phase high-pass on each window.
Self-contained; reads trial 7's continuous cache and has no builder.

## Router

| where | what |
|---|---|
| `highpass.md` | **start here** — why, how it works on short windows, the queue, what was checked |
| `../models/` | the network. Its own `info/CONTEXT.md` |
| `../data/`, `../losses.py`, `../lit.py` | unchanged from trial 7 |
| `../config.py` | `Trial9Config` |
| `../cli/` | `train`, `collate` (H1 against H0), `predict` (two modes) |
| `../../../../configs/trial9/` | `trial9.yaml` is H0's base; `h1_highpass_05.yaml`; `runs.txt` |
| `../../../../pbs/trial9.pbs` | the array job, 6 lines |
| `../../trial7/info/` | pipeline, shapes, the rules question — all still true here |
| `../../trial8/info/` | the reach knob, still present and off |

## The order

```
trial 7's cache on RDS -> qsub -J 1-6 pbs/trial9.pbs -> cli.collate -> cli.predict
```

Unlike trial 8's reach, the filter acts in `--window-only` mode too: it is the
first trial since 7 whose change can move the rules-safe score.
