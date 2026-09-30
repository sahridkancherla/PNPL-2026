# trial1/ — the whitened 73-component attempt

One complete run: cache, model, training, evaluation. Self-contained by
design; nothing here imports from another trial.

## Router

| where | what |
|---|---|
| `../data_processing/` | builds the cache. Its own `info/CONTEXT.md` |
| `../models/` | the network. Its own `info/CONTEXT.md` |
| `../data/` | reads the cache: crop, jitter, class weights |
| `../lit.py` | LightningModule — loss, optimiser, BAcc logging |
| `../evaluate.py` | per-class recall report, beyond the headline scalar |
| `../cli/` | `build_cache`, `train`, `evaluate` |
| `../config.py` | `Trial1Config` — shared fields plus this trial's |
| `pipeline.md` | the stage order and every shape between stages |

## The order

```
cli.build_cache  ->  cli.train  ->  cli.evaluate
```

⚠️ `build_cache` needs `pnpl2026.cli.prepare` to have run first, on a machine
with outbound internet. It writes ~18 GB.
