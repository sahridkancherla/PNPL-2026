# trial6/ — Variant C, as an ablation queue

Trial 5's network with Variant C's switches, run as eleven one-change
comparisons against C0, which is trial 5 itself. Self-contained by design;
nothing here imports from another trial.

## Router

| where | what |
|---|---|
| `variant-c.md` | **start here** — the queue, how to submit it, departures from the design, what is not built |
| `pipeline.md` | stage order and every shape, both streams |
| `../models/` | the network, including stream B. Its own `info/CONTEXT.md` |
| `../data/` | trial 5's cache reader, plus shuffled labels and an unjittered `fit_set` |
| `../losses.py` | D-SigLIP, with C7's normalisation switch |
| `../lit.py` | LightningModule — three losses, warmup/cosine, parameter EMA |
| `../config.py` | `Trial6Config`, and a loader that merges `model_kwargs` key by key |
| `../cli/` | `train`, `evaluate`, `predict`, `session_probe` (C9), `build_word_embeddings` |
| `../../../../configs/trial6/` | `trial6.yaml` is C0; one file per run; `runs.txt` is the queue |
| `../../../../pbs/trial6.pbs` | array job over `runs.txt`, and C6's own line |

## The order

```
cli.build_word_embeddings  ->  pbs/trial6.pbs  ->  cli.session_probe (C9)
                           ->  cli.evaluate    ->  cli.predict
```

⚠️ C8 is **not** built — it needs word audio and an all-words cache. See
`variant-c.md`.
