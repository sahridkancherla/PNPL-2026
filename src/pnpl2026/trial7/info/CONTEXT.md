# trial7/ — Variant D, matched to the holdout

A per-word encoder, then a transformer across the words of a sentence, trained
on windows and sentences shaped like the holdout's. Self-contained by design;
nothing here imports from another trial.

## Router

| where | what |
|---|---|
| `variant-d.md` | **start here** — preconditions, the queue, departures from the design, the rules question |
| `pipeline.md` | stage order and every shape |
| `../data_processing/` | `build_continuous` — whole recordings, whitened, plus word tables |
| `../data/` | holdout-shaped sequences, windows sliced on read, flat token batches |
| `../models/` | the per-word encoder and the network. Its own `info/CONTEXT.md` |
| `../losses.py` | SigLIP, deduplicated, normalised by N |
| `../lit.py` | LightningModule — per-word losses, gradient probe, split metrics |
| `../config.py` | `Trial7Config`, and the key-by-key `model_kwargs` loader |
| `../cli/` | `build_continuous`, `train`, `predict` (two modes), `build_word_embeddings` |
| `../../../../configs/trial7/` | `trial7.yaml` is D1; one file per run; `runs.txt` pairs runs with seeds |
| `../../../../pbs/` | `trial7_build.pbs` once, then `trial7.pbs` as arrays |

## The order

```
cli.prepare -> pbs/trial7_build.pbs -> pbs/trial7.pbs -> cli.predict [--window-only]
```

⚠️ The default `predict` mode uses holdout MEG outside each word's 1 s window.
The rules do not say whether that is allowed; `--window-only` is the safe mode.
