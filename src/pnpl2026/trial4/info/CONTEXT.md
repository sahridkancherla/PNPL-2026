# trial4/ — the Conformer with a semantic head

One complete attempt: a multi-rate Conformer encoder, trained on three losses at
once. Self-contained by design; nothing here imports from another trial.

**Why this architecture, and the evidence for every choice:**
`handoff-docs/trial4-design.md`. Read that before changing anything here.

## Router

| where | what |
|---|---|
| `../models/` | the network. Its own `info/CONTEXT.md` |
| `../data/` | reads trial 1's cache: crop, jitter, session ids, class weights |
| `../losses.py` | `d_siglip_loss` and the semantic scoring path |
| `../embeddings.py` | loads and **centres** the t5 target table |
| `../lit.py` | LightningModule — three losses, warmup/cosine, weight EMA |
| `../evaluate.py` | per-class recall, beyond the headline scalar |
| `../cli/` | `build_word_embeddings`, `train`, `evaluate`, `predict` |
| `../config.py` | `Trial4Config` — shared fields plus this trial's |
| `pipeline.md` | the stage order and every shape between stages |

## The order

```
cli.build_word_embeddings  ->  cli.train  ->  cli.evaluate  ->  cli.predict
```

⚠️ `build_word_embeddings` needs outbound internet and `transformers`, which is
deliberately **not** in this project's environment — installing it could move
`torch` under a running job. Its docstring has the isolated-install recipe.

## ⚠️ What this trial is NOT

The design calls for **raw 306 sensors** and a learned spatial front end. This
trial reads trial 1's **73 whitened components**, because the continuous
recordings live on an external drive. It is the one place the implementation
departs from the design, and it is the first thing to fix when the drive is back.
