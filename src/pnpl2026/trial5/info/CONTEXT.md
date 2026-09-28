# trial5/ — conv trunk plus a flat transformer

One complete attempt, **variant B**: a dilated convolution trunk feeding a flat
transformer that holds 79% of the 8.0M parameters. Self-contained by design;
nothing here imports from another trial.

The bet, in one line: trial 2 scored 34.70% with a sequence stage of ~0.6M,
while the 73.23% reference puts 98.8% of 203.7M into its transformer. This is
the first run that spends the parameters where the reference spends them.

⚠️ **Variant A is not built.** A learned `ChannelMerger` over raw 306 sensors
and their geometry needs the external drive, for the continuous recordings and
the sensor positions. `spatial_width: 270` matches the reference's merger so
that variant A can replace that one stage without touching anything else.

## Router

| where | what |
|---|---|
| `../models/` | trunk, transformer, heads. Its own `info/CONTEXT.md` |
| `../data/` | reads trial 1's cache: crop, jitter, session ids, class weights |
| `../losses.py` | `d_siglip_loss` and the semantic scoring path |
| `../embeddings.py` | loads and **centres** the t5 target table |
| `../lit.py` | LightningModule — three losses, warmup/cosine, weight EMA |
| `../../../../configs/trial5.yaml` | the run config, with the reasoning per knob |
| `../evaluate.py` | per-class recall, beyond the headline scalar |
| `../cli/` | `build_word_embeddings`, `train`, `evaluate`, `predict` |
| `../config.py` | `Trial5Config` — shared fields plus this trial's |
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
