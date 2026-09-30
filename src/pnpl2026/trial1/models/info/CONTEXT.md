# models/ — the trial-1 network

## Router

| where | what |
|---|---|
| `../block.py` | `EncoderBlock` (parallel branches, mixer, FFN) and `QuadratureBranch` |
| `../net.py` | `TrialOneNet` — stem, block stack, pooled head. Also `MeanPool` / `AttentionPool` |
| `../__init__.py` | the registry: `model:` string -> class |

## Shapes

The block **preserves width and length**, so it can be neither the first nor
the last thing in the network. `net.py` supplies both ends: a 1x1 stem lifting
73 components to `d_model`, and a head that collapses time then maps to 50.

## Choices worth measuring, not arguing about

| knob | default | the alternative |
|---|---|---|
| `pooling` | `mean` | `attention` — learns *where* to look, but starts near-uniform |
| `with_quadrature` | `true` | `false` is the ablation |
| `ffn_expansion` | `2` | transformers use 4; small-data may prefer 1 |
| `d_model` | `192` | past ~2x the input rank of 73 this buys little |

⚠️ At `d_model=192`, `n_blocks=4`, mean pooling: **1,285,874 parameters**.
