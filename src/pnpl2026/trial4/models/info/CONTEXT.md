# models/ — the trial-4 network

## Router

| where | what |
|---|---|
| `../norms.py` | `BiasNorm`, `SwooshR`/`SwooshL`, `DropPath` — Zipformer's pieces |
| `../frontend.py` | `RobustInstanceNorm`, `FixedLowpassDecimate` — fixed, not learned |
| `../conformer.py` | `ConformerBlock`, `ConvolutionModule` (with the quadrature branch), `MultiHeadSelfAttention` (rotary, reusable weights) |
| `../encoder.py` | `MultiRateEncoder` — the U-Net of stacks. `SimpleDownsample`/`SimpleUpsample` |
| `../heads.py` | `AttentionPool`, `SemanticHead`, `ClosedSetHead`, `SessionAdversary` |
| `../net.py` | `TrialFourNet` — front end, encoder, three heads |
| `../__init__.py` | the registry: `model:` string -> class |

## ⚠️ Two layouts, one transpose

| where | layout |
|---|---|
| `frontend.py`, inside `ConvolutionModule` | **channels first** `[B, C, T]` |
| everything else | **channels last** `[B, T, D]` |

`net.py` owns the single boundary transpose, after the front end. `BiasNorm`
handed a `[B, D, T]` tensor normalises over TIME instead of features and does
not raise.

## Shapes, at the real configuration

```
[B, 73, 250]  ->  instance norm  ->  decimate 2  ->  [B, 73, 125]
              ->  stacks at 125 / 63 / 32 / 63 frames  ->  [B, 63, 192]
              ->  pool  ->  [B, 192]  ->  50 logits + a 1024-d embedding
```

## Choices worth measuring, not arguing about

| knob | default | the alternative |
|---|---|---|
| `with_quadrature` | `true` | `false` is the ablation |
| `reuse_attention` | `true` | `false` gives every block its own weights |
| `with_instance_norm` | `true` | `false` trusts the cache's fitted stats |
| `pooling` | `attention` | `mean` — no parameters, no failure mode |
| `stack_divisors` | `[1,2,4,2]` | `[1,1,1,1]` is a flat Conformer |
| `decimate_factor` | `2` | `5` reaches the reference model's 50 Hz, but needs `lowpass_hz` under 25 |

⚠️ Kernels shrink as the frame rate drops so each spans ~250 ms. A constant
kernel across a rate change quadruples the receptive field per stack.

⚠️ At the default config: **10,549,287 parameters**.
