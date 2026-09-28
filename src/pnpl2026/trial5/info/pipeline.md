# The trial-5 pipeline — stage order and every shape between stages

Measured from the built model, not read off the design. Reproduce with:

```bash
PYTHONPATH=src .venv/bin/python -c "
from pnpl2026.trial5.models import TrialFiveNet
net = TrialFiveNet(n_channels=73, n_classes=50, n_sessions=28, embed_dim=1024)
print(net.n_frames, net.trunk.dilations, net.trunk.reach,
      sum(p.numel() for p in net.parameters()))"
```

## Forward

| stage | in | out | learned |
|---|---|---|---|
| cache read + crop | `[B, 73, 400]` | `[B, 73, 250]` | no |
| `RobustInstanceNorm` | `[B, 73, 250]` | `[B, 73, 250]` + `[B, 73]` | no |
| `BaselineCorrect` | `[B, 73, 250]` | `[B, 73, 250]` | no |
| `FixedLowpassDecimate` | `[B, 73, 250]` | `[B, 73, 50]` | no |
| `ComponentDropout` + `TimeMask` | `[B, 73, 50]` | `[B, 73, 50]` | no — **training only** |
| `spatial` (variant B) | `[B, 73, 50]` | `[B, 270, 50]` | yes |
| `initial_linear` | `[B, 270, 50]` | `[B, 512, 50]` | yes |
| `DilatedConvTrunk` | `[B, 512, 50]` | `[B, 256, 50]` | yes |
| `.transpose(1, 2)` | `[B, 256, 50]` | `[B, 50, 256]` | — |
| `FlatTransformer` | `[B, 50, 256]` | `[B, 50, 256]` | yes |
| `AttentionPool` | `[B, 50, 256]` | `[B, 256]` | yes |
| `ClosedSetHead` | `[B, 256]` | `[B, 50]` | yes |
| `SemanticHead` | `[B, 256]` | `[B, 1024]` | yes |
| `SessionAdversary` | `[B, 256]` | `[B, n_sessions]` | yes |

`scale_projection` runs off the norm's `[B, 73]` side-channel to `[B, 256]` and
is added to every frame inside the transformer, so the amplitude the instance
norm discarded is available without setting the input range.

## The trunk, layer by layer

Frame rate 50 Hz throughout, so one frame is 20 ms and the length never changes.

| layer | conv | dilation | reach | residual | GLU after |
|---|---|---|---|---|---|
| 1 | 512 → 160 | 1 | 3 fr / 60 ms | 1×1 projection | no |
| 2 | 160 → 160 | 2 | 7 fr / 140 ms | identity | **yes** |
| 3 | 160 → 160 | 4 | 15 fr / 300 ms | identity | no |
| 4 | 160 → 160 | 8 | 31 fr / 620 ms | identity | **yes** |
| 5 | 160 → 256 | **1** | **33 fr / 660 ms** | 1×1 projection | no |

Layer 5's dilation resets because `dilation_period = 4`. Without the reset it
would run at 16 and reach 63 frames — past the 50-frame window, so most output
positions would read a third padding. `DilatedConvTrunk` raises
`ReceptiveFieldError` rather than allowing it.

Final reach is **66% of the window**, against the reference model's 42% of its
own. Both leave global integration to the transformer, which is the point.

## Parameters

| group | params | share |
|---|---|---|
| front end (norm, baseline, decimate) | 0 | 0.0% |
| spatial + `initial_linear` | 158,732 | 2.0% |
| `scale_projection` | 18,944 | 0.2% |
| dilated conv trunk | 828,064 | 10.3% |
| **flat transformer** | **6,318,592** | **78.7%** |
| pool + three heads | 706,510 | 8.8% |
| **total** | **8,030,842** | |

## Checked, and how

| check | result |
|---|---|
| distinct primes `B,C,T,K,E,S = 3,5,53,7,11,13` | all four outputs correct |
| backward, all three losses | 159/159 receive gradient, none exactly zero |
| batch size 1 | correct — no `squeeze()` bug |
| real shape `[4, 73, 250]` | logits and semantic logits finite |
| full `LitTrialFive._step` on real cached batches | train ~4.22, val 1.70 |
| augmentation does not perturb the instance norm | train sigma == eval sigma |
| augmentation live in train, inert in eval | confirmed both ways |
| `dilation_period: null` | raises `ReceptiveFieldError` as intended |

⚠️ Not yet checked: any run longer than five steps. **Nothing here has been
trained.**

## ⚠️ Where the augmentation lives, and why it moved

`ComponentDropout` and `TimeMask` are **inside the model**, after
`RobustInstanceNorm`, not in the LightningModule. Applied to the raw window
first — where they started — the zeroed samples enter the median and MAD the
instance norm computes:

| samples zeroed of 250 | sigma | normalised amplitude |
|---|---|---|
| 0 | 0.993 | 1.00x |
| 20 | 0.897 | 1.11x |
| 40 | 0.785 | **1.26x** |
| 80 | 0.499 | **1.99x** |

The instance norm exists to make a training window and a holdout window
comparable. An augmentation upstream of it destroys exactly that property, on
training windows only, silently. Inside the model both modules are gated on
`self.training`, so validation and `predict.py` get the clean path without
either having to know they exist.
