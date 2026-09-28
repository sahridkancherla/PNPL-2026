# The trial-6 pipeline — stage order and every shape between stages

Measured from the built models, not read off the design. `T` is the cropped
window in samples, `F` the frame count the trunk sees.

## Forward, C0 (trial 5's network)

| stage | in | out | learned |
|---|---|---|---|
| cache read + crop | `[B, 73, 400]` | `[B, 73, 250]` | no |
| `RobustInstanceNorm` | `[B, 73, 250]` | `[B, 73, 250]` + `[B, 73]` | no |
| `BaselineCorrect` 0–0.1 s | `[B, 73, 250]` | `[B, 73, 250]` | no |
| `FixedLowpassDecimate` 20 Hz, /5 | `[B, 73, 250]` | `[B, 73, 50]` | no |
| `GaussianNoise`, `ComponentDropout`, `TimeMask` | `[B, 73, 50]` | `[B, 73, 50]` | no — **training only** |
| `spatial` | `[B, 73, 50]` | `[B, 270, 50]` | yes |
| `initial_linear` | `[B, 270, 50]` | `[B, 512, 50]` | yes |
| `DilatedConvTrunk` | `[B, 512, 50]` | `[B, 256, 50]` | yes |
| `.transpose(1, 2)` — the only one | `[B, 256, 50]` | `[B, 50, 256]` | — |
| `FlatTransformer`, fused attention | `[B, 50, 256]` | `[B, 50, 256]` | yes |
| `AttentionPool` | `[B, 50, 256]` | `[B, 256]` | yes |
| heads | `[B, 256]` | `[B, 50]`, `[B, 1024]`, `[B, n_sessions]` | yes |

## Stream B, one band (C4, C5, ab_combined)

Reads the **raw** `[B, 73, T]` window, not stream A's normalised one.

| stage | in | out | learned |
|---|---|---|---|
| `BandEnvelope` — reflect-pad 62, FFT band mask x analytic, abs | `[B, 73, 250]` | `[B, 73, 250]` | no |
| `FixedLowpassDecimate` 20 Hz, /5 | `[B, 73, 250]` | `[B, 73, 50]` | no |
| log, floored at 1e-4 | `[B, 73, 50]` | `[B, 73, 50]` | no |
| `FittedProjection` — PCA-whitened, clipped ±5 | `[B, 73, 50]` | `[B, 32, 50]` | **fitted** on training windows |

Streams are concatenated on axis 1 before the augmentation: `73 + 32k` wide,
or `32k` with stream A off. Both use the same taps and factor, so their frame
counts agree; the constructor raises if they ever do not.

## Per run

| run | T | F | trunk dilations, reach | spatial in | params |
|---|---|---|---|---|---|
| C0, C3, C7, R1, shuffled | 250 | 50 | 1,2,4,8,1 · 33 | 73 | 8,030,713 |
| C1 | 250 | 50 | 1,2,4,8,1 · 33 | 73 | 3,292,153 |
| C2 | 250 | 50 | 1,2,4,8,1 · 33 | 73 | 1,712,633 |
| C4, C5 | 250 | 50 | 1,2,4,8,1 · 33 | 32 | 8,000,699 |
| W05 | 125 | 25 | 1,2,4,1,2 · 21 | 73 | 8,030,713 |
| C6 | 250 | 250 | 1,2,4,8,1 · 33 | 73 | 8,030,713 |
| ab_combined | 250 | 50 | 1,2,4,8,1 · 33 | 137 | 8,047,993 |

Counts use 13 training sessions for the adversary head, the number in the
real cache's manifest.

## Checked, and how

| check | result |
|---|---|
| C0 loads trial 5's state dict strictly and matches its outputs | max diff 7e-7 |
| fused attention vs trial 5's explicit version | the same comparison, eval mode |
| distinct primes `B,C,T,K,E,S = 3,37,251,7,17,13`, 7 layouts, batch 3 and 1 | 145 checks, 0 failures |
| backward, all three losses, every layout | every parameter gets a gradient |
| envelope of a 30 Hz carrier with 3 Hz modulation | modulation recovered to 0.04 |
| out-of-band leakage, 1 s window | ~2% amplitude (−34 dB) at 8–10 Hz outside |
| fitted PCA | identity covariance on its fit data; raises if unfitted |
| D-SigLIP default vs trial 5's | identical; the C7 reduction is ~500x larger at batch 512 |
| every config through `cli.train` on a synthetic cache | 12 of 12, 2 epochs each |
| resume after 2 of 3 epochs | continues from epoch 2; PCA restored, not refit |
| `cli.evaluate`, `cli.session_probe`, `cli.predict` on those checkpoints | all run; predict crops W05 to 0.5 s |
| `pbs/trial6.pbs`, locally with uv swapped for the venv | preflight, staging and training all run |

⚠️ Not yet checked: anything on real data. Every run above is synthetic, and
the envelope PCA's variance figures (~49% kept) are for white noise, not MEG.
