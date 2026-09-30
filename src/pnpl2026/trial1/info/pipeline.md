# The trial-1 pipeline, stage by stage

Every shape, and what changes it. Written 2026-09-24.

## Stages

| stage | in | out |
|---|---|---|
| `load_recording` | h5 | `[306, n_times]` float32 |
| `ChannelStandardiser.apply` | `[306, n_times]` | same, per-channel unit scale |
| `Whitener.apply` | `[306, n_times]` | **`[73, n_times]`** decorrelated |
| `cut_windows` | `[73, n_times]` | `[n_windows, 73, 400]` |
| `clip_and_cast` | float32 | float16, clipped at 20 sigma |
| `CachedWindows.__getitem__` | `[73, 400]` | **`[73, 250]`** float32 |
| `TrialOneNet.stem` | `[B, 73, 250]` | `[B, 192, 250]` |
| `EncoderBlock` x4 | `[B, 192, 250]` | `[B, 192, 250]` — preserved |
| `pool` | `[B, 192, 250]` | `[B, 192]` |
| `classifier` | `[B, 192]` | `[B, 50]` logits |

## ⚠️ The 400 → 250 crop

The cache holds `(-0.32, 1.28)` s = 400 samples. The competition scores
`(0.0, 1.0)` s = 250 samples, starting at **sample 80**.

That 0.32 s margin exists for one reason: **time-jitter augmentation**. A
cache cut to exactly 250 samples has nothing for the crop to slide over.

Training jitters the offset by ±25 samples (0.1 s); **validation and holdout
take offset 80 exactly**, or the score is not comparable to the leaderboard's.

⚠️ Every offset is derived from the manifest in `crop_geometry`, never
hard-coded. A literal `80` would not follow a change to the builder's bounds,
and the failure is silent.

## Where the label space is decided

`build_cache` calls `vocab.to_id` when it writes `labels_*.npy`, so the cache
is already in competition column order. Nothing downstream re-maps, and
nothing should.

## The quadrature branch

`EncoderBlock` carries four branches, not three: three linear depthwise
convolutions plus `QuadratureBranch`, which returns `sqrt(a^2 + b^2)` over a
learned filter pair — an amplitude envelope, invariant to phase.

It is the block's only nonlinear branch. Single-trial MEG phase is not
reproducible across repetitions of a word; induced power is. Ablate with
`model_kwargs.with_quadrature: false`.
