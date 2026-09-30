# The trial-4 pipeline, stage by stage

Every shape, and what changes it. Written 2026-09-26.

## Stages

| stage | in | out |
|---|---|---|
| `CachedWindows.__getitem__` | `[73, 400]` fp16 on disk | `[73, 250]` float32, jittered |
| collate | | `[B, 73, 250]`, `[B]` labels, `[B]` session ids |
| `_augment` (train only) | `[B, 73, 250]` | same, spans and components zeroed |
| `RobustInstanceNorm` | `[B, 73, 250]` | same + `[B, 73]` log-scale |
| `FixedLowpassDecimate` | `[B, 73, 250]` | **`[B, 73, 125]`** |
| transpose | `[B, 73, 125]` | `[B, 125, 73]` |
| `stem` + static bias | `[B, 125, 73]` | `[B, 125, 144]` |
| stack A (2 blocks) | `[B, 125, 144]` | `[B, 125, 144]` |
| down 2, widen | | `[B, 63, 192]` |
| stack B (3 blocks) | `[B, 63, 192]` | `[B, 63, 192]` |
| down 2, widen | | `[B, 32, 256]` |
| stack C (3 blocks) | `[B, 32, 256]` | `[B, 32, 256]` |
| up 2, narrow, **+ skip from B** | | `[B, 63, 192]` |
| stack D (2 blocks) | `[B, 63, 192]` | `[B, 63, 192]` |
| `AttentionPool` | `[B, 63, 192]` | `[B, 192]` |
| `closed_set` | `[B, 192]` | `[B, 50]` logits |
| `semantic` | `[B, 192]` | `[B, 1024]` embedding |
| `adversary` | `[B, 192]` | `[B, 13]` session logits |

## ⚠️ The 400 → 250 crop

The cache holds `(-0.32, 1.28)` s = 400 samples. The competition scores
`(0.0, 1.0)` s = 250 samples, starting at **sample 80**. That 0.32 s margin
exists for one reason: **time-jitter augmentation**. Training jitters the offset
by ±25 samples; **validation and holdout take offset 80 exactly**, or the score
is not comparable to the leaderboard's.

Every offset is derived from the manifest in `crop_geometry`, never hard-coded.

## The three losses

```
L = 1.0 * D-SigLIP(embedding, t5[label])    the bet -- what 73.23% trains on
  + 0.3 * CE(logits, label, w = 1/sqrt(n)) the scored metric, directly
  +       CE(session_logits, session)       through a gradient reversal
```

⚠️ The scored logits are a **blend** of the two classification heads, both
log-softmaxed first because a scaled cosine and a raw logit are not on the same
scale. `val/bacc10_closed` and `val/bacc10_semantic` are logged separately so
`inference_semantic_weight` can be set from evidence.

## ⚠️ Validation is scored on the EMA weights

`on_validation_start` swaps the averaged weights in and `on_validation_end`
swaps them back. So `best.ckpt` is selected by what the **average** scored, while
the checkpoint's live weights are whatever the last step produced. `predict.py`
and `cli.evaluate` therefore call `apply_ema_weights()` by default. Predicting
with the live weights submits a different model from the one that was chosen.

## Where the label space is decided

`build_cache` (trial 1's) called `vocab.to_id` when it wrote `labels_*.npy`, so
the cache is already in competition column order. `embeddings.py` re-checks that
target row `i` is `vocab.PRIMARY[i]` on every load, because that is a second
place the same silent failure could enter.
