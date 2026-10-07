# Variant K — G2, scaling-proof, with a second target head

Trial 13's **G2** (isolated-word training) is the best recipe so far: 63.41 as
a 3-seed ensemble, 62.82 for its best seed. Our models score 69–72 on the
local test split, close to the organisers' 73.23 reference, then lose 8–22
points on Kaggle. So K changes the two things that separate local data from
the holdout and that ensembling cannot fix, and keeps the network as it is.
The full reasoning, with figures, is the "Trial 16 Variant K" page Kamil
holds; this file is the record.

## The changes

| # | change | config | why |
|---|---|---|---|
| 1 | per-window normalisation | `model_kwargs.input_norm: instance` | the holdout must now use a fixed training scaler; each window rescaled by itself makes recording gain and offset irrelevant |
| 2 | no baseline | `baseline_seconds: 0.0` | the "baseline" is the first 0.5 s after onset: the word's own response, and half of a 1 s isolated window |
| 3 | second head | `second_embeddings: <early-layer table>` | every model shares t5's middle-layer geometry and so the same hub words; a second target space has different hubs |
| 4 | onset jitter ±40 ms | `jitter_seconds: 0.04` | holdout onsets sit on the 20 ms grid, and prediction averages ±1-sample shifts |
| 5 | weight EMA | `ema_decay: 0.999` | narrows the seed-to-seed spread |

**K0** has changes 1, 2, 4 and 5. Loaded and diffed: K0 differs from trial
15's I2 by `baseline_seconds`, `jitter_seconds` and `ema_decay` only, and K
from K0 by `second_embeddings` only.

**The second head.** `ReferenceNet(second_head=True)` adds `project_second`,
a 256 → 1024 linear map from the same word vector. It is trained with its own
deduplicated SigLIP, temperature and bias against the second table, and the
loss is `loss_first + second_loss_weight * loss_second`. Everything that
ranks (validation metrics, `cli.evaluate`, `cli.predict`) reads
`candidate_scores`: with one head exactly the old scaled cosines; with two,
each head's 50 scores z-scored per row and averaged, the ensemble script's
rule. `val/bacc10_first` and `bacc10_first_head` in `evaluation.json` give
head A alone. 6,114,070 parameters against G2's 5,850,902.

**The layer.** t5-large has 25 hidden states; the existing table is state 12.
`pbs/trial16_embed.pbs` builds states 3, 6 and 18. `cli.hub_check` ranks the
50 words by closeness to each table's centroid (where uncertain predictions
land) and applies a rule fixed in advance: the candidate with the lowest
Spearman against state 12 that shares at most 5 of the 10 most central words.
`k.yaml` names state 6 until the check has run.

## Running it

From the repo root on CX3.

```bash
git pull
qsub pbs/trial16_embed.pbs                          # CPU, the three layer tables
PYTHONPATH=src .venv/bin/python -m pnpl2026.trial16.cli.hub_check \
    --candidates data/word_embeddings_t5_layer3.npz \
                 data/word_embeddings_t5_layer6.npz \
                 data/word_embeddings_t5_layer18.npz   # then fix k.yaml if needed
uv run python -m pnpl2026.trial16.cli.training_scaler  # -> data/trial16_training_scaler.npz
qsub -J 1-6 pbs/trial16.pbs                         # K x3, K0 x3
uv run python -m pnpl2026.trial16.cli.collate       # -> runs/trial16_results.md

# compliant predictions with onset shifts, then each 3-seed ensemble
for run in trial16_kn trial16_k0n; do for seed in 0 1 2; do
    uv run --frozen python -m pnpl2026.trial16.cli.predict \
        --ckpt runs/${run}_s${seed}/best.ckpt --holdout-dir data/holdout --onset-shifts -1 0 1
done; done
```

K0 needs no second table, so `qsub -J 4-6` can start before the embed job ends.

## Decision rule (fixed 2026-10-07, before any score)

Every comparison is between 3-seed ensembles, all scored compliantly.

1. K beats I2 by more than 2 points: K replaces G2 in the final ensemble.
2. K within 2 points of I2: K joins the final ensemble as one more recipe
   group, the others' vote shares unchanged.
3. K more than 2 below I2: K is dropped; K0 against I2 says which part failed.

If I2 itself loses to compliant G2 by more than 2, per-channel normalisation
costs more spatial information than it saves: switch K and K0 to a
topography-keeping version (trial 8's log-spread fed back, or one scale per
sensor type), or to `input_norm: none` with the training scaler.

## What was checked (2026-10-07, laptop, synthetic cache)

- K and K0 configs load; the diffs against I2 and K0 are as stated above
- smoke runs, 2 epochs each on CPU: both trained, early-stopped, saved
  checkpoints and scored themselves under every protocol. K logs `loss_first`,
  `loss_second` and `bacc10_first`; EMA weights apply at load
- `cli.predict` on the real holdout files (960 rows) with the K checkpoint:
  compliant scaling, onset shifts −1 0 +1, in-context and `--window-only`;
  rows finite, each summing to 1. The combined scores differ from head A's;
  combining a head with itself keeps its ranking
- regression: the K0 checkpoint predicted through trial 15's `cli.predict`
  and trial 16's gives identical CSVs (largest difference 0.0)
- `cli.hub_check` and `cli.ensemble` run on the synthetic tables and CSVs
- not run: the shape-check skill. The only new tensor is head B's
  `[n, 1024]`, exercised end to end by the smoke runs and the holdout predict

## Results

### 2026-10-07 — the rule's last clause fired; relaunched without per-window normalisation

Trial 15's I2 (G2 + per-window normalisation) scored **56.17** compliant,
against compliant G2's **63.41**: 7.2 below, past this file's 2-point line
(I1 lost 6.0 to G1 the same way). So K and K0 were stopped and relaunched with
`input_norm: none`; no-baseline, jitter, EMA and the second head unchanged.
The run names gained an "n" (`trial16_k0n`, `trial16_kn`) so the job's
`--resume` cannot continue the earlier normalised checkpoints, which have the
same parameters and would load without complaint.

What the comparisons now read: **K0n against compliant G2** (63.41) -- the
three remaining changes together -- and **Kn against K0n**, the second head.
The thresholds above stand (2 points).

**Hub check (2026-10-07): layer 3.** Against the reference (state 12):
layer 3 Spearman +0.27 with 3 of the top 10 central words shared; layer 6
+0.53 with 5; layer 18 +0.57 with 7. By the rule fixed in advance, `k.yaml`
now names `data/word_embeddings_t5_layer3.npz`. Every table's simulated
top-10 share runs from 0% to 92-100% for its most central word (a fair share
is 20%), so hubs are strong in all of them.

Pending: the K0n/Kn wave,
`runs/trial16_results.md`, and the compliant 3-seed ensembles on the
leaderboard.
