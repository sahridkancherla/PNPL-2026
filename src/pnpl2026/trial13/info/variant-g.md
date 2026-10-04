# Variant G — F1, trained and judged under holdout conditions

Trial 10's F1 scores **71.07** BAcc@10 on pnpl's test partition but **58.02**
on the leaderboard; trial 12's w512 scores 71.44 locally and **49.13** on the
leaderboard. Trial 13 asks why local and leaderboard disagree, fixes what can
be fixed in training, and builds a local score meant to track the leaderboard.
The target is ~0.70 on the leaderboard; the top teams sit at 72.7–76.5.

## Why local and leaderboard disagree — measured 2026-10-02

From the real holdout file, the 12 submission CSVs and the run logs, nothing
retrained.

**1. Local evaluation is a different task.** The reference protocol gives the
model every word of a whole sentence (up to 64 tokens) and a full 3 s window
per word. The holdout gives this:

| | real holdout, subject 0 |
|---|---|
| sentences | 156 runs of **consecutive** vocabulary words — lengths 1–9: 2, 0, 2, 24, 55, 37, 23, 11, 2 |
| gap between onsets | median **0.21 s** (5–95%: 0.07–0.57); 99.4% under 0.8 s |
| span, first to last onset | median 1.17 s |
| epoch | starts **exactly at the first word's onset**; ends 1.13–1.68 s after the last (median 1.36) |
| window per scored word | median **1.94 s**; only 4.4% reach 3 s |
| isolated words | 92 of 960 rows, exactly 1 s each |

A local score of 80 on the reference protocol would not mean 70 on the
leaderboard: F1 and w512 tie locally and differ by 9 points there.

**2. Trial 10's local "holdout protocol" was mis-specified.** It took every
vocabulary word of long narrative sentences as tokens, skipping the unscored
words between them — sparse tokens the holdout never serves — and read ~27
for every run. The real holdout is dense runs of consecutive words, closer to
training, which is why the leaderboard reads 41–58 rather than 27. It is kept
as the `sparse` protocol, for comparison only.

**3. The isolated words collapse.** On the 92 isolated rows, every trial 10/12
model predicts "really" or "so" for 75–93% of them, with only 2–10 distinct
top-1 words, and 24–35 words almost never in the top 10. None ever trained on
a lone 1 s window. That is 9.6% of rows at or below chance: roughly 3–4 BAcc
points.

**4. One leaderboard score carries a large training-seed component.** On the
holdout, models with equal local scores agree on only 30–48% of their top-1
picks, and F0's two seeds agree 41% — no more than different architectures
do. So "F1 beats w512 because it is smaller" is not established by one seed
each. The same disagreement is why an **ensemble** should help.

Also measured — hubness: every pre-trial-10 model leaves 3–6 words almost
never in the top 10 (Spearman −0.92 with the leaderboard over 11 submissions);
trial 10/12 models have none, but top-10 occupancy still runs 4.5–48% of rows
against an even 20%.

## The Zipformer, on F1

Not as a drop-in: F1's sentence transformer sees 4–8 word tokens on the
holdout, and multi-rate downsampling over a handful of tokens has nothing to
do. The coherent version is trial 11's stream — one epoch, one timeline, words
read off at their onsets — ported to trial 10's front end with every-word
targets; the holdout's geometry (one short epoch per fragment, starting at the
first onset) suits it. Deferred: in the old pipeline Z1 (37.2) did not beat E0
(38.0), that gap is inside single-seed noise, and the port is large. Revisit
as one seed-averaged comparison against G's best.

## What was built

A copy of trial 10's code with these changes; trial 10's cache and t5 table
are read, not rebuilt. Trial 10/12 checkpoints load here unchanged (config
superset, identical state-dict keys, `trial10_reference` registry alias).

| piece | change |
|---|---|
| `data/sequences.py` | **fragment** protocol: maximal runs of consecutive vocabulary words (adjacent table rows, one sentence, gaps ≤ 0.8 s), 4–9 words, MEG cut 1.13–1.68 s after the last onset; other vocabulary words served alone with 1 s, at the holdout's 9.6% |
| `data/dataset.py` | **shaping**, training only: per sentence, `fragment_p` becomes a random run of consecutive words (lengths weighted as the holdout's) with the MEG cut after it; `isolated_p` becomes one word with 1 s. Every word keeps its t5 target |
| `models/encoder.py` | **masked BatchNorm**: batch statistics over real samples only, so padding never enters them. Identical to BatchNorm when nothing is padded |
| `data/datamodule.py` | validation under `selection_protocol` (fragment): best.ckpt and early stopping follow the holdout-shaped score |
| `cli/evaluate.py` | three protocols; `--dump` keeps every scored word's scores; `--bn-adapt` adds a score after re-estimating BatchNorm on the scored inputs (AdaBN) |
| `cli/calibrate.py` | every local score beside the leaderboard with its Spearman; the local ensemble; column normalisation and AdaBN, measured |
| `cli/ensemble.py` | averages submission CSVs (row-wise z-scored log-probabilities); `--column-normalise` z-scores each word across the rows of each holdout source |
| `cli/predict.py` | `--bn-adapt` |

⚠️ `--column-normalise` and `--bn-adapt` pool all holdout rows' inputs (never
a label). The rules do not say whether that is allowed; both are off by
default and measured locally before any use.

## The queue — `configs/trial13/runs.txt`, 4 configs × 3 seeds

All F1-sized (5.85M); each changes one thing from G0.

| run | `fragment_p` | `isolated_p` | answers |
|---|---|---|---|
| G0 | 0 | 0 | control: F1 with fragment-protocol selection only |
| G1 | 0.5 | 0 | does training on holdout-like fragments help? |
| G2 | 0 | 0.1 | does training on lone 1 s words fix the isolated collapse? |
| G3 | 0.5 | 0.1 | both |

Read configs by their **seed means**, and confirm on the leaderboard with the
3-seed ensemble of the best one.

## Running it

From the repo root on CX3. Record every leaderboard score in
`configs/trial13/leaderboard.csv`.

```bash
# Phase 0 -- no training: one ensemble of the ten trial 10/12 prediction files (login node, light)
S=submissions
uv run python -m pnpl2026.trial13.cli.ensemble --out $S/ensemble_t10t12_in_context.csv --inputs \
    $S/trial10_f0_s0_in_context.csv $S/trial10_f0_s1_in_context.csv $S/trial10_f1_s0_in_context.csv \
    $S/trial10_f3_s0_in_context.csv $S/trial12_w512_s0_in_context.csv $S/trial12_w1024_s0_in_context.csv \
    $S/trial12_d8_s0_in_context.csv $S/trial12_d16_s0_in_context.csv $S/trial12_wd_s0_in_context.csv \
    $S/trial12_e320_s0_in_context.csv
# optional second file: the same inputs plus  --column-normalise --holdout-dir data/holdout

# Phase A -- the calibration job: existing checkpoints under the new protocols
qsub pbs/trial13_calibrate.pbs            # -> runs/trial13_calibration.md

# Phase B -- the queue
qsub -J 1-12 pbs/trial13.pbs
uv run python -m pnpl2026.trial13.cli.collate        # -> runs/trial13_results.md

# Phase C -- the best config's three seeds as one submission (CPU job for predict)
qsub -l select=1:ncpus=8:mem=32gb -l walltime=01:00:00 -N predict-13 -j oe -o logs/ <<'EOF'
cd "$HOME/PNPL-2026"; source "$HOME/.local/bin/env"
export PYTHONPATH="$PWD/src" UV_OFFLINE=1 HF_HUB_OFFLINE=1
for seed in 0 1 2; do
    uv run --frozen python -m pnpl2026.trial13.cli.predict --ckpt runs/trial13_g3_s$seed/best.ckpt --holdout-dir data/holdout
done
EOF
uv run python -m pnpl2026.trial13.cli.ensemble --out submissions/trial13_g3_ensemble_in_context.csv \
    --inputs submissions/trial13_g3_s0_in_context.csv submissions/trial13_g3_s1_in_context.csv \
             submissions/trial13_g3_s2_in_context.csv
```

Replace `g3` with whichever config wins. Submission is manual, through the
competition web page. ⚠️ Default (in-context) predictions read holdout MEG
outside each word's own 1 s window; the rules do not settle that either.

## What was checked (2026-10-02)

- **ensemble CLI** on the real CSVs: one file in gives every row's ranking
  back unchanged; ten files' top-10s match a hand computation; a reordered or
  duplicated input is refused before anything is written; column
  normalisation brings top-10 occupancy to 0.12–0.27 per source
- **fragment protocol**, synthetic recordings in pnpl's layout: runs are
  consecutive vocabulary words of one sentence, 4–9 long, tails 1.13–1.68 s,
  isolated words exactly 1 s and never fragment words, 9.0% isolated,
  deterministic. Span (median 1.16 s) and window (2.10 s) match the real
  holdout's (1.17, 1.94)
- **shaping**: 3,000 draws, every served item's targets, columns and windows
  are the drawn slice; rates 10.4% isolated, 50.1% fragments; length 5 the
  commonest
- **masked BatchNorm**: equals BatchNorm with nothing padded (outputs and
  running statistics); padded values change neither; trial 10's F1 checkpoint
  gives bit-identical outputs in trial 10 and trial 13
- **shape-check skill**: primes, training and eval mode, padding invariance
  (exactly 0 in both), batch of one, backward, a wrong-shaped mask raises; real
  shaped batches for G0–G3 at real dimensions in fp16, targets aligned
- smoke: G3 and G0 trained 2 epochs on the synthetic cache, selected on the
  fragment protocol, scored themselves (three protocols, ± AdaBN, dumps);
  `calibrate`, `collate`, and `predict --bn-adapt` on the real holdout files

## Results

### Phase 0 — ensembles of existing predictions (leaderboard, 2026-10-02/03)

| submission | members | leaderboard |
|---|---|---|
| `ensemble_t10t12` | F0 s0/s1, F1, F3, w512, w1024, d8, d16, wd, e320 | **61.13** |
| `ensemble_t10t12_v2` | F1, w1024 | **62.42** |

The 10-file ensemble scores 61.1, about 7 points above its members' mean
(53.8 over the 8 with scores) and level with its best member (w1024, 61.3).
The members were not chosen by leaderboard score, so 61.1 is an honest
estimate. ⚠️ v2's two members WERE chosen by their leaderboard scores, on the
same 960 rows that score v2, so its 62.4 is optimistic.

### Phase A — calibration (`runs/trial13_calibration.md`): inconclusive

The fragment protocol scores only 128 test words (38 classes, 13 isolated)
and 239 validation words: runs of 4+ consecutive vocabulary words are rare in
pnpl's narrative recordings, and MOCHATIMIT has none. At ~3 words per class
every fragment score carries ±4-5 points of noise, so the Spearman values
(n = 5) and the ensemble/switch effects there decide nothing. It also reads
far below the leaderboard: F1 scores 22.1 on it against 58.0 there.

Informative: runs agree on top-1 as rarely locally (0.275) as on the holdout
(0.305), so the disagreement is single-trial noise, not a holdout shift.

### Phase B — the queue (`runs/trial13_results.md`, local, 2026-10-03)

Seed means ± half-range, BAcc@10 %; FRAG test is 128 words, frag val 239.

| run | FRAG test | frag val | isolated | ref test |
|---|---|---|---|---|
| G0 control | 32.36 ± 4.16 | 39.05 ± 2.13 | 16.67 ± 5.21 | 70.03 ± 2.89 |
| G1 fragments | **49.22** ± 1.52 | **50.04** ± 1.11 | 21.53 ± 4.17 | **73.27** ± 1.00 |
| G2 isolated | 31.80 ± 1.87 | 38.44 ± 2.40 | **32.64** ± 8.33 | 69.32 ± 2.51 |
| G3 both | 47.98 ± 1.05 | **51.09** ± 1.96 | 29.86 ± 6.25 | 70.70 ± 2.37 |

- **Fragment training is the largest local effect in any trial**: G1 beats G0
  by +17 on the test fragments and +11 on the validation fragments (separate
  recordings), in every seed, while the full-context reference score does not
  drop (73.3 vs 70.0). Fragment noise is ±4-5 per run, far below the gap.
- **Isolated-word training raises isolated recall** (G2 32.6 vs G0 16.7), but
  13 isolated test words make that ±10 noise; it costs nothing elsewhere.
- ⚠️ G1 and G3 were trained on, and selected by, the geometry this protocol
  measures, so part of the gain may be specific to the proxy. Only the
  leaderboard can confirm it: submit each config's 3-seed ensemble.
- Absolute fragment scores are not leaderboard numbers (F1: 22 here, 58 there).
- BatchNorm re-estimation lowers the fragment score in G1 and G3; not used.

### Phase C — leaderboard, 3-seed ensembles (2026-10-03)

| submission | leaderboard | vs G0 |
|---|---|---|
| G0 control | 57.58 | — |
| G1 fragments | 62.84 | +5.3 |
| **G2 isolated** | **63.41** | **+5.8** |
| G3 both | 60.85 | +3.3 |
| G1 + G3 seeds + the 10 trial 10/12 files | 61.41 | +3.8 |

- **Holdout-shaped training helps on the leaderboard**: every shaped config
  beats the control's ensemble by 3-6 points, against ~1.5-2 points of noise
  per submission. Far less than the +17 the local fragment protocol showed,
  which over-predicted.
- **G2, the isolated-word fix, is the best submission not chosen by
  leaderboard score**: consistent with recovering the ~3-4 points the isolated
  collapse cost, plus some. G3 (both) is below either alone; unexplained, and
  within ~1-2 standard errors of G1/G2.
- Seed ensembling of the control is worth nothing here: G0's 3 seeds (57.58)
  sit level with F1's single seed (58.02), the same recipe. F1's 58 was a
  fortunate seed; the ten-file ensemble's +7 came from pooling ten different
  models.
- Adding more models plateaus: 13 models (61.4) score below G2's 3 seeds.

**Deferred, in order:** train the final models on all 162 recordings; a
stream (Zipformer-style) variant on this pipeline; test-time onset jitter.
Multi-subject pretraining is **ruled out**: the deep track does not allow
training on other subjects' data.
