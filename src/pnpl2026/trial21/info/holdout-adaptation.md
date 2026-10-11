# Trial 21 — holdout adaptation: distillation on the holdout epochs, and the hybrid

## Why

The review of 2026-10-10 (`handoff-docs/architecture-review-2026-10-10.md`)
read every trial's record against the board. Every whole-point gain closed a
gap between training conditions and what the holdout serves: all-word t5
supervision on 306 sensors (+20), holdout-shaped items (+5.8), ensembling
different recipes (+3 to +6), the holdout-fitted hub correction (+0.9).
Every loss added a prior learned from the training text (L −23.6, the
word-pair fusions, the closed-set head). The single-recipe ceiling has been
63.4 since 3 Oct; the mix is 69.76 and saturating.

Two gaps are still open, and both are about the holdout itself:

- **the session**: a recording no weight or BatchNorm statistic has seen.
  Measured today, its spatial covariance is not an outlier (nearer the
  training recordings than they are to each other), but it is still unseen
- **the token format**: list words only, with invisible neighbours and
  unknown gaps. G1 and G2 matched truncation and loneness, not sparsity; L
  matched sparsity with list-word-only sequences and learned their order

The 156 sentence epochs and 92 lone words are the only data that have both,
and no run has trained on them. The organisers call unsupervised use of the
holdout MEG not prohibited (CLAUDE.md, 2026-10-08); the hub correction
already uses it.

## Part A — transductive self-distillation (`cli/distil.py`)

A trained member is fine-tuned briefly. Each step has two forward passes:

| pass | items | loss |
|---|---|---|
| training | the recipe's own batch, shaped as it trained (G2's lone words, dedup, its heads) | its own D-SigLIP |
| holdout | 16 holdout items as `predict` serves them (a sentence as one item, a lone word as a 1 s item), network in TRAINING mode | KL(teacher ‖ student) over the 50 scored words, weight 0.5 |

Both distributions are softmaxes of per-row z-scored scores at one shared
temperature (the ensemble script's space), so the student's SigLIP scale is
untouched. The temperature makes the teacher's mean top-1 probability 0.3,
about its likely top-1 accuracy. The teacher is the best submitted file,
hub-corrected and lone-filled: a function of MEG, not of text, so the student
cannot learn list-word order from it. 600 steps, lr 3e-5, AdamW, no
selection (no local score can judge it). BatchNorm runs in training mode
throughout and is re-estimated on all holdout items at the end (`--final-bn
holdout`), which is what the holdout passes saw. The checkpoint is written
in the student's family's own format, so `cli.predict` reads it unchanged.

What it buys: the encoder, merger and BatchNorm adapt to the holdout session;
the sentence stage sees the holdout's sparse list-word tokens with
consistent targets; each student inherits the 23-model mix's ranking. What
bounds the risk: the retained training loss, the KL weight, the step count.

**A0, the gate (no training, one submission).** The best mix re-predicted
with `--bn-adapt` for every per-word member (`cli.predict` lifts the old
refusal under the training scaler). Above +0.5: the session shift is real and
A2 goes ahead. Below −1: the GPU goes to part B.

## Part B — the hybrid (`stem: dascoli`)

The continuous family reads each epoch once, its context is time-resolved
and a lone word is the same path, but its stem is weak: 60 to 65 on full 3 s
sentences against G2's 69 to 73. `stem: dascoli` replaces the four MEGState
branches and the two LayerNorm blocks with the per-word family's conv stack
(10 dilated layers of 160, masked BatchNorm, GLUs, the final 1x1 pair) --
trial 18's modules under trial 18's names, so `init_encoder_from:
runs/trial13_g2_s{seed}/best.ckpt` loads a trained G2's merger, 1x1, subject
layer, conv stack and t5 head key for key (`lit.warm_start_encoder`; 94
tensors; the per-word attention pool is left out). The time transformer,
readout and tower train from scratch on SPW's recipe. Marker dropout is
random (`marker_drop_mode: random`, p 0.3) instead of "keep the list words'
markers", trial 19's identity-based rule. B-cold is the same network from
scratch, a second wave.

## Decision rules (fixed 2026-10-11, before any score)

All files hub-corrected on the holdout, lone rows I3's, against 69.76.

1. **A0**: the best mix with `--bn-adapt` members. Above +0.5 → run A2;
   below −1 → skip A2
2. **A2**: the 12 students (G2, K2, K18, SPW × 3 seeds) replace their
   parents in the mix. Keep if above +1. Above +2 → **A3**, one more round
   with the new mix as teacher. Never a third round
3. **B** alone (3 seeds, `_ts_lone`) against G2's 63.41, recorded; in the
   mix, keep if above +0.5; replace SPW's three if it beats them alone by
   more than 2
4. A gap under about 1 point is a tie (one row of a single-row class moves
   the board by 2.000), and the public number overstates the final one for
   a file whose members were chosen on the public rows

## Running it

From the repo root on CX3, after `git pull`.

```bash
uv run python -m src.pnpl2026.trial21.cli.training_scaler     # seconds -> data/trial21_training_scaler.npz

# A0 -- the gate: every per-word member of the best mix with AdaBN (GPU, minutes)
for run in trial13_g1_s0 trial13_g1_s1 trial13_g1_s2 trial13_g2_s0 trial13_g2_s1 trial13_g2_s2 \
           trial16_kn_s0 trial16_kn_s1 trial16_kn_s2 trial18_k2_s0 trial18_k2_s1 trial18_k2_s2 \
           trial18_n_s0 trial18_n_s1 trial18_n_s2 trial18_k18_s0 trial18_k18_s1 trial18_k18_s2 \
           trial10_f1_s0 trial12_w1024_s0; do
    uv run python -m src.pnpl2026.trial21.cli.predict --ckpt runs/$run/best.ckpt --holdout-dir data/holdout --bn-adapt
done
# then the same ensemble / hubnorm_holdout / lone_fill commands as for the 69.76 file,
# with the *_in_context_bnadapt_ts.csv files in place of the *_ts.csv ones (SPW's three unchanged)

# A2 -- the students, then their CSVs join the mix in their parents' place
qsub -J 1-12 pbs/trial21_distil.pbs
qsub -v RUN=3,TEACHER=submissions/<new best>.csv pbs/trial21_distil.pbs   # A3, per line

# B -- the hybrid
qsub -J 1-3 pbs/trial21.pbs                                   # ~3 h each
uv run python -m src.pnpl2026.trial21.cli.collate             # -> runs/trial21_results.md
```

Predict every trial 21 checkpoint with trial 21's `cli.predict`; it reads
both families. `cli.ensemble`, `cli.hubnorm_holdout` and `cli.lone_fill` are
trial 20's, unchanged.

## What was checked (2026-10-11, laptop, synthetic cache, CPU)

- **both stems, distinct primes**: contracts hold; padding past each item's
  length changes nothing (difference exactly 0); an item alone equals itself
  in a batch (1e-6 megstate, 6e-4 dascoli, the latter the conv kernels'
  float accumulation over a different padded length); every parameter gets
  a gradient with all heads on; the dascoli stack's reach is 62 samples
  (1.24 s) each side
- **warm start**: from a random-weight trial 13 checkpoint, 94 tensors
  loaded, the per-word pool skipped, nothing missing; the hybrid's merger,
  1x1, subject layer, conv stack and final 1x1s give the source encoder's
  output exactly (difference 0.0), and its SigLIP scale and t5 head are the
  source's
- **hybrid training end to end** on the synthetic cache: one epoch with the
  tower and phoneme head on, selection on the fragment protocol, best.ckpt
  scored under every protocol with dumps
- **distillation end to end** for a per-word student (G2's config) and a
  continuous one (SPW's): training items shaped as trained, 248 holdout
  items (156 sentences, 92 lone words, 960 rows), the teacher's temperature
  fitted, three steps, BatchNorm re-estimated (10 layers per-word, 0
  megstate), checkpoints written. **Round trip**: with 0 steps and blended
  BatchNorm, the saved student predicts byte-for-byte what its parent
  predicts through the same `cli.predict` (largest difference 0.0, every
  top-10 set equal), for both families
- **predict, both families**, with and without `--bn-adapt`, on the real
  holdout files: 960 rows, every row once, distributions valid
- `bash -n` on both PBS scripts

⚠️ Not run: anything on real MEG. The HPC cache and checkpoints are on RDS.

## Results

Pending: A0, A2, B.
