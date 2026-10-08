# Variant M — HDND's refinement on a frozen checkpoint

## Why

Everything that taught the model more about word order lost on the board:
word-pair fusion (-0.8 to -3.6), trial 17's L (-23.6). Both levers that won
on runs cut its reliance on context: G1's fragments (+5.3), G2's lone words
(+5.8, though the public board has no lone rows). So the next change should
add evidence from each word's own MEG.

HDND (Li et al., arXiv 2609.24095, Sept 2026) does exactly that on top of
d'Ascoli's encoder and sentence transformer -- ours. Balanced top-10 over
250 words against that baseline: Armeni (MEG, ~10 h per subject, the deep
case) 38.1 -> 47.2; Schoffelen listening +6.4; Gwilliams (2 h) +1.2; every
one of nine conditions up, mean +4.1. Never tested on LibriBrain.

It is trained as a SECOND STAGE on a finished model, which stays frozen. So
it costs no retraining: any checkpoint we have -- G1, G2, F1, w1024 -- can be
refined in well under an hour of GPU.

## What it does (`models/refine.py`)

1. the gated outputs of conv stages 2 and 4 (ConvSequence layers 3 and 7,
   160 channels each), averaged in 1, 2 and 4 time bins: 1,120 numbers per
   depth
2. standardised by training statistics, one bias-free map per depth to 128,
   tanh; the normalised prediction z0 the same way, as the query
3. attention over the two depths, the query's dot product with each
4. a zero-initialised map to 1024, bounded: B(r) = c r / sqrt(c^2 + |r|^2),
   c = 0.25; centred by its mean over training batches
5. z = z0 + strength * (B(r) - centre), scored by retrieval as before

Loss (`lit.py`, only the refiner learns): the base's own D-SigLIP over every
word on the refined z, + 0.25 x KL(base || refined) over the 50 words, the
anchor, + 0.3 x HDND's pairing loss: the residual from the word's own
features must point at its target more than the residual from another
word's. HDND's 128-wide family: lr 2e-4, KL 0.25, pairing 0.3.

After training, validation is scored at strengths 0, 0.125, 0.25, 0.5 and 1.
The best goes into best.ckpt; ties go to the smaller, and 0 is the base
itself, so a refiner that does not help on validation is switched off.

## Departures from the paper, and why

| | HDND | here | why |
|---|---|---|---|
| pooled span | the whole 3 s window | the first 1 s | every item has its first second whole (lone words are 1 s; holdout runs end 1.13 s+ after their last onset); later samples are cut short on the holdout and whole in training |
| query skip (eta) | 0 or 1 | 0 | the correction then comes from the word's own MEG; one learned from z0 alone resembles trial 14's closed-set head, which lost on the board |
| retrieval loss | cross-entropy over 250 candidates, frequency-weighted | the base's D-SigLIP over every word | a 50-word cross-entropy learns the training prior over the 50 words; every board loss so far came from learning training-side statistics |
| pairing scale | "the fixed logit scale" | the base's SigLIP scale | the same idea |
| centre | training mean, fixed | running mean over training batches, fixed at evaluation | the same, in one pass |

## Decision rule (fixed 2026-10-08, before any score)

The 3-seed M ensemble against G2 (63.41), then in the best mix (66.12):

1. M more than 2 above G2: refine the mix's other members (G1, F1, w1024)
   and build the refined mix
2. M within 2 of G2: submit the best mix with M in place of G2 once; keep
   it only if it beats 66.12 by more than 1
3. M more than 2 below G2: dropped

## Running it

From the repo root on CX3, after `git pull`. The base checkpoints must be
on RDS: `runs/trial13_g2_s{0,1,2}/best.ckpt`.

```bash
uv run python -m src.pnpl2026.trial18.cli.training_scaler   # -> data/trial18_training_scaler.npz
qsub -J 1-3 pbs/trial18.pbs                                 # M on G2, x3
uv run python -m src.pnpl2026.trial18.cli.collate           # -> runs/trial18_results.md
cat runs/trial18_mg2_s*/refine_strength.json                # the strength curves
```

then predict each seed, plain and with onset shifts, as trial 17 does
(`--ckpt runs/trial18_mg2_s${seed}/best.ckpt`).

## What was checked (2026-10-08, laptop)

- shape check, distinct primes on every axis: the contracts hold; at the
  start the refined prediction IS the normalised base; pooling matches a
  hand loop; batch size 1 and a 0.6 s window stay finite; with the base
  frozen only the refiner's four maps train and get gradients; |B(r)| < c
- smoke on the synthetic cache: a 1-epoch base, then the refiner on it for 2
  epochs (27 s): base loaded and frozen, statistics from 2,363 training
  words, 548,864 trainable parameters, the strength curve written into
  best.ckpt, four protocols scored with dumps
- leak check: the refined smoke model scored 80.9 on validation against the
  base's 21.6 (the synthetic data plants a word-specific response); with
  each word's features swapped for another's it fell to 19.6, chance. The
  gain comes from the word's own features only
- `cli.predict` on the real holdout files (960 rows), at the stored strength
  and with `--refine-strength 0 --onset-shifts -1 0 1`

## Results

### M on G2, local (2026-10-08): the refiner did not help -- a negative result

Validation BAcc@10 (fragment protocol) at each strength; 0 is G2 itself:

| seed | 0 | 0.125 | 0.25 | 0.5 | 1 | chosen |
|---|---|---|---|---|---|---|
| s0 | **39.66** | 38.60 | 38.13 | 38.35 | 38.11 | 0 |
| s1 | 35.43 | **35.61** | 34.96 | 33.80 | 34.44 | 0.125 |
| s2 | **40.22** | 37.43 | 37.43 | 38.01 | 37.71 | 0 |

Two seeds of three switched the refiner off; the third kept a +0.18 that is
noise. Every non-zero strength on s0 and s2 cost 1.3-2.8. The test columns
therefore equal G2's (FRAG test 31.80, ref test 69.31 against 69.32). No
submission: the 3-seed M ensemble is G2 with one seed nudged.

Pending: whether training BAcc rose while validation fell (overfitting the
training sessions) or neither moved (the refiner never learned), from each
run's metrics.csv.
