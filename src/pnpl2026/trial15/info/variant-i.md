# Variant I — a word-pair model, and per-window input normalisation

The best submission so far is **66.39**: G1 ×3 + G2 ×3 + F1 + w1024, one
recipe family (trial 10's word decoder) trained three ways. Adding more models
to it stopped paying (every addition lost 2–5 points, 2026-10-06), and
submissions close **15 Oct 2026**. Trial 15 makes two independent changes,
chosen from a literature search on 2026-10-06.

## Part 1 — the word-pair model (no training)

**What the models do not know.** Every holdout "sentence" is a run of 4–9
consecutive words that are *all* in the 50-word list (156 runs, 868 of 960
rows). Every model so far trained on narrative, where most neighbours are
unscored words, so none knows that constraint. If word 3 of a run is "the",
word 4 can only be one of the few list words that directly follow "the".

**The model.** `cli.wordpairs` counts how often each list word directly
follows another within a sentence of subject 0's own **training**
transcripts (the cache's word tables — no MEG, no outside text), smooths each
row toward the overall next-word distribution (pseudo-counts chosen on the
validation transcripts), and saves the transition matrix with its stationary
distribution.

**The fusion.** `cli.fuse` runs forward–backward along each holdout run. Each
word's score becomes its own evidence plus what the other words' evidence says
about it, as a likelihood ratio:

    z / temperature  +  strength · log P(other words' MEG | this word)

The prior is divided out, so a frequent word earns nothing for being frequent
(the metric weights every word equally). Isolated words, and runs of one, are
untouched. `temperature` turns z-scored log-probabilities into calibrated
log-likelihoods; strength 1 is exact Bayes. Both come from `cli.fuse_check`,
which rebuilds the submission's ensemble from per-word dumps, fits the
temperature on validation words, and **picks the setting on validation alone
by a rule fixed in advance**: exact Bayes unless another setting beats it by
more than 0.5 points.

**Evidence.** Language-model rescoring is the step d'Ascoli et al. name next
for the decoder our models copy; "Unlocking Non-Invasive Brain-to-Text"
(arXiv 2505.13446) and a 2026 listened-speech MEG decoder (arXiv 2609.31997,
neural top-10 gating plus GPT-2 next-word probabilities) both gain from it.
Here the model is smaller and the constraint stronger: 50 words, all-list runs.

**Assumption to confirm** before trusting it: the holdout's runs are directly
adjacent words. Inside its runs, onset gaps are 0.07 / 0.14 / 0.21 / 0.34 /
0.57 s (5/25/50/75/95%, 712 gaps), measured from the real holdout files.
`cli.wordpairs --holdout-dir` prints the training transcripts' gaps for
adjacent pairs beside those across one unscored word: the holdout's should
match the first.

⚠️ Like the in-context mode, the fusion reads the other words of a run (their
scores) to score each word; the rules do not settle whether that is allowed.

## Part 2 — per-window input normalisation (I1, I2)

**Why.** Training windows are scaled per recording, from each recording's own
continuous MEG; the holdout comes from recordings the organisers have not
revealed, scaled by what `cli.predict` can fit on its epochs. The 2025
LibriBrain phoneme winner (MEGConformer, arXiv 2512.01443) found
standardising each window by its own statistics the largest single fix for
the shift between their validation data and the holdout (+200% relative over
no normalisation), and a 2026 study (arXiv 2604.04129) found
normalisation-induced shift the main cause of validation-to-test drops. A
local test shares the training scaling, so it cannot see such a shift — one
candidate reason no local score has predicted the leaderboard.

**The change** (`input_norm: instance`, `models/encoder.py`): after the first
±5 clamp, each window's channels are standardised by the window's own mean
and spread over its real samples (never its padding), then clamped again,
then the 0.5 s baseline as before. No parameters, no new state-dict keys;
earlier checkpoints load and run with "none".

**The queue** — `configs/trial15/runs.txt`, 2 configs × 3 seeds, everything
else trial 13's: I1 = G1 + normalisation, I2 = G2 + normalisation. Read I1
against G1 and I2 against G2, seed for seed; only the leaderboard can judge a
normalisation change.

## Running it

From the repo root on CX3.

```bash
git pull

# Part 1 -- login node, CPU, about a minute in all
uv run python -m src.pnpl2026.trial15.cli.wordpairs --holdout-dir data/holdout
C=runs/trial14_calibration
uv run python -m src.pnpl2026.trial15.cli.fuse_check \
    --csv submissions/ens_g1g2_f1w1024_in_context.csv \
    --runs $C/trial13_g1_s0 $C/trial13_g1_s1 $C/trial13_g1_s2 \
           $C/trial13_g2_s0 $C/trial13_g2_s1 $C/trial13_g2_s2 \
           $C/trial10_f1_s0 $C/trial12_w1024_s0
# -> runs/trial15_fuse_check.md; its last line is the cli.fuse command to run
#    (writes submissions/ens_g1g2_f1w1024_in_context_wp.csv)

# Part 2 -- six GPU runs, one array wave, each ~1.5-2.5 h on an RTX6000
qsub -J 1-6 pbs/trial15.pbs
uv run python -m src.pnpl2026.trial15.cli.collate      # -> runs/trial15_results.md
```

To fuse an ensemble that includes I1/I2, run `cli.fuse_check` with its
members' run directories (trial 15's own, `runs/trial15_i*_s*`, have dumps
too) and `--fit-protocol reference --protocols reference fragment`: trial 13
and 15 dumps have no chunked protocol. It refuses dumps that do not list the
same tokens.

## What was checked (2026-10-06)

- **fusion = exact Bayes**: `fuse_run` against brute-force enumeration of every
  word sequence (runs of 1–4, 3–5 words): equal to 1e-15. Strength 0 returns
  the input; a run of one word is unchanged. On a planted chain with noisy
  scores, top-1 0.47 → 0.76
- **counting**: no pair across an unscored word, a sentence edge, or (without
  sentence ids) a gap over 0.8 s; gap arrays as counted
- **`cli.fuse` on the real holdout's layout** (960 rows, 154 runs of 2+):
  same index column and order; isolated rows, runs of one and Moses keep their
  top-10 sets. ⚠️ Caught: softmaxing fused log-likelihoods made rows so peaked
  that 6-decimal CSV rounding could tie ranks 2–10 at 0.000000, leaving the
  scorer's top 10 arbitrary. Fixed by z-scoring each row before the softmax
  (rankings unchanged); no row ties across rank 10/11
- **`cli.fuse_check`** on fabricated dumps from a known chain: +9–15 points on context words,
  temperature fitted, exact Bayes chosen on validation; on a chance-level run
  it warns that the temperature sits at the edge of its grid
- **shape check**, prime dimensions: per-window statistics over real samples
  only (mean 0, spread 1), padding exactly 0 and never read; a per-channel gain
  and offset change nothing with "instance" and do change "none"; batch of one
  equals the batch; backward reaches every parameter; fp16 autocast finite; a
  trial 13 network loads strictly and gives identical outputs with "none"
- **smoke** (158 s): I2 one epoch on the synthetic cache, end-of-run scoring
  with dumps; the checkpoint reloads with `input_norm: instance`; `predict` on
  the real holdout files; `fuse_check --fit-protocol reference` on its dumps;
  `collate`

## Results

Pending: `runs/trial15_fuse_check.md`, the fused 66.39 ensemble's leaderboard
score, `runs/trial15_results.md`, and I1/I2's 3-seed ensembles.
