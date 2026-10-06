# Variant I — a word-pair model, and per-window input normalisation

The best submission so far is **66.39**: G1 ×3 + G2 ×3 + F1 + w1024, one
recipe family (trial 10's word decoder) trained three ways. Adding more models
to it stopped paying (every addition lost 2–5 points, 2026-10-06), and
submissions close **15 Oct 2026**. Trial 15 makes two independent changes,
chosen from a literature search on 2026-10-06.

## Part 1 — the word-pair model (no training)

**What the models do not know.** Every holdout "sentence" is a run of 4–9
words that are *all* in the 50-word list (156 runs, 868 of 960 rows), most
neighbours directly adjacent in the speech and about a third with one
unscored word between (see Results). Every model so far trained on
narrative, where most neighbours are unscored words, so none knows that
structure. If word 3 of a run is "the", word 4 can only be one of the few list
words that follow "the" directly or one word later.

**The model.** `cli.wordpairs` counts, within sentences of subject 0's own
**training** transcripts (the cache's word tables — no MEG, no outside text),
how often each list word directly follows another (adjacent pairs) and how
often it is the next list word with exactly one unlisted word between
(skip-one pairs). Each table's rows are smoothed toward the overall
next-word distribution (pseudo-counts chosen on the validation transcripts),
and each kind's onset gaps get a log-normal model.

**Gap-aware chain** (the second version, after the first lost on the
leaderboard). Each neighbouring pair of a holdout run gets its own transition
matrix: the adjacent and skip-one tables mixed by P(adjacent | the pair's
onset gap). That probability comes from the two gap models, shifted by one
global speech-rate factor, and the share of adjacent pairs; both are fitted on
the holdout's own gaps by maximum likelihood (inputs only, never labels). The
first version, every pair adjacent, survives as `--transitions adjacent`.

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

`cli.fuse_check` copies the holdout's structure locally (section B): chains of
scored words in the chunked dumps that may skip one unscored word, each pair
given an onset gap drawn from the gap model of its true kind, the mixture
fitted on those gaps as `cli.fuse` fits the holdout's. It scores no fusion,
the adjacency-only chain, the gap-aware chain and an oracle told each pair's
kind, so the first version's failure should be visible locally too.

**Evidence.** Language-model rescoring is the step d'Ascoli et al. name next
for the decoder our models copy; "Unlocking Non-Invasive Brain-to-Text"
(arXiv 2505.13446) and a 2026 listened-speech MEG decoder (arXiv 2609.31997,
neural top-10 gating plus GPT-2 next-word probabilities) both gain from it.
Here the model is smaller and the constraint stronger: 50 words, all-list runs.

⚠️ A pair with two or more unscored words between is treated as skip-one;
only 0.6% of the holdout's gaps are over 0.8 s, so such pairs are rare.

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
# the table (rebuild after any change to wordpairs.py: the file format grew)
uv run python -m src.pnpl2026.trial15.cli.wordpairs --holdout-dir data/holdout
C=runs/trial14_calibration
uv run python -m src.pnpl2026.trial15.cli.fuse_check \
    --csv submissions/ens_g1g2_f1w1024_in_context.csv \
    --runs $C/trial13_g1_s0 $C/trial13_g1_s1 $C/trial13_g1_s2 \
           $C/trial13_g2_s0 $C/trial13_g2_s1 $C/trial13_g2_s2 \
           $C/trial10_f1_s0 $C/trial12_w1024_s0
# -> runs/trial15_fuse_check.md. The gate: in section B's (chosen) rows,
#    gap-aware above `before` on chunked validation AND test
grep "chunked | validation\|chunked | test" runs/trial15_fuse_check.md | grep chosen
# the fuse command the check wrote -> submissions/ens_g1g2_f1w1024_in_context_wpgap.csv
CMD=$(grep "cli.fuse --csv" runs/trial15_fuse_check.md); echo "$CMD"; eval "$CMD"

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
- **gap-aware version** (2026-10-06, second build): `fuse_run` with a
  different chain per step equals brute force to 1e-15; the gap mixture
  recovers a planted speech-rate shift and adjacent share (0.16 / 0.71 for a
  true 0.15 / 0.70) and, in `fuse_check`'s section B on fabricated dumps, the
  true adjacent share (0.72 fitted, 0.69–0.71 true); skip-one pairs counted
  only across exactly one unscored word; chains cross one unscored token and
  break at two or at a sequence edge; `cli.fuse --transitions gap` on the real
  holdout layout keeps every CSV check above, and warns when the mixture sits
  at the edge of its grid (gap models that do not fit the holdout)
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

### Part 1 — the fusion lowered both leaderboard scores (2026-10-06)

| submission | plain | fused (`_wp`) | change |
|---|---|---|---|
| G2 ensemble | 63.41 | 62.58 | −0.8 |
| G1 + G2 + F1 + w1024 | 66.39 | 64.09 | −2.3 |

Both moved down; each change is within about one leaderboard noise band
(~±2), but two of two in the same direction is a negative result: the
word-pair fusion as built does not help on the holdout. Recorded in
`configs/trial13/leaderboard.csv` with mode `in_context_wp`, so calibrate
leaves them out.

**Likely cause: the adjacency assumption is wrong for about a third of the
holdout's pairs.** `cli.wordpairs` on the real cache (97,618 training pairs):

| onset gap, s (5/25/50/75/95%) | | | | | |
|---|---|---|---|---|---|
| adjacent scored words, training | 0.06 | 0.10 | 0.14 | 0.18 | 0.32 |
| across one unscored word, training | 0.28 | 0.40 | 0.50 | 0.70 | 1.16 |
| holdout, inside its runs | 0.07 | 0.14 | 0.21 | 0.34 | 0.57 |

Only 5% of adjacent training pairs are over 0.32 s apart; over a quarter of
the holdout's are. As a two-part mixture that is roughly 30% of holdout pairs
with a skipped word between them (slower speech in the unrevealed sources
may explain part of the shift, not the heavy tail). The chain then imposed
"b directly follows a" where it does not hold, with full confidence.

The information itself is strong: the TRUE previous word alone names the next
at **66.3% BAcc@10** (validation pairs, prior divided out). A gap-aware
version -- per pair, the adjacent and the skip-one transition tables weighted
by how likely each is given that pair's onset gap -- is the fix, if pursued.

Pending: `runs/trial15_results.md` and I1/I2's 3-seed ensembles.
