# Trial 20 — what the second note was worth, built on SPW

## Why

The best file is `ens_k18_spw_ts_hubh_lone`, 69.76 (10 Oct): the 69.08 mix
plus trial 19's SPW. A second architecture note (11 Oct) proposed eleven
changes to reach 73. Checked against what this project has measured:

| the note says | what we know | taken up as |
|---|---|---|
| the family "lacks the word-level transformer" | its time transformer already mixes words through the steps; a word-level stage after the readout is still the one piece G2 has and the family lacks, and Brain2Qwerty v2 has both stages | **H**: `word_depth: 2` |
| read out to +2.5 s | config only; holdout tails are 1.13-1.68 s, so only earlier words gain | **R** |
| replace the per-step LayerNorm | pre-norm blocks with a residual keep the raw amplitude; per-window input normalisation (I1/I2) lost 6-7 on the board | not taken |
| whole sentences only, few fragments | the family already trains G2's recipe; fragments are off | nothing to change |
| scale G2 to 8x512 or 16x1024 with EMA | on the board (trials 10/12): 204M F0 49.7 vs 5.85M F1 58.0; w512 49.1, wd 8x512 51.7, d16 54.8; K0n with EMA 2 below G2. The cited 73.23 is the organisers' local test split, not the holdout | not taken (the user agreed) |
| fine-tune and validate on TIMIT / MOCHA / Moth | all already in pnpl's train partition; MOCHA-TIMIT and TheMoth sessions are in the test partition; extra fine-tuning stages lost twice | **the validation half**: a by-corpus calibration (`cli/calibrate_tasks.py`) |
| family weights in the mix | right: 12 trial 19 models drowned the mix (-0.29), 3 lifted it (+0.68). No local set has ranked recipes like the board, so weights are submitted, not fitted | **`cli/ensemble.py --group --weights`** |
| drop the phoneme head | SPW (with it) is the best file; within noise either way | kept; every run builds on SPW (the user agreed) |
| time masking | the best external evidence (Feghhi 2025; Brain2Qwerty v2), already a switch | **M**: `time_mask_p: 0.5` |
| TTA, an LM prior, distillation, anchor swaps, 100 Hz | onset-shift TTA was noise (+0.25, -1.0); LM priors lost four times; no time | not taken |

## The runs (each SPW with one change; `configs/trial20/`)

| run | file | change |
|---|---|---|
| H | `h_word_transformer.yaml` | two word-transformer blocks after the readout |
| R | `r_readout.yaml` | `readout_after_s: 2.5` |
| M | `m_time_mask.yaml` | `time_mask_p: 0.5` |
| SPW+ | `spw_more.yaml` | SPW itself, seeds 3-5, as `trial20_spw` |

`trial20.yaml` is trial 19's SPW field for field (diffed), plus
`word_depth: 0`. `runs.txt` lines 1-12.

**The word transformer** (`models/continuous.py`): the readout's [N, d]
vectors are scattered into an [items, words, d] grid by each word's
position (`word_grid`, trial 18's `_grid` as a function), passed through a
`SentenceTransformer` of `word_depth` blocks (rotary over word order, every
word reads its item's words), and gathered back. `position` is already in
every batch. With `word_depth: 0` the network is trial 19's exactly; trial
19's checkpoints load under trial 19's model name.

**Family weights** (`cli/ensemble.py`): `--group NAME=a.csv,b.csv,...`
(repeatable) averages each family's z-scored files, then averages the
families by `--weights NAME=w` (default 1). Equal weights over equal-sized
groups reproduce the flat average exactly; `--inputs` alone is unchanged.

**By-corpus calibration** (`cli/calibrate_tasks.py`): for every board point
whose members have test dumps, the ensemble rebuilt locally (holdout-fitted
hub correction redone on the local tokens where the file had it), BAcc@10
on all sentence tokens and per corpus (Sherlock1, MOCHATIMIT, TheMoth), then
Spearman and the share of pairs ordered as the board orders them. Rows with
the same members and correction are one point.

## Decision rule (fixed 2026-10-11, before any score)

- each of H, R, M added to the 69.76 mix (hub-corrected, lone rows I3's):
  kept if above 69.76; SPW+ the same
- family-weighted mixes of TODAY's members, submitted alongside: one weight
  per family; the trial 19 family at 2x; at 0.5x. The best weighting is the
  base the new members join
- the calibration decides nothing alone: a corpus subset that orders the
  board's points at 80% of pairs or more becomes the first local proxy

## Running it

From the repo root on CX3, after `git pull`:

```bash
uv run python -m src.pnpl2026.trial20.cli.training_scaler     # seconds -> data/trial20_training_scaler.npz
uv run python -m src.pnpl2026.trial20.cli.calibrate_tasks     # minutes, login node -> runs/trial20_calibration.md
qsub -J 1-12 pbs/trial20.pbs                                  # H, R, M, SPW+ x 3 seeds
uv run python -m src.pnpl2026.trial20.cli.collate             # -> runs/trial20_results.md
```

Predict every run with trial 20's `cli.predict`.

## What was checked (2026-10-11, laptop, short runs only)

- shape check, distinct primes: the word grid round-trips exactly, empty
  slots invalid; `word_depth 0` loads trial 19's weights strictly and gives
  identical outputs; with the word transformer on: padding invariance, the
  lone word alone equals itself in a batch, moving one word's readout
  changes its item's other words and no other item's; refused without
  positions; every parameter gets a gradient with all heads on; batch size 1
- configs: the base equals trial 19's SPW field for field; each run file
  changes only its own key
- smoke, synthetic cache, one epoch: H with both heads (9.45M parameters;
  losses logged apart; four protocols scored); `cli.predict` on the real
  holdout files (960 rows)
- ensembles: equal groups of equal size equal the flat average (difference
  0); a 2:1 weighting matches a hand computation (5e-7, CSV rounding; the
  same top 10 in every row); a file in two groups, `--inputs` with
  `--group`, `--weights` without groups, and an unknown weight name are
  refused; the lone-word path works with groups
- calibration, fabricated dumps and leaderboard: duplicate rows merged
  (8 rows -> 7 points); a per-corpus score and a hub-corrected score equal
  hand computations; missing and frozen-hub points skipped with reasons;
  Spearman and pair order equal a hand loop; the command line end to end
- `bash -n pbs/trial20.pbs`; the ASCII-runtime-strings scan

## Results

### Calibration (2026-10-11): no local subset passes; non-Sherlock tracks the board better

83 board rows -> 69 points, 60 with every member's test dumps:

| protocol | subset | Spearman | pairs ordered as the board |
|---|---|---|---|
| reference | all | +0.12 | 52% |
| reference | Sherlock1 | **+0.02** | **50%** (a coin flip) |
| reference | TheMoth | +0.28 | 60% |
| reference | MOCHATIMIT | **+0.45** | **66%** |
| fragment | all | **+0.49** | **69%** |
| fragment | Sherlock1 | +0.44 | 65% |
| fragment | TheMoth | +0.45 | 66% |

None reaches the 80% bar, so the board stays the only judge. The note's
hypothesis holds in direction: full-sentence Sherlock scores carry no
information about the board, MOCHA-TIMIT (short isolated sentences) carries
the most of the reference subsets, and the holdout-shaped fragment protocol
does best overall. Use: a tie-breaker between files the board cannot
separate, never a ranking on its own.

### Board (2026-10-11): family weights and more trial 19 members both lost

Against the 69.76 flat mix (G1, G2, Kn, K2, N, F1, w1024, K18, SPW):

| file | board | against 69.76 |
|---|---|---|
| families at equal weight (`ens_fam_equal`) | 69.222 | -0.54 |
| SPW's family at 0.5x (`ens_fam_spw05`) | 69.222 | -0.54 |
| SPW's family at 2x (`ens_fam_spw2`) | 68.790 | -0.97 |
| + SW beside SPW (`ens_k18_sw_spw`) | 69.058 | -0.70 |
| + S and SW beside SPW (`ens_k18_s_sw_spw`) | 68.540 | -1.22 |

One vote per file stays best. The continuous family helps at about 3 of 23
votes; every file that raised its share lost (2x, +SW, +S+SW). Wave rule
consequence: test each new variant IN PLACE OF SPW, not beside it.

### The wave, local (2026-10-11): all twelve trained; nothing ranks them

Seed means, BAcc@10: FRAG test / frag val / ref test / ALONE test / train.

| run | FRAG test | frag val | ref test | alone test | train |
|---|---|---|---|---|---|
| H | 41.4 +/- 8.4 | 47.5 | 64.0 | 35.7 | 78.2 |
| R | 44.0 +/- 3.6 | 51.8 | 64.2 | 36.9 | 72.1 |
| M | 44.7 +/- 1.4 | 48.8 | 61.7 | 35.7 | 67.3 |
| SPW+ (s3-5) | 41.7 +/- 3.8 | 53.7 | 63.3 | 38.8 | 72.4 |

H s2 is the outlier: FRAG test 31.8 (s0 48.6) with a normal frag val (47.6)
and the highest train score (82.8); H fits the training set hardest. M
trains lowest (67.3), as masking should. Screen: every group healthy, seeds
agreeing 62-73% on the holdout; top-1 agreement with the reference H 48%,
M 49%, SPW 45%, R 44%. Check 2 against the 69.76 file: top-10 overlap 95.1%
(swap H), 96.3% (R), 96.1% (M), 97.8% (SPW6).

Submitted as swaps (each variant takes SPW's three votes) and SPW6 (six SPW
seeds sharing three votes), per the board result above.

### Board (2026-10-11): the wave lost; 69.76 stands

| file | board | against 69.76 |
|---|---|---|
| six SPW seeds as three votes (`ens_spw6`) | 69.222 | -0.54 |
| M in place of SPW (`ens_swap_m`) | 69.008 | -0.75 |
| R in place of SPW (`ens_swap_r`) | 68.722 | -1.04 |
| H in place of SPW (`ens_swap_h`) | 68.540 | -1.22 |

None of H, R or M beats plain SPW in its place, and the word transformer
(H) loses most -- the time transformer already does the word mixing.
SPW6 is the same recipe as the 69.76 file with three more seeds, and it
scores 0.54 lower: part of 69.76 is those three seeds, which the private
board will not necessarily reward. Final picks should therefore include
files that do not depend on one set of three seeds.

Nine submissions since the 69.76 file, all 68.5-69.2: the mix of these
families is at a plateau. Trial 20 is closed.
