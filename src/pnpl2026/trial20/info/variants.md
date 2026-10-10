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

Pending: the calibration, the wave, and the family-weighted mixes.
