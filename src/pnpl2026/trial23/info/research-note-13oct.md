# Trial 23 — the 13 Oct research note, built

The note: `handoff-docs/cross-subject-note-2026-10-13.md` ("Which cross-subject
and transduction ideas could plausibly add 2+ points"). Built 13 Oct.

## Why

Best file `ens_t22_tc1b75_ts_hubh_lone`, **70.682**: the 23-member mix ->
transduction v1 (pull 1, blend 0.75) -> holdout hub correction (zscore) ->
I3 lone rows. Part of it is luck: the blend curve peaks sharply at 0.75
(0.65: 70.18, 0.9: 69.68) and SPW's other seeds score 69.31. Trial 22's
cross-subject votes all lost; the row-specific diagnostic showed why -- they
echo subject 0's own timing (r = 0.78 with the mix). The note's remedy:
remove timing from the features, never link rows of one sentence,
cross-fit, and **gate every idea on a labelled replica of the holdout**
before it gets a slot.

User decisions (13 Oct): fresh seeds 3-5 (18 runs); MEG-XL on the A100 with
the RTX6000 as fallback; the decision rule is the user's own -- a file is
kept if it beats the current best on the board (the note's "<1.5 points is
noise" was declined: two points are not noise on this board); P6's parents
G1, G2, Kn, K2, K18 at seed 0.

**Facts that corrected the note.** All 14 of subject 0's TIMIT sessions are
in pnpl's training partition, so the note's "TIMIT test" was trained on;
the clean short-sentence and podcast checks are MOCHA-TIMIT (validation
ses-2, test ses-3) and TheMoth (ses-29, ses-30). Subject 0's Sherlock1
validation and test chapters are ses-11 and ses-12 **run 2**; the cached
broad subjects recorded ses-11/12 **run 1** -- the same audio is confirmed by
matching word onsets, not assumed (`cli/testbed_check.py`).

## Step 0 — the labelled Sherlock testbed (gates P1, P3, P8a, P2, P4)

- `cli/testbed_check.py`: subject 0's and each cached subject's ses-11/12
  word tables at 250 Hz, aligned by text (difflib); a subject counts when
  matched words keep one constant onset offset within a sample. Writes
  `runs/trial23_testbed_check.md` and `data/trial23_testbed_map.npz` (exact
  pairs, the 50 Hz offset, the other recording's key).
- New dump protocol `chains38` (`cli/embed.py --protocol chains38`): the
  chain protocol at 3-8 vocabulary words, at most one skipped word, gaps
  of 0.8 s or less; each token also carries `onset`, `key`, `table_index`.
- `cli/testbed.py`: ses-11 tunes, ses-12 tests. Subject 0's rows get the
  70.68 pipeline (transduction set up from the OTHER partition's reference
  dumps, so no row sees its own label; the hub fitted on the rows), then
  each cross-subject method from the 12 others' aligned recordings. Grids
  fixed in advance: C 20/30/40; alpha 0.7/0.8/0.9; P8a scale 0.5/1/2 over
  sqrt(K); weights 0.1-0.75. Also 3, 6, 12 subjects (5 draws), the
  sqrt(39/12) extrapolation as an upper bound, neighbour purity against the
  base rate (bar: 2x), split-half reliability, the echo with the base and
  with the gap voter. Writes `data/trial23_testbed_settings.json`.

## P1, P3, P8a — cross-subject votes without the timing echo

`crossfeatures.py` + `crossvotes.py`, one implementation for the testbed and
the holdout (`cli/cross_graph.py`):

- 50 Hz, 20 Hz zero-phase low-pass; MCCA (60 PCs per subject) to C shared
  components; windows -0.04..+0.46 s around each row's own onset, first
  4 samples' mean removed.
- **Timing residualization** per subject: ridge on position, reversed
  position and length one-hots, B-splines of log gap before/after,
  missing-gap and silence (>0.5 s) flags. Only residuals are kept.
- Same-sentence pairs never count. **P1**: each other subject's z-scored
  row correlations, averaged with reliability weights (correlation with the
  others' mean, floored at 0), kNN k = 10, power 3, symmetrised, label
  propagation from the mix's posteriors, posterior over share. **SNF**:
  Wang et al.'s cross-diffusion instead. **P3**: cross-fitted LDA per
  subject (5 sentence folds, PCA 100, Ledoit-Wolf, soft class means with a
  pseudo-count). **P8a**: the sum of those log-likelihoods x scale/sqrt(K).
- Holdout: `z(base) + w z(vote)` on the matched sentence rows only, w from
  the testbed; every other row keeps its ranking (checked). Candidates
  `<base>_x<method>.csv`, then hub + lone.

## P6 — boundary / short-sentence specialist

- `cli/boundary.py diagnose`: BAcc@10 for first, middle and last rows of
  MOCHA and TheMoth `chains38` (validation + test pooled, bootstrap over
  chains), plain and transduced; on the holdout, each family's seed
  agreement by position. No interval-excluding gap, no P6 (the note).
- Model: `ReferenceNet(boundary_features=True)` adds a zero-initialised
  `Linear(6, d_model)` -- is-first, is-last, log gap before/after over
  0.25 s, has-gap flags (`boundary.py`) -- to the word vectors before the
  sentence transformer, so the fine-tune starts as the parent exactly
  (shape-checked). Inputs come from token onsets in training and from
  `onset_s` on the holdout.
- `cli/finetune.py` (`pbs/trial23_finetune.pbs`, 5 parents): 3 epochs at
  0.3x the parent's learning rate, TIMIT + MOCHA at half of every epoch's
  items, the last epoch kept; then its `chains38` dumps and holdout CSV.
- `cli/boundary.py apply`: weight from {0, 0.25, ..., 2} chosen on MOCHA
  VALIDATION boundary rows; first/last rows `z(base) + w z(specialists)`,
  middle and lone rows exactly the base. `<base>_p6.csv`, then hub + lone.

## P7 — lone words (private board only)

- I3 seeds 3-5 (`configs/trial15/runs.txt` lines 13-15).
- TTA (`tta.py`): onset shifts -1/0/+1 samples (a lone epoch starts at its
  onset, so -1 repeats the first sample) and 8 passes with ONLY the spatial
  merger's sensor dropout in train mode; z-scored passes averaged. In
  predict's lone paths (`--onset-shifts -1 0 1 --dropout-tta 8`, files
  `_tta_dtta8`) and in embed's isolated protocol (`--lone-tta`, dumps in
  `runs/<run>_tta/dumps/`, the layout `cli.lone_check` reads). Locally the
  -1 shift reads the real sample before the onset, on the holdout a
  repeated one: one sample of 50.
- `cli.lone_check` now also reads a run's `embeddings/isolated_*.npz`
  (`configs/trial23/lone.txt`: I3 x6, SW, SPW, B), so every group is
  compared on identical rows; its validation winner fills the lone rows.
- The isolated-word files' `shuffle_seed` is never read.

## P2 — row-identification encoder (CL-SSTER-style)

`row_encoder.py` + `cli/row_encoder.py`. **Deviation, and why:** the note
trains one encoder on the 40 holdout files plus the testbed, but each MCCA
fit has its own component space. So the holdout's MCCA defines one space;
subject 0 heard both datasets, so its holdout projection reads its testbed
recording directly, and each other testbed subject is mapped onto subject
0's components by ridge regression fitted on ses-11 only. One encoder then
trains on the holdout and ses-11 together; ses-12 is never seen.

- Input: the 0.5 s residualized windows (C x 25), scaled per component.
  Model: per-time component mix -> MLP -> 64-d unit embedding.
- InfoNCE (temperature 0.1): positives are the same row in two subjects;
  negatives exclude same-sentence rows and rows sharing any of the base
  mix's top-3 words. No label read; stopped on held-out sentences.
- Gate: ses-12 neighbour purity of the 12 others' embeddings must beat the
  testbed's residual MCCA features. The four votes on the embeddings are
  tuned on ses-11, tested on ses-12. `cli.cross_graph --features
  data/trial23_row_encoder_holdout.npz --settings
  data/trial23_row_encoder_settings.json` -> `<base>_x<method>_enc.csv`.

## P5 — skip-aware sequence prior

`sequence_prior.py` + `cli/sequence_prior.py`; about 30% of holdout pairs
skip a non-vocabulary word, so both priors keep adjacent and one-skip
transitions apart and mix them per pair by a skip probability read from the
pair's onset gap (logistic on log gap and log gap against the chain's
median, fitted on validation pairs; `table_index` steps give the labels).

- **lm**: GPT-2 medium's own unconditional text (4M tokens, top-p 0.95 --
  nothing about the stimulus goes in, so the rules' grey zone does not
  arise), adjacent and one-skip pairs counted
  (`pbs/trial23_lm.pbs`, `data/trial23_lm_transitions.npz`).
- **em**: the evaluated rows' own posteriors, soft-counted over neighbouring
  rows, ONE pass cross-fitted over 5 chain folds (the note: no multi-round
  EM).
- Both as PMI (a transition's lift over the base rate, kappa = 50
  shrinkage), so neither adds frequency. Forward-backward with emissions
  T x z(base) (T by validation NLL) and transitions exp(w PMI), w in
  {0, 0.025, 0.05, 0.075, 0.1} chosen on pooled validation; w = 0 is the
  base exactly (checked against brute force).
- Gate: MOCHA test must improve (the note's; TIMIT is not clean);
  Sherlock ses-12 and TheMoth test reported beside it. `apply` refuses a
  prior whose check chose 0 -> `<base>_p5<prior>.csv`.

## P4 — MEG-XL as a frozen feature extractor

`megxl.py` + `cli/megxl_features.py` (`pbs/trial23_megxl.pbs`).

- MEG-XL (MIT; `meg-xl-med-v2.ckpt` from pnpl/MEG-XL; the repo's BioCodec
  tokenizer) loaded exactly as its README does, and run in fp32 as it runs
  it (the plan's bf16 is `--bf16`, opt-in: the tokenizer's codes are a
  nearest-codebook choice that precision can flip); only `einops` is
  installed beside the venv.
- Sensors: LibriBrain's fine-calibration file (`metadata/neo/sss_cal.dat`
  on the pnpl/LibriBrain dataset) holds every channel's position and coil
  axes; MEG-XL's own loader reads loc[9:12] (magnetometer) or loc[3:6]
  (planar gradiometer); positions normalised as its `norm_sensor_positions`.
- Input in MEG-XL's own LibriBrain form: one 3 s word window per row
  (-0.5 to +2.5 s), 50 to a 150 s segment, each window baseline-corrected,
  robust-scaled and clipped by MEG-XL's `preprocess_segment_with_subsegments`.
  Rows in one seeded SHUFFLED order: the note warns that MEG-XL's long
  context re-votes the timing prior. A row's features are the 3 tokens
  covering its 0.5 s, averaged over channels, then residualized like P1's.
- Gate: ses-12 purity against the residual MCCA features; the holdout's
  features are computed only when it passes (or `--always-holdout`).
  `cli.cross_graph --features data/trial23_megxl_holdout.npz --settings
  data/trial23_megxl_settings.json` -> `<base>_x<method>_megxl.csv`.

## P8b — session re-centering (Euclidean alignment)

`recentre.py` + `cli/recentre.py`. M = C_ref^1/2 C_sess^-1/2, both on
their top 90 eigendirections (SSS rank). C_ref: the mean covariance of the
training recordings in the cache's scaled space. A session is one holdout
source group (subject 0's sentence epochs; its lone words) or one cached
recording. `--recentre` in embed (dumps `_rc`) and in both predict paths
(files `_rc`). Gate: `cli.recentre check` -- the 70.68 recipe's pooled test
gain over every task's `chains38` rows; only then all members re-predicted.

## Fresh seed set B and seed averaging

`configs/trial23/fresh.txt`: G1, G2, Kn, K2, N, K18 at seeds 3-5, F1 s1,
w1024 s1, SPW s3-5 -- the 70.68 mix's shape. B through the 70.68 pipeline
measures the seed noise floor; A and B's transduced mixes averaged, then hub
and lone, is the seed average. Every winning method also runs on both.

## Submission order (the note's), judged by the user's rule

A file is kept only if it beats the current best (70.682); local gates
decide whether a candidate gets a slot at all.

| # | file |
|---|---|
| 1 | P6 on the 70.68 base (if the boundary gap is real) |
| 2 | P1 or P3 -- the testbed's winner -- on the 70.68 base |
| 3 | the 70.68 pipeline on seed set B (the noise floor) |
| 4-5 | P6 + the winning vote, on A and on B |
| 6 | P8a (if P3 won) or SNF (if P1 won) |
| 7 | P7's lone upgrade on the best file (public board blind: a no-break check) |
| 8 | P5 if it passed, else P2 if it passed purity, else P4 if it passed |
| 9-11 | the best pipeline on A, on B, and their average |
| 12-13 | final candidates |

## Checks run before any job (laptop, short, synthetic)

- Testbed and votes: residual purity 0.742 against raw 0.034 under a
  shared timing confound; echo +0.27 against +0.89; testbed gains +8.2 to
  +8.5; cross-fitting never reads a row's own posterior.
- P6: zero-initialised boundary layer reproduces the parent exactly; a
  1-epoch fine-tune runs; diagnose/apply route only first/last rows.
- P7: dropout switches only the merger; with no option both families'
  predictions equal trial 22's on all 960 rows; TTA dumps land beside the run.
- P5: forward-backward equals brute force on 50^3 paths; a planted Markov
  process gains +2.1 on test with lm and passes the gate; lone rows untouched.
- P2: one space works (ses-12 purity 22x the base rate); the gate compares
  against residual MCCA; cross_graph refuses an embeddings file of other rows.
- P4: MEG-XL's real code and tokenizer load by the README's recipe (0
  unexpected keys; only the RoPE table missing, by design); a tiny random
  transformer runs end to end and correctly fails the gate.
- P8b: the transform maps a session onto the reference exactly; embed and
  predict `--recentre` run for both families.
- Plain embed dumps equal trial 22's in every shared field.

## Results

**Board, 13 Oct -- averages of trial 22's transduced mixes** (no new job;
each average then hub + lone, as for 70.68):

| file | averages | board |
|---|---|---|
| `ens_t23_robust` | blends 0.65, 0.75, 0.9 + SPW seeds 3-5 at 0.75 | **70.849 (new best, +0.17)** |
| `ens_t23_blendavg` | blends 0.65, 0.75, 0.9 | 70.682 (tie) |
| `ens_t23_spwavg_tc1b75` | SPW seeds 0-2 and 3-5, both at 0.75 | 70.563 |

Averaging over settings and seeds held or beat the best single setting;
the base for trial 23's candidates can move to `ens_t23_robust_ts.csv`.

Pending: the testbed check, the embed waves, the jobs.
