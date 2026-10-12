# Trial 22 — the Route to 73, built

## Why

Best file `ens_k18_spw_ts_hubh_lone`, 69.76 (10 Oct); nine files since
landed at 68.5-69.2, and trial 21's self-distillation lost 3.6 (A2). On 12
Oct the organisers confirmed that the holdout data may be used in any way
before a submission is generated, including the other 39 subjects' files.
The team's "Route to 73" ranks six changes on that; this trial builds them.
User decisions (12 Oct): change 1 in full; change 3 a single
beside-parents probe of A2's students; change 5 the lone-word mix only;
change 2 on scores first, on embeddings next.

## Change 1 -- other subjects' recordings of the same sentences

**The repeat check (12 Oct): every other subject heard subject 0's
sentences.** All 39 files hold 156 sentences and 868 words; 154 sentences
match subject 0's uniquely with IDENTICAL onsets (0 ms); only the two
one-word sentences cannot be matched by timing. So 866 of 868 sentence
rows have 39 more recordings. Subject 0's own file repeats nothing. A first
look at subject 1 (downloaded): a different recording; matched epochs line
up exactly (same onsets, lengths within 2 samples, channel IQRs mostly
within 25%); raw sensor correlation on matched sentences +0.015 against
+0.012 on mismatched ones, best channels 0.10-0.14 -- a weak shared signal,
which is why the doc aligns before averaging.

| route | what | tool |
|---|---|---|
| 1 | the multi-subject model (trial 18's all-data G2, 3 seeds) on each subject that has a subject layer, through that layer, its own pooled scaling | `cli.predict --subject K --subject-layer auto --scaling holdout` |
| 2 | multiway CCA over subject 0 and 1-39; the shared-component count chosen by how well the others' mapped average predicts subject 0's held-out sentences; every subject mapped into subject 0's raw sensor space and averaged (weights: held-out correlation); written as a virtual holdout with subject 0's metadata, so every member predicts it unchanged | `cli.align`, then `cli.predict --holdout-dir data/trial22_virtual/mcca` |
| both | z(base + w * z(votes)) on matched sentence rows only; every other row keeps its ranking exactly | `cli.cross_votes` |

## Change 2 -- transduction (v1: each member's 50 word scores)

TransCLIP-style per member over the 868 sentence rows: one Gaussian cluster
per word (started at the word's validation prototype), the KL term towards
the member's own scores weighted by `--pull` (1 = TransCLIP; larger stays
closer), a graph of the 3 nearest rows in OTHER sentences, word shares
estimated and floored; blended 0.5 into the member's scores and ranked by
posterior over share (balanced accuracy's rule). Two pulls submitted, 1
and 2, with and without the hub correction.

Found on synthetic data and fixed before any submission: an exponent on
yhat (the first draft) sharpens instead of pulling; leaving the estimated
shares in the ranking costs accuracy. Caution: the estimated shares absorb
hub bias, so dividing them out already acts like a hub correction -- the
"without hub" file tests exactly that.

### v2, the embedding wave, and the kNN memory voter

`cli.embed` (GPU, `pbs/trial22_embed.pbs`, one member per line of
`configs/trial22/members.txt`) dumps each member's holdout rows -- gathered
exactly as predicted, checked to rank as the member's own CSV -- and the
CHAINS protocol on its train, validation and test partitions: only the
vocabulary words as tokens, chained across at most one unscored word as the
holdout lists them (`chain_sequences`, both families; identical tokens in
both). Never used for training: L trained on such runs and lost 24.

- `--space embedding` (v2): the clusters and the graph in the member's own
  embeddings (every head, unit length, side by side); prototypes = each
  word's mean training-chain embedding (the doc's); tau from validation
  chains; the local check on test chains, which are holdout-shaped
- `cli.knn_vote`: class-balanced soft kNN (K 32, temperature 0.05) of each
  holdout row against the member's training chains (capped at 2,000 tokens
  per word), heads averaged; the members' votes mixed into one file that
  joins a stack as one extra voter. Local check per member: test chains,
  kNN alone, member alone, both

## Change 6 -- the onset-gap voter

`cli.gap_voter`: per word, a Gaussian of log(gap to the next listed onset /
the sentence's median gap), fitted on subject 0's training events (250 Hz,
chained as the holdout lists words), re-fitted on the holdout by EM with
the best mix's soft labels; 712 of 960 holdout rows get a vote (not a
sentence's last word; sentences of 3+ words). Joins a stack at a 1/24
share. Report: gap-only BAcc@10 on validation and test chains, and the
label-free link between each row's gap and its top-1 word's typical gap.

## Change 3 -- A2's students beside their parents (probe)

One file: the 69.76 mix's 23 members plus the 12 blend students (35
files), hub-corrected, lone-filled. No new round unless it beats 69.76.

## Change 4 -- correction variants

`hubw` (the z-score leaving out each word's likely rows), `emsk` (BCTS on
validation, Saerens EM for the holdout's shares, Sinkhorn to them),
`hubpos` (first/middle/last, shrunk), `quota` (entropic transport with
share-proportional capacities, the top 10 read off the plan), `qbnorm`
(QB-Norm, frozen validation bank), plus the pending `hubb` (balance_half).
`cli.correct` writes each and checks each against the z-score on 200
uneven draws of non-Sherlock test tokens; the doc drops what loses there.
Judged as a group on the board.

## Change 5 -- lone rows

`cli.lone_check` over I3, SW, SPW and B; `lone_fill` with the validation
winner. Private board only.

## Decision rules (fixed 2026-10-12, before any score)

- a file is kept only if it scores above the current best (69.758)
- change 4's variants are judged as a group; if most lose, all are dropped
- cross-subject weights 0.15 and 0.30, fixed in advance
- lone rows: `lone_check`'s validation winner

## What was checked (2026-10-12, laptop, short runs)

- repeat check: a fake subject (subject 0's sentences permuted, 2 ms
  jitter) matched all 154 eligible sentences correctly; on synthetic files
  look-alike sentences are dropped as ambiguous, never mis-paired; then the
  real run over Hugging Face range reads (no MEG downloaded)
- MCCA on 6 synthetic subjects with 3 shared sources in shuffled files: the
  held-out choice picked C = 3 (0.336 against 0.001 for the raw average);
  the virtual epochs correlate 0.885 with subject 0's true shared signal;
  pnpl's loader serves the virtual holdout with subject 0's exact metadata;
  `shuffle_seed` is never copied
- cross_votes: route 1's row map round-trips exactly (115 rows); rows
  without votes keep their ranking (checked inside the tool)
- transduction: the graph never links rows of one sentence; deterministic;
  on planted clusters both pulls raise BAcc (75.1 -> 75.9 / 76.2)
- corrections: EM recovers planted shares (error 0.003); Sinkhorn margins
  exact; BCTS recovers a planted temperature; hubw at T -> inf equals the
  z-score; qbnorm equals a hand computation; the CLI end to end
- the ASCII-runtime-strings scan; `bash -n` on both PBS scripts

## Results

### Local (2026-10-12)

**Alignment (route 2):** 39 subjects, 154 sentences. Held-out correlation of
the others' mapped average with subject 0's MEG: C = 5 0.151, 10 0.147, 20
0.132, 30 0.114, 40 0.092, 60 0.018, 80 0.069; the raw average, unmapped,
0.089. Chosen C = 5; shared eigenvalues 20.4, 10.8, 5.2, 4.9, 4.7 (one
subject's worth = 1). Per-subject weights 0.053-0.096. Caution: five
components are mostly broad evoked responses; how much word identity they
carry only the members' predictions on the virtual holdout can say, so a
C = 30 version (`mcca30`) is built beside it.

**Transduction v1:** rows sharing a top-1 word are far closer than other
cross-sentence pairs (cosine 0.605 vs 0.141): the clusters exist. Pull 1:
seeds agree more in 5 of 7 families (G2 72.1 -> 73.4, K18 +1.1, SPW +0.9;
Kn -1.5), local check +0.62 +/- 1.71 (wins 60%), top-10 overlap with the
plain mix 82.0%. Pull 2: local -0.74 (wins 25%), overlap 85.4%. The
estimated shares FAIL the doc's plausibility check: the highest are new,
really, always, people (7.9-10.5%) and function words sit at the floor --
the shares measure hub bias, not frequency.

**Correction variants (200 uneven non-Sherlock draws, against the
z-score's 69.15):** every variant loses -- balance_half -0.94, hubw -0.05
(wins 46%), emsk -3.32, hubpos -1.30, quota -3.21, qbnorm -2.50. The EM
shares are implausible (always 31.7%, can 12.9%). By the fixed rule the
whole family is dropped before submission; `hubb` with it.

### Board (2026-10-12): transduction wins; cross-subject votes lose

| file | board | against 69.758 |
|---|---|---|
| transduction v1, pull 1, hub-corrected (`ens_t22_tc1`) | **70.147** | **+0.39** (new best) |
| route 2, MCCA C = 5, w 0.15 (`ens_x2w15`) | 69.425 | -0.33 |
| route 2, w 0.30 (`ens_x2w30`) | 69.358 | -0.40 |
| route 1, all-data G2 on 12 subjects, w 0.15 (`ens_x1w15`) | 69.243 | -0.52 |

Transduction is the first post-processing gain since the hub correction.
Both cross-subject routes lose, the larger weight more: at C = 5 the other
listeners' votes add noise faster than signal. In `ensembles.csv` a
pseudo-member (`+transduce...`, `+cross_votes...`) marks each file's
post-processing, so the calibration never mistakes it for the plain mix.

### Board (2026-10-12, late): the blend weight is the lever; the hub correction stays

| file | board | against 70.147 (pull 1, blend 0.5) |
|---|---|---|
| pull 1, blend **0.75**, hub-corrected (`ens_t22_tc1b75`) | **70.682** | **+0.54** (new best; +0.92 over 69.758) |
| pull 0.5, blend 0.5, hub-corrected (`ens_t22_tc05`) | 69.760 | -0.39 |
| pull 1, blend 0.5, NO hub correction (`ens_t22_tc1_ts_lone`) | 67.378 | -2.77 |

More of the transduced scores in each member helps; a looser pull (more
clustering away from each member's own scores) does not. Dividing out the
estimated shares is no substitute for the hub correction: without it the
file loses 2.8. Next: blend 0.9 and 1.0 at pull 1, then the winner as the
base for B, the gap voter and the kNN voter.

### Change 1 closed (2026-10-12, late)

The row-specific diagnostic (each word's mean over rows removed, matched
rows against shuffled): route 1 r = 0.770 (shuffled 0.015), route 2 r =
0.784 (shuffled 0.004); no collapse (top-1 covers 41-47 words). The other
listeners' votes are strongly row-specific, yet all three files lost: they
mostly echo subject 0's own picks. Every subject's file has identical onsets,
so position, sentence length and gaps are shared on every row and the
models read them; a vote that repeats the base adds its errors and little
else. The pass bar fixed for the C = 30 version (matched beats shuffled by
0.05) cannot tell such an echo from new information, so it is withdrawn:
change 1 closes as a negative result. The 30-component virtual subject, if
built, gets a slot only if one is spare.
