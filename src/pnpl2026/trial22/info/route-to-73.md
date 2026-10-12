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

Pending.
