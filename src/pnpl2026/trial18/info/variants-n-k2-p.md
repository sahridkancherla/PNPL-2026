# Variants N, K2 and P — what the research note was worth, tested

## Why

A research note (2026-10-08) proposed seven changes. Checked against what
this project has already measured:

| suggestion | already known | taken up as |
|---|---|---|
| drop the sentence transformer (its "timing shortcut") | each word's own 3 s window already holds the next word's onset, and the holdout gives onsets, so timing is real test-time information; but no no-context model of the current kind was ever scored on the board (trial 7, older model: -11 locally on sentence words) | **N** |
| w2v-bert-2.0 acoustic targets beside t5 | pnpl's LibriBrain100 ships no audio (TIMIT and TheMoth audio is licensed); a second, non-semantic target space is the one model change that ever helped the mix (Kn: 56.0 alone, +1.7 in the mix) | **P**, and **K2** as its control |
| Conformer backbone; MEG-XL, RVQ, criss-cross attention | capacity has never been the limit (F0 204M lost to F1 5.9M; trial 18's M had nothing to add); MEG-XL needs 150 s of context | not taken |
| SigLIP instead of InfoNCE | already the loss (D-SigLIP, d'Ascoli's) | nothing to do |
| time and sensor masking | sensor dropout is on (the merger's 0.2); onset jitter (K0n) did not help | not taken |
| class weights 1/sqrt(n) | D-SigLIP's dedup keeps one positive per word per batch, cutting the 121x imbalance to ~3x; the holdout-fitted hub correction rebalances predictions (+0.9 on the board) | not taken |

## The three runs (trial 18's code; each file changes G2 in one place)

| run | file | the change |
|---|---|---|
| **N** | `n_no_context.yaml` | `sentence_depth: 0`: every word from its own 3 s window alone, no attention across words |
| **K2** | `k2_layer3.yaml` | a second head on t5 layer 3 -- Kn's, without K0n's no-baseline, jitter and EMA |
| **P** | `p_phonology.yaml` | a second head on each word's sound shape (`cli/build_phoneme_embeddings.py`) |

P's table: every word's commonest pronunciation in the event tables (phoneme
rows linked by `sentenceidx` and `wordidx`, ordered by `phonemeidx`, TIMIT
labels through pnpl's `TIMIT_TO_ARPABET`, stress digits and silence
dropped), as HDND's form embedding: boundary-marked phoneme 1- to 3-grams,
each a fixed Gaussian vector seeded by an md5 hash, summed and normalised.
Rows are the t5 table's words in its order; a word no event table spells out
falls back to its letters.

## Decision rule (fixed 2026-10-08, before any score)

- each recipe alone (3-seed ensemble): recorded, not decisive -- Kn was 56
  alone and helped the mix
- each recipe added to the best mix (G1, G2, Kn, F1, w1024), hub-corrected
  on the holdout, against **67.00**: kept only if more than 1 above (one row
  of a single-row word is 2.000; less is a tie, and the simpler mix stays)
- K2 also in Kn's place; P and K2 together if both pass
- **N alone against G2 (63.41)** answers the note's main question: within 2,
  the transformer's local gain does not reach the board; far below, it does

## Running it

From the repo root on CX3, after `git pull`:

```bash
uv run python -m src.pnpl2026.trial18.cli.build_phoneme_embeddings   # minutes, CPU
ls data/word_embeddings_t5_layer3.npz data/trial18_training_scaler.npz
qsub -J 4-12 pbs/trial18.pbs                                          # N, K2, P x 3 seeds
uv run python -m src.pnpl2026.trial18.cli.collate
```

The builder's last lines name the closest-sounding scored pairs; a WARNING
there (many identical pairs) means the phoneme rows were not linked to their
words -- do not train P on that table.

## What was checked (2026-10-08, laptop)

- configs: each differs from trial18.yaml (G2) in exactly its own key(s)
- builder: synthetic events (one phoneme only, so every word identical, as
  expected) -- rows equal the t5 table's, none zero, two builds identical;
  a hand-made table -- "The" normalised, stress digits and `sil` dropped,
  `pcl` mapped to P, rows reordered by `phonemeidx`, the commonest of two
  pronunciations kept; "the" nearer its variant (0.44) than "people" (0.20)
- shape check, distinct primes: N's output is [5, 17], every parameter gets a
  gradient, batch size 1 runs, and word 0's output does not move when word
  1's window changes (no context); the second head's two outputs, gradients
- smoke, synthetic cache, one epoch each: N (2.69M parameters, none in a
  transformer) and P (second table loaded; `loss_first`/`loss_second`
  logged) trained, scored four protocols and wrote dumps; both predicted the
  real holdout files (960 rows)

## Results

Pending.
