# Variant D — holdout-matched, as implemented

A per-word encoder turns each word's window into one vector; a transformer then
runs across the words of a sentence, so every word's prediction can use the MEG
of its neighbours. Built on what the holdout actually serves, not on what the
design assumed — the difference is most of this document.

## The design's preconditions, checked 2026-09-30

| # | precondition | finding |
|---|---|---|
| 1 | sentence grouping and onsets at test time | **yes, for 868 of 960 words**: 156 sentences of 1–9 scored words. The other 92 are isolated 1 s words with no context |
| 2 | neighbours' MEG may be used | **unknown** — the rules page, FAQ and competition paper are silent. Ask the organisers (draft below) |
| 3 | bidirectional context allowed | **unknown**, same. D3 measures the causal version |
| 4 | learned sequence priors acceptable | **unknown**, and the rules say the holdout uses "stimuli that are entirely different from anything released" |
| 5 | ~64 words, chance ~16% | **no**: 50 words, chance 20%. Trial 6's shuffled run measured 0.204 |

## What the holdout serves, and what trial 7 trains on

Measured from `subj00_holdout01_sentence.npz` and `subj00_holdout2_word.npz`.

| | holdout | training and validation sequences |
|---|---|---|
| context tokens | scored words only — no other word has an onset | in-vocabulary words only |
| words per sentence | 1–9, median 5 | a sentence's in-vocabulary words, cut into near-equal runs of ≤ 9 |
| MEG after the last word | 1.13–1.68 s (5th–95th pct) | a tail drawn from U(1.1, 1.7) s; fixed once for validation |
| window per word | onset → sentence end: median 1.94 s, ≥ 3 s for only 4.4% | onset → min(onset + 3 s, sequence end) |
| isolated words | 92 of 960, 9.6%, 1 s each | 9.6% of words, 1 s each |

Validation sequences use a seed of their own, so every run and every seed is
scored on identical windows.

## The queue

`configs/trial7/runs.txt`: 11 runs × 3 seeds, most important first.

| lines | run | change from D1 | question |
|---|---|---|---|
| 1–3 | `d1_full` | — | the main result |
| 4–6 | `d0_control` | no sentence transformer; frame transformer depth 2 | a Variant B-style per-word model on the same windows |
| 7–9 | `d2_no_context` | sentence depth 0 | how much comes from cross-word context |
| 10–12 | `d9_shuffled` | training labels shuffled | the chance line |
| 13–15 | `d8_context_only` | every word predicted with its own vector masked | the honesty check |
| 16–18 | `d3_causal` | causal sentence transformer | what later words are worth |
| 19–21 | `d4_plain_siglip` | no deduplication | what the D in D-SigLIP is worth |
| 22–27 | `d5_depth2`, `d5_depth8` | sentence depth 2, 8 | transformer size |
| 28–30 | `d6_no_masking` | no word-token masking | does masking help |
| 31–33 | `d7_dilation32` | sixth trunk layer, reach 127 frames | a longer receptive field |

**Adopt Variant D if D1 beats D0 by more than the seed spread.** Read
`val/bacc10_sentence` and `val/bacc10_isolated` apart: context can only help
the first, so D1 against D2 should move it and leave the second alone.

⚠️ **D8 is not brain-free.** Words are ~0.3 s apart and windows up to 3 s, so
the previous words' windows contain this word's response. If D8 lands near D1,
the model leans on neighbours' windows rather than its own — not necessarily
on sequence priors.

## Results

All 33 runs complete (3 seeds each), from `runs/trial7_results.md` via
`cli.collate`. Cells are mean ± SD of `val/bacc10` over seeds; chance is 0.20.

| run | BAcc@10 | sentence | isolated | Δ vs D0 | rule | verdict |
|---|---|---|---|---|---|---|
| D9 shuffled | 0.205 ± 0.001 | 0.205 | 0.207 | — | — | chance line sits at the nominal 0.20: no pipeline bias |
| D0 control, no context | 0.362 ± 0.002 | 0.364 | 0.344 | — | reference | a Variant B-style per-word model |
| D2 no sentence transformer | 0.352 ± 0.003 | 0.354 | 0.331 | −0.010 | ≈ D0 | confirms D0's gain is not from its frame transformer |
| **D1 full** | **0.460 ± 0.002** | **0.474** | **0.341** | **+0.098** | **> D0 + spread** | **adopt — clearly beats D0** |
| D8 context only (own word masked) | 0.346 ± 0.003 | 0.361 | 0.204 | −0.016 | diagnostic | sentence words nearly match D0 from NEIGHBOURS alone; isolated collapses to chance, as it must |
| D3 causal | 0.418 ± 0.002 | 0.426 | 0.337 | −0.042 vs D1 | diagnostic | later words are worth ~4 points; bidirectional beats causal |
| D4 plain SigLIP (no dedup) | 0.467 ± 0.004 | 0.481 | 0.338 | +0.007 vs D1 | > D1 | marginal gain; the D in D-SigLIP is not earning its keep here |
| D5a sentence depth 2 | 0.448 ± 0.002 | 0.461 | 0.338 | −0.012 vs D1 | size | smaller, slightly worse |
| D5b sentence depth 8 | 0.464 ± 0.004 | 0.477 | 0.344 | +0.004 vs D1 | size | depth 4 (D1) is already near the plateau |
| D6 no word-token masking | 0.465 ± 0.001 | 0.478 | 0.345 | +0.005 vs D1 | > D1 | masking did not help; tightest SD of the queue |
| D7 dilation-32 trunk | 0.456 ± 0.009 | 0.471 | 0.323 | −0.004 vs D1 | ≈ D1 | no gain from a longer receptive field; ⚠️ one seed cut short, see below |

**Adopt rule, as `cli.collate` computes it:** D1 − D0 = +0.098, against a
largest seed SD of 0.002 for the two. That clears "beats D0 by more than the
seed spread" by roughly 50×, the largest effect measured anywhere in this
project.

**The gain is exactly where it should be.** D1 against D0 moves `sentence`
words by +11.0 points and leaves `isolated` words at +0.3 — noise, since
isolated words have no neighbours for the sentence transformer to read. D2
(no sentence transformer, same windows as D1) sits within 1 point of D0 on
both splits, confirming the windows themselves are not the source of the gain.

**D8 is the finding that needs the most care.** Predicting a sentence word
from its NEIGHBOURS alone, with its own window hidden, scores 0.361 on
sentence words — almost D0's 0.364, which uses the word's own window and
nothing else. Two readings are both consistent with this and cannot be told
apart from D8 alone:

- the neighbours' windows physically overlap this word's brain response
  (words are ~0.3 s apart, windows up to 3 s), so the model reads the
  response through a neighbour's window rather than the word's own; or
- the model has learned which words tend to follow which from the training
  transcripts, and is predicting from word order.

D3 (causal) bears on this a little: if D8's score were pure word-order
learning, a causal mask should cost little, since English word order is
mostly forward-predictable. It costs 4.2 points instead, which is more
consistent with genuine overlapping brain signal than with a pure language
model — but this is not conclusive either way.

**The ablations (D4–D7) are all within the queue's own noise band**, roughly
±0.01, except D3's causal mask. None is worth adopting alone; D4 and D6's
small, same-direction gains (dedup off, masking off) are worth combining in
one run together before trusting either individually.

⚠️ **D7's third seed was cut short** (epoch 18, walltime, best at epoch 17),
inflating its SD to 0.009 — the largest in the table. Rerun that line before
reading D7 as a clean negative.

**What this does not settle.** The 46.0% score depends on using MEG outside
each word's own 1 s window, which is the exact thing CLAUDE.md flagged as
unconfirmed by the rules (precondition 2, above). Until the organisers answer,
`--window-only` is the only submittable mode, and it scores like the isolated
split — about 34%, not 46%.

## Running it

```bash
qsub pbs/trial7_build.pbs            # once: data/trial7_continuous/, ~4.6 GB
qsub -J 1-12 pbs/trial7.pbs          # D1, D0, D2, D9 -- the adopt decision
qsub -J 13-33 pbs/trial7.pbs         # D8 and the ablations
PYTHONPATH=src python -m pnpl2026.trial7.cli.collate   # any time: runs/trial7_results.md
```

`cli.predict` has two modes. The default runs each holdout sentence whole.
`--window-only` gives every example only its own 1 s window and no context —
**the only mode known to be within the rules** until the organisers answer. D1
trains 9.6% of its words that way, so one checkpoint serves both.

## Where this departs from the design

| design | implemented | why |
|---|---|---|
| a 3 s window for every word | stopped at the sentence's end | 95.6% of holdout words have less |
| every word as context | in-vocabulary words only | the holdout gives no other word an onset |
| semantic loss on every word | in-vocabulary words only | the all-word version needs a t5 table for every word type and a cache of out-of-vocabulary windows — not built |
| sentences from punctuation, `W_max` 48 | the events' `sentenceidx`, ≤ 9 words | the table carries sentences; the holdout's longest has 9 |
| batch 16–32 sentences | 64 sequences, ~330 words | holdout-sized sequences are shorter |
| trunk "unchanged from B", reach 63 | 1,2,4,8,16, reach 63 | B's trunk was 1,2,4,8,1, reach 33; the design's 63 now fits 150 frames |
| D0 = B, frame depth 8 | frame depth 2 | trial 6's C1 tied depth 8 at 41% of the parameters; 150 frames make 8 three times the cost |
| baseline 0–0.5 s | as designed | B used 0–0.1 s; trial 6's C3 found no baseline +1.2. `with_baseline: false` is the switch |
| augmentation | jitter ±30 ms, component dropout, noise | as designed; B's time masking dropped |

Masked words are not scored in D1: masking is a regulariser, and scoring them
would train prediction from context alone. D8 scores them, since that is its
question.

## Known gaps between training and the holdout

- **Word spacing.** Holdout sentences look like consecutive vocabulary words
  (median gap 0.21 s). Training tokens sit in audiobook sentences, with
  out-of-vocabulary words between them. Their MEG is in every window; only
  their token is missing.
- **Sequence statistics.** Word order learned from the audiobooks meets
  sentences from "entirely different" stimuli.
- **Validation set.** The continuous cache holds what RDS holds: 156 of trial
  1's 162 recordings unless pnpl's validation and test partitions are also
  downloaded. Two validation recordings (TheMoth sessions 29, 30) are then
  missing, so trial 7's numbers are not on exactly trials 5–6's set.

## A question for the organisers (draft)

> For the Deep track, may a prediction for a holdout word use MEG outside its
> own 1 s window — the rest of its sentence in `holdout01_sentence.npz`,
> including the other words' windows? If so, may it use later words as well
> as earlier ones? And is a model that learns word-order statistics from the
> training transcripts acceptable?

## Cost

Laptop RTX 3060, 64 sequences a batch (~333 words), synthetic data. HPC times
are these scaled by spec, not measured.

| run | step | peak memory | 50 epochs, RTX6000 |
|---|---|---|---|
| D1, D3, D4, D6, D9 | 233 ms | 1.4 GB | ~1.7–2.1 h |
| D2 | 211 ms | 1.3 GB | ~1.5–1.9 h |
| D5b, D7, D8 | ~260 ms | ≤ 1.6 GB | ~1.9–2.4 h |
| D0 | 492 ms | 3.3 GB | ~3.6–4.4 h |
