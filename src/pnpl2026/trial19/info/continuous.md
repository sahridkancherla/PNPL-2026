# The continuous family — encode the sentence once, read each word at its onset

## Why

Every member of the 69.08 mix (10 Oct) is the same network: one 3 s cut per
word, pooled to one vector, then a transformer across words. The 10 Oct
architecture note's first change is a new family: the whole item goes through
the network once, and a readout looks at the steps around each word's onset.
What it fixes, in the note's words and ours:

- **context is the biggest lever** -- N without the transformer 33.2, G2
  with it 63.4 -- and here the words share the transformer's full time
  resolution, not one pooled vector each
- **short holdout sentences waste the old design**: six 3 s cuts of a 2 s
  sentence cover the same signal six times, shifted, half of them zero
  padding; here the sentence is read once, as it arrives
- **a lone word is a 1 s item with one onset**, the same path, no special case
- **a new family is a different voter**: members that differ are what lifted
  the mix (Kn +1.7, K2 +1.5, K18 +0.3, N +0.3)

Evidence it rests on: Brain2Qwerty v2 (arXiv 2608.18114) ran a continuous
Conformer over MEG with an auxiliary head after the conv stem; its gap to
the event-locked design "collapsed to just 2%" at 10 h per subject (we have
~80 h). MEGState (arXiv 2512.17978) decodes LibriBrain phonemes with four
parallel convs of 500/250/125/62.5 ms: macro-F1 41.1 with them, 34.8
without. Neither tested this design on MEG word classification: a reasoned
bet, judged on the board.

## The note's checks (10 Oct, the real holdout files)

| check | result |
|---|---|
| does a sentence start at its first word? | 155 of 156 do; one has 1.46 s before it. No lead-in in training items either |
| is every word given? | onset gaps 5/50/95%: 0.07/0.21/0.57 s, 96.8% under 0.6 s. Mostly consecutive, a few skipped: unlabelled words are context, and need no marker |
| do the three sensors at a site share a position? | no: MNE's Vectorview-all layout gives 306 distinct centres (0.03 apart within a site, 0.07 to the next site). The type is still never given, so the merger gets a learned offset per sensor type, zero at start |
| "it's" | matched: `normalise_word` unifies U+2019 |
| TIMIT / MOCHA-TIMIT in training? | yes, pnpl's train partition spans all 12 tasks |
| independent directions | known: rank ~73-96 (CLAUDE.md) |

Holdout shape, for the design: epochs 1.95-4.12 s (median 2.52), the first
to last onset 1.17 s median, the tail 1.13-1.68 s after the last onset.

## The network (`models/continuous.py`)

```
signal [B, 306, T]                 the span, zero past each item's length
  clamp +/-5; one baseline per ITEM (the first 0.5 s)
  ChannelMerger + type offset       [B, 270, T]   reused, plus [270, 3] zeros
  1x1, SubjectLayer                               reused
  four parallel convs 500/250/125/62.5 ms, 64 each, 1x1 -> 256  MEGState
  2 blocks: LayerNorm per step, depthwise conv 7, 1x1 x4, GELU, 1x1, skip
  strided conv, stride 2            [B, 256, T']  25 steps/s; no BatchNorm
  phoneme head (SP, SPW)            [B, T', 40]   a teacher on the steps
  + marker at each marked onset
  time transformer 6 x 256 x 8      [B, T', 256]  SentenceTransformer reused
  readout per word: steps 0.5 s before to 1.5 s after its onset, each plus
    a learned relative-step vector, AttentionPool (reused)   [N, 256]
  heads: t5 (-> 1024), Variant K's second table, the tower's space (-> 256)
```

7.8M parameters (G2: 5.85M), 61% in the time transformer. The span is
padded to a whole number of steps, so a word in the last sample still has a
step; a step is valid when its first sample is real; a readout window is
never empty (its onset step always exists).

**Items** (`data/dataset.py`, `ContinuousItems`): the span runs from the
first kept onset to the last kept onset plus a tail -- the sequence's own
for fragments and lone words (exactly 1 s), drawn from 1.13-1.68 s for whole
training sentences (`whole_tail: holdout`), 3 s under the reference
protocol. Shaping is trial 13's (G2: a tenth of sentences as lone words).
Sentences are cut into runs of at most 16 words in training (`max_tokens`),
so spans stay under ~8 s. With `marker_drop_p` 0.5 a training item tells the
network only its scored words' onsets, so it never relies on knowing every
onset; every word stays a target. `time_mask_p` (off) blanks stretches.

**Phoneme head** (`cli/build_phoneme_labels.py`): every cache sample's
phoneme from the event tables (39 ARPAbet classes + silence); each step
predicts the label at its centre minus 150 ms, the brain's lag, with
cross-entropy at weight 0.5. Nothing at test time.

**Word tower** (`models/tower.py`, `cli/build_word_descriptions.py`): per
word, t5 layers 12, 3, 6 and 18, the sound shape (P's table), letter
n-grams and the median spoken duration, each block L2-normalised, through
LayerNorm -> 512 -> GELU -> 256, unit length. The brain side maps the readout
into the same space; D-SigLIP with its own scale and bias pulls matching
pairs together. The frozen t5 head's loss stays on (`tower_anchor_weight`
1.0), so training starts where G2's does. Scoring z-scores and averages the
heads present (Variant K's rule), or the tower alone (`tower_score_mode`).

## Departures from the note

| the note | here | why |
|---|---|---|
| 100 Hz input | 50 Hz | the existing cache and scaler; `cli/build_reference.py --sfreq 100` and `pbs/trial19_build.pbs` make the 100 Hz cache for wave 2 (`stem_stride: 4` then) |
| audio-derived targets | phoneme labels | pnpl ships no audio |
| a lead-in before the first word | none by default | the holdout has none (155 of 156) |
| time masking | a switch, off | one change at a time in wave 1 |

## Decision rule (fixed 2026-10-10, before any score)

- **S alone** (3-seed ensemble, `_ts_lone`) against G2's 63.41: how good the
  family is by itself -- informative, not decisive (Kn was 56 alone, +1.7)
- **each of S, SP, SW, SPW added to the 69.08 mix**, hub-corrected on the
  holdout, lone rows I3's: kept if above 69.08; the best such mix is the
  new best
- **S + SP + SW + SPW** as a separate family file for the final-pick spread
- SP against S and SW against S decide whether the phoneme head and the
  tower earn wave 2

## Running it

From the repo root on CX3, after `git pull`:

```bash
uv run python -m src.pnpl2026.trial19.cli.build_phoneme_labels      # minutes, login node
uv run python -m src.pnpl2026.trial19.cli.build_word_descriptions   # seconds
uv run python -m src.pnpl2026.trial19.cli.training_scaler           # seconds; 50 Hz as trial 18's
qsub -J 1-12 pbs/trial19.pbs                                        # S, SP, SW, SPW x 3 seeds
uv run python -m src.pnpl2026.trial19.cli.collate                   # -> runs/trial19_results.md
```

The builders' last lines say what they wrote (phoneme coverage; the
description blocks found -- t5 layers 3/6/18 and the P table are optional
but expected on RDS). Predict as trial 18, with trial 19's `cli.predict`
(`--onset-shifts` and `--window-only` kept).

## What was checked (2026-10-10, laptop, short runs only)

- shape check, distinct primes on every axis: contracts; the span padded
  to whole steps (an odd length, a word in its last sample); valid steps
  per item; padding invariance (23 extra padded samples change nothing);
  batch independence (a lone word alone equals itself in a batch); markers
  change their own item only; phoneme label indices lagged and clamped; the
  tower's unit candidates; every parameter gets a gradient with all three
  heads on; batch size 1
- `build_phoneme_labels` and `build_word_descriptions` on the synthetic
  data: labels int8 with the right classes; the optional blocks skipped
  with a note, the main table's word order kept
- smoke, synthetic cache, one epoch each: S and SPW (three losses logged
  apart, four protocols scored, dumps with the keys the tools read)
- learning: S at lr 1e-3 for 15 epochs on the synthetic cache's planted
  responses: chance until epoch 6, then 96-99% BAcc@10 on fragments and
  lone words by epoch 11. The readout-at-onset path trains end to end
- `cli.predict` on the real holdout files (960 rows): plain, onset shifts,
  window-only; `hubnorm_holdout` and `lone_check` read the run's dumps and
  CSV unchanged
- `bash -n` on both pbs files; the ASCII-runtime-strings scan

## Results

### Wave 1, local (2026-10-10), seed means; G2 and K2 for comparison

| run | frag val | FRAG test | ref test | ALONE test | train |
|---|---|---|---|---|---|
| G2 (trial 13) | 38.44 | 31.80 | 69.32 | ~29 | -- |
| K2 (trial 18) | 41.32 | 37.60 | 72.00 | 32.33 | 75.55 |
| **S** | 49.96 | 38.17 | 60.43 | 33.96 | 73.40 |
| **SP** | 49.65 | 42.43 | 59.62 | 35.77 | 73.99 |
| **SW** | 52.75 | **49.52** | **65.36** | **38.96** | 72.04 |
| **SPW** | **53.23** | 40.96 | 63.03 | 38.34 | 72.49 |

- every variant leads G2 on the holdout-shaped columns (fragments +11 to +15
  on validation, lone words +5 to +10) and trails it on full sentences with
  3 s windows (60-65 against 69) -- the shape it trained on is the
  holdout's, not the reference protocol's
- the word tower is the strongest addition locally (SW: FRAG test 49.5,
  ALONE 39.0); the phoneme head adds ~4 on FRAG test alone (SP vs S), but
  not on top of the tower (SPW vs SW)
- local columns have never ranked recipes like the board: the mix decides

### Board (2026-10-10): SPW lifts the best mix to 69.76

Each added to the 69.08 mix (G1, G2, Kn, K2, N, F1, w1024, K18),
hub-corrected, lone rows I3's:

| file | board | against 69.08 |
|---|---|---|
| **+ SPW** (`ens_k18_spw_ts_hubh_lone`) | **69.758** | **+0.68, the new best** |
| + SW | 69.508 | +0.43 |
| + S | 69.040 | -0.04 |
| + all four | 68.790 | -0.29 |
| + SP | 68.722 | -0.35 |
| the family alone (`trial19_all_ens_ts_lone`) | 60.185 | G2 alone 63.41 |

The word tower is what helps the mix: both variants with it gained, both
without it did not. The family alone is 3.2 below G2 -- a different voter,
as Kn (56.0) and N (33.2) were. Twelve of its models at once was too much
weight.
