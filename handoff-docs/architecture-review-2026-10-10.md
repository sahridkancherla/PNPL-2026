# Architecture review and proposal, 2026-10-10

Written five days before submissions close (15 Oct). Best file:
`ens_k18_spw_ts_hubh_lone`, **69.76** public, roughly 20th. The top five sit at
about 72.7 to 76.5, so the target is **+4 on the final-ranking partition**.

## Status, 2026-10-11

| | |
|---|---|
| **Done** | the review (sections 1 to 3), the proposal (section 4), four local measurements on the holdout files; **trial 21 built and smoke-tested** (`src/pnpl2026/trial21/`, `configs/trial21/`, `pbs/trial21.pbs`, `pbs/trial21_distil.pbs`): part A's `cli.distil` for both families, part B's `stem: dascoli` with the G2 warm start, a dual-family `cli.predict` with `--bn-adapt` under any scaling. Details and the decision rules: `src/pnpl2026/trial21/info/holdout-adaptation.md` |
| **Kamil's calls, open** | (1) run A0 (one submission) and, on its result, A2; the organisers called this kind of holdout use "against the spirit" though not prohibited; (2) queue B (`qsub -J 1-3 pbs/trial21.pbs`); (3) the partition check in C, which needs the leaderboard page or the organisers |
| **Not started** | anything on real MEG: the training scaler for trial 21, the A0 predictions, the A2 wave, B's three seeds |
| **Deferred, logged only** | 16 holdout gaps under 50 ms (2% of rows, near-duplicate onsets, nothing to do); a 100 Hz cache; a cleaner all-data retest with subject 0's val/test recordings only; B-cold (the hybrid from scratch, `configs/trial21/b_cold.yaml`, runs.txt lines 4-6) as a second wave |
| **Trial 20** | H, R, M, SPW+ were pending when this was written; their results slot into the schedule beside B |

Everything below is read from the trial records (`src/pnpl2026/trial*/info/`),
`configs/trial13/leaderboard.csv`, `configs/trial14/ensembles.csv`, the model
code of trials 18 to 20, and four measurements made today on the holdout files.

---

## 1. Where the points came from

Public leaderboard, 3-seed ensembles unless marked. The order is by size.

| lever | gain | evidence |
|---|---|---|
| **Every word a token, all-word t5 targets, 306 sensors + merger** (d'Ascoli recipe) | **+20** | E0 38.0 (single) to F1 58.0 (single); F2 with scored-only tokens 41.5 |
| **Holdout-shaped training items** | **+5.8 / +5.3** | G2 (10% lone 1 s words) 63.4, G1 (fragments) 62.8, against G0 57.6 |
| **Ensembling different recipes** | **+3 to +6** | 63.4 to 66.4 (8 files) to 69.8 (23 files) |
| Second target space (t5 layer 3, 18, 6) | +1.7, +1.5, +0.3, 0.0 | Kn, K2, K18, K6 in the mix; returns shrinking |
| Transductive hub correction (zscore fitted on the holdout's own 868 rows) | +0.9 | 66.1 to 67.0 |
| Continuous family (SPW, 3 models) | +0.7 | family alone 60.2 |
| No-context model N in the mix | +0.3 | N alone 33.2 |
| Onset-shift TTA | +0.25 | one row |

| what lost | cost | reading |
|---|---|---|
| **List-word-only token sequences**: L | **−23.6** | locally +15; a prior over list-word order learned from the training stories |
| Scored-only tokens: F2 vs F0; E0; Z1 | −8; ~40 absolute | the same thing at three strengths |
| Word-pair LM fusion, three versions | −0.8 to −3.6 | the same prior, injected at inference |
| Per-window input normalisation (I1, I2) | −6 to −7 | destroys the topography |
| All-data training (12 broad subjects + val/test, fixed epochs) | −8.6 | confounded: no selection, repeated sessions |
| 50-way head with logit adjustment; more fine-tuning | −1.8 to −4.2 | learns the training prior over the 50 words |
| 0.5 Hz high-pass | −3 (single seed) | the slowest band carries word information |
| No baseline + jitter + EMA (K0n) | −2 | a tie by the rule |
| Bigger transformer (w512, d16, wd, F0) | unresolved | single seeds differ by ±3 to 4 on the board; w1024 (53M) was the best single run at 61.3 |
| HDND refiner on frozen G2 | 0 locally | the encoder already uses what the first second carries |

## 2. The structural reading

1. **Supervision volume and target richness dominate.** The one +20 came from
   training every one of ~780k words against a semantic target, with hundreds
   of distinct negatives per batch. Capacity (F0 204M vs F1 5.85M) and the
   spatial front end (F3) moved nothing.
2. **Matching the holdout's format is the second lever**, and every recipe
   that did it gained (G1, G2, SPW) while every recipe that *also* learned the
   training stories' list-word order lost (L, fusion).
3. **Context is worth ~30 points on the holdout** (N 33.2 vs G2 63.4), and it
   is physical: the five windows of a 1.2 s run overlap almost completely, and
   the onsets are given. It is not linguistic; linguistic priors are poison
   here because the holdout stimuli are "entirely different".
4. **No local score tracks the board** because the local test partition
   shares the training corpora. Anything that exploits the stories (list-word
   order, fragment shaping) scores high locally and flat or negative on the
   board: the fragment protocol said +17 for G1 (board +5) and +15 for L
   (board −24). Decisions have had to be made on the board, at ±2 noise, with
   the best file being the maximum of ~70 tries. Expect the final partition to
   read one to two points below the public number for the same file.
5. **The single-recipe ceiling has been 63.4 since 3 Oct.** Every point since
   came from ensembling, and the ensemble is saturating: the last four
   additions gave +0.3, +0.7, −0.3 and −0.3.

A +4 step therefore has to come from a lever of the size of the first three.
There are only two left untouched, and both are about the holdout itself.

## 3. What the holdout files say (measured today, locally)

Files: `subj00_holdout01_sentence.npz` (156 epochs, 306 × up to 1030 samples at
250 Hz, onsets for up to 9 words) and `subj00_holdout2_word.npz` (92 × 306 × 250).

| measurement | result | consequence |
|---|---|---|
| Onset gaps (712) | 5/25/50/75/95%: 0.07/0.14/0.21/0.34/0.57 s; 31% of gaps ≥ 0.30 s; 16 gaps under 50 ms | consistent with trial 15's reading that roughly a third of neighbouring pairs span an unscored word; the 2% of near-duplicate onsets are unpredictable from timing |
| Session structure | per-sentence sensor-variance profiles: two halves of one sentence correlate 0.881, two different sentences 0.866; sentence vs lone-word covariance distance 3.1 | **one homogeneous recording session**, lone words included; holdout-wide statistics are coherent |
| Spatial covariance vs training | Riemannian distance holdout to training recordings median 13.9; training to training median 17.3; nearest: MOCHATIMIT ses-1/2, TIMIT ses-1 (3.9 to 5.1) | the holdout session is **not an outlier**; it most resembles the read-sentence sessions, which is also what the holdout stimuli are shaped like |
| Spectrum | 16% of power above 25 Hz, 20% in 12.5 to 25 Hz | the 50 Hz cache discards 16% of the power; untested on this pipeline |
| Public weight of lone rows | zero (reversal probe, 2026-10-07); the paper confirms two disjoint partitions, public and final, no sizes given | **all 92 lone words sit in the final-ranking partition.** 868/960 = 90.4%. If the public share is ~90%, the final ranking is decided on the 92 lone words alone |

## 4. The proposal: train on the holdout's own session and format

**Thesis.** Every whole-point gain closed a gap between what the network saw
in training and what the holdout serves. The gaps still open are the holdout
*session* (a new recording, never seen by any weight or BatchNorm statistic)
and its *token format* (list words only, with invisible neighbours and unknown
gaps). The 156 epochs and 92 words are the only data that have both, and no
run has ever trained on them. The organisers have said unsupervised use of the
holdout MEG is not prohibited; the project already does it for the hub
correction.

Three parts, ranked by expected gain per day. A and B are independent and can
run on the queue at the same time.

### A. Transductive self-distillation on the holdout epochs (the main bet)

**A0, the gate (no training, one submission).** Re-predict every member of the
best mix with `--bn-adapt` (exists in every `cli.predict` since trial 13;
lift the refusal under `--scaling training`), rebuild the mix, hub-correct,
submit. This re-estimates every BatchNorm layer on the holdout's own windows.
Trial 13 saw it lower local fragment scores, but those windows came from the
training sessions, where the statistics already match. Rule: above +0.5, the
session shift is real and A2 goes ahead; below −1, skip A2 and spend the GPU
on B.

**A1, the teacher.** The current best file's hub-corrected, z-scored scores,
turned into a 50-way distribution with one temperature chosen so that the
teacher's mean top-1 probability is about 0.3 (its likely top-1 accuracy).
Lone rows take the lone-word teacher (I3, hub-corrected). Nothing is fitted
on labels.

**A2, the students.** G2, K2, K18 (per-word family) and SPW (continuous), 3
seeds each: 12 short fine-tunes from `best.ckpt`.

- batches: 48 training items with the recipe's own loss and shaping
  unchanged, plus 16 holdout items (sentence epochs cut exactly as
  `gather_in_context` cuts them; lone words as 1 s items)
- holdout loss: KL(teacher ‖ student) over the 50 candidate scores
  (`candidate_scores`), weight 0.5
- BatchNorm in training mode on the mixed batches; EMA off; lr 3e-5 for the
  network (trial 14's value); exactly 3 epochs; no selection, because no
  local score can judge it
- predict every student as before; the students replace their parents in the
  mix; hub-correct on the holdout; lone-fill; submit

Rule: keep if the mix beats 69.76 by more than 1. If it gains more than 2,
run **A3**: one more round with the new mix as teacher. Stop after two rounds;
a third only confirms the teacher's mistakes.

What it buys: the encoder, merger and BatchNorm adapt to the holdout session;
the sentence transformer sees the holdout's sparse list-word tokens with
consistent targets, without ever being shown list-word *order* from the
training stories (the teacher is a function of MEG, not of text); each
student inherits the 23-model ensemble's ranking. Expected: +1 to +3 on top of
69.76. Risk: the teacher's errors are reinforced; the weight 0.5, three
epochs and the retained training loss bound that.

Cost: 12 runs of about 30 min on an RTX6000 plus predictions; one day of code
(`cli.distil`, reusing `predict`'s holdout cutting and `lit`'s losses).

### B. The hybrid: G2's encoder inside the continuous family (trial 21)

The continuous family is the structurally right model for the holdout: it
reads each epoch once, its context is time-resolved (steps at 25 Hz, onset
markers, rotary over real time rather than word index), and a lone word is
the same path. Its weakness is the stem: on full 3 s sentences it scores 60
to 65 against G2's 69 to 73, while on holdout-shaped local data it leads G2
by 6 to 17. Give it the proven encoder.

- `models/continuous.py`: a `stem: dascoli` option replacing the four-branch
  MEGState stem and the two LayerNorm blocks with `ConvSequence` (10 dilated
  layers, masked BatchNorm, GLU) and the `final` 1x1 pair, then the stride-2
  step conv as now. Reach ±1.24 s at 50 Hz, right for a stream
- **warm start from a G2 checkpoint**: `encoder.spatial`, `initial_linear`,
  `subject`, `convs`, `final` are key-for-key compatible; the time
  transformer, readout and heads train from scratch. The model starts where
  G2's features are and only has to learn the time-resolved context
- marker dropout: drop each onset marker at random with p 0.3 instead of
  trial 19's "keep only the list words' markers" with p 0.5, which is an
  identity-based rule of L's kind
- the tower and phoneme heads as SPW; 3 seeds; then predict, add to the mix

Expected: alone 63 to 66 (G2's encoder, the family's match to the holdout);
in the mix +1 to +2, more than SPW's +0.7 because it is both stronger and
still a different voter. Rule: keep in the mix if above +0.5; replace SPW's
three if it beats them alone by more than 2. Cost: half a day of code, 3 runs
of about 3 h.

### C. Settle which partition decides, before spending another GPU-hour

Read the public leaderboard page's "percentage of the test data" line, or ask
the organisers. 868 of 960 rows are sentence words.

- public share about 90%: the final ranking is the 92 lone words. Everything
  then goes to lone-word decoding: A on the 92 words with the lone teacher,
  an ensemble of the context-free recipes (I3, N, SW) on lone rows, the hub
  zscore (+8 locally on lone words), and no further work on sentence rows
- public share about 45 to 50%: lone words are about 17% of the final rows;
  the lone-fill already in place is worth about +1.5 there, and A should
  include the 92 words as students' items

Either way the public number overstates the final one for the current file,
because its members were chosen on the public rows.

### Schedule

| day | queue | login node |
|---|---|---|
| 10 Oct | A0 predictions with `--bn-adapt` | C; code `cli.distil` and the hybrid stem; A0 submission |
| 11 Oct | B 3 seeds; A2 12 students | A2 ensemble and submission in the evening |
| 12 Oct | A3 if A2 earned it; trial 20's H/R/M results folded in | B predictions, mix, submission |
| 13 Oct | buffer; re-runs | final-pick spread: files that differ in what they bet on (with and without A, with and without B) |
| 14 Oct | nothing new | choose the final picks; record everything |

### What not to spend the remaining days on

- more second target spaces (+0.3, 0.0 on the last two) or more seeds of
  existing recipes (every seed addition lost)
- a bigger transformer judged on one seed: unresolvable at the board's noise
- anything that learns list-word order from the training transcripts, at
  training or inference; anything with per-window normalisation; closed-set
  heads; further fine-tuning on training sessions alone
- a 100 Hz cache: plausible but a rebuild plus a retrain with no precedent,
  in five days

---

## Appendix: every architecture, one line each

| trial | input | per-word encoder | context | targets / loss | training format | board |
|---|---|---|---|---|---|---|
| 1, 2 | 73 whitened comps, 250 Hz, 1 s | ConvNeXt-style blocks + quadrature branch | none | 50-way CE | random windows, session split | val 0.35 (trial 2) |
| 4 | 73 comps, 125 Hz | multi-rate Conformer U-net | none | D-SigLIP t5 + CE + session adversary | windows | not submitted |
| 5 | 73 comps, 50 Hz | dilated conv trunk + 8 × 256 flat transformer (8M) | none | SigLIP + CE + adversary | windows | val 0.351 |
| 6 | ablations of 5 | depth 2 ties depth 8; 20 to 40 Hz envelope alone = chance; 0.5 s window −3.5; 100 Hz −1 | | | | |
| 7 | 73 comps | trial 5's trunk per word | sentence transformer 4 × 256 over **scored words only** | SigLIP + CE | holdout-shaped sequences | val 0.46 (no-context control 0.36) |
| 8 | 73 comps | same | attention-reach schedules: all tie | | | 38.0 in-context, 28.0 window-only |
| 9 | +0.5 Hz high-pass | | | | | 35.0 |
| 10 | **306 sensors + position merger, 50 Hz, 3 s** | d'Ascoli 10-layer dilated conv, BatchNorm, GLU, attention pool | 16 × 1024 (F0) or **4 × 256 (F1)** over **every word** | **D-SigLIP vs t5-large for every word** | whole sentences | F0 49.7, **F1 58.0**, F2 41.5, F3 51.3 |
| 11 | 73 comps | one stream per sentence, Zipformer multi-rate blocks | frame level | SigLIP + CE | | Z1 37.2, Z0 33.9 |
| 12 | F1 at six sizes | | | | | 49 to 61, no trend: seed noise |
| 13 | F1 | | | | **+ lone 1 s words (G2), fragments (G1)**, fragment-protocol selection | 57.6 / 62.8 / **63.4** / 60.9 |
| 14 | G2 + 50-way logit-adjusted head | | | | fine-tuned | 61.6, 61.6, 59.6 |
| 15 | per-window norm (I1, I2); word-pair fusion; lone specialist I3 | | | | | 56 to 57; fusions −1 to −4; I3 only on private rows |
| 16 | K0n: no baseline, jitter, EMA; Kn: + t5 layer-3 head | | | | | 61.4; 56.0 alone, +1.7 in mix |
| 17 | L: list-word runs as training items; hub correction | | | | | **39.8**; hub +0.9 |
| 18 | M HDND refiner; N no context; K2 / K6 / K18 heads; P phonology head; all-data | | | | | 0 local; 33.2; +1.5 / 0 / +0.3; +0.4; 54.8 |
| 19 | **continuous family**: MEGState 4-kernel stem, 2 LayerNorm blocks, stride 2 | 6 × 256 **time** transformer, onset markers, readout −0.5 to +1.5 s | | t5 + phoneme head + word tower | G2's recipe, items end as the holdout's | family 60.2; SPW +0.7 in mix |
| 20 | SPW + word transformer (H), 2.5 s readout (R), time masking (M), seeds 3 to 5 | | | | | pending |
