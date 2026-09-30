# Trial 4 — design: a multi-rate Conformer with a semantic head

**Design document, written 2026-09-26. Nothing here is implemented yet.**
Supersedes nothing; `trial3-run.md` still records the running trial-3 job.

This is the architecture proposal, the evidence it rests on, and the order to
build it in. It is deliberately long: the reasoning is the deliverable, because
the point is to stop guessing at architectures and start betting on the ones
the literature has already measured.

---

## 1. The gap we are trying to close

| | BAcc@10 |
|---|---|
| Chance | 20.0% |
| **Trial 2** (conv + transformer), epoch 6 | **34.7%** |
| **Trial 3** (pre-norm attention, 0.2–0.6 s, 100 Hz), epoch 6 | **27.6%**, plateaued |
| Kamil's "worth continuing" bar | 55% |
| **Competition reference model** (d'Ascoli et al.) | **73.23%** |
| Leaderboard leader | 74.7% |
| Kamil's goal | 70% |

**Trial 2 and trial 3 are not 40 points short because their encoders are too
small.** Trial 3 was *underfitting* at 4.3M parameters — `train/bacc10` sat
*below* `val/bacc10` for six epochs and the loss curve was flat at 3.72. More
capacity in the same shape does not close a gap like that. Something structural
is missing, and the literature says exactly what.

---

## 2. What the models that actually score well do

Two papers matter more than everything else, because one *is* the number we are
chasing and the other *won* last year's competition on this dataset.

### 2.1 The reference model — d'Ascoli et al., *Nature Communications* 2025

This is the 73.23% on the Deep track. Its recipe:

| component | what it is |
|---|---|
| Preprocessing | bandpass 0.1–40 Hz, **resample to 50 Hz**, robust scaling, clamp to ±5, baseline-correct over the first 0.5 s |
| Window | **3 s starting at word onset** |
| Front end | spatial attention over sensor positions (2-D Fourier basis) → subject-specific 1×1 → stack of dilated conv blocks |
| Backbone | **16-layer, 16-head transformer, d=1024, rotary positional embeddings**, attention dropout 0.1 |
| Pooling | single-head self-attention with unit output dimension |
| **Target** | **t5-large middle-layer embedding of the word, non-contextualised, d=1024** |
| **Loss** | **D-SigLIP** — SigLIP with in-batch duplicate words discarded from the loss |
| Optimiser | AdamW, lr 1e-4, batch 64, ~50 epochs, cosine decay, early stop on val balanced top-10 |

Its own ablations:

| comparison | result |
|---|---|
| CLIP vs SigLIP vs **D-SigLIP** | 15.7 → 14.7 → **18.2** top-10 (p<0.005) |
| Ridge → EEGNet → BrainModule | BrainModule ≈ **2× ridge** |
| BrainModule → BrainModule + transformer | **+50%** |

⚠️ **The reference model is not a classifier.** It is a retrieval model against
a language-model embedding space. That is the single biggest difference between
it and everything we have built.

### 2.2 The 2025 phoneme winner — MEGConformer (de Zuazo, Saratxaga, Navas)

Won the 2025 PNPL phoneme track on *this dataset*, with a Conformer, at 65.83%
F1-macro against a 6.10% baseline and 60.96% for 10th place.

| component | phoneme config |
|---|---|
| Input | **raw 306 channels at 250 Hz**, no PCA, no whitener |
| Front end | 1-D conv projection 306 → 144, dropout 0.1 |
| Encoder | **Conformer: 7 layers, 12 heads, d=144, FFN 2048, depthwise kernel 127** |
| Normalisation | **instance norm** — each channel normalised over the time axis *of its own window* |
| Loss | CE with **w ∝ 1/√n_c** |
| Optimiser | AdamW, lr 1e-4, wd 5e-2, batch 256, patience 10 |
| Inference | **5-seed majority vote** |

Its ablations, and they are the useful part:

| change | effect |
|---|---|
| **Instance norm instead of BatchNorm, on holdout** | **+200%** (+17.8 points) |
| Dynamic grouping loader instead of fixed groups | +13.3% relative |
| Class weighting | +7.6% |
| **5-seed ensemble, on holdout** | **55.09% → 65.82%, +19.5% relative** |
| Longer window (speech task, 0.5 s → 2.5 s) | +10.8% relative |
| Conformer instead of SEANet (speech task) | +9.0% |
| Removing augmentation | +0.01% — **not significant** |

⚠️ Two of the four biggest wins there are not architecture at all: **per-window
normalisation** and **ensembling**. Both are cheap. Both are absent from
trials 1–3.

---

## 3. The holdout has a door in it nobody has opened

Read `pnpl/competition/holdout.py`. The Deep-track holdout for `subj00` has
**two** sources, and they are not the same problem:

| source | file | what is stored |
|---|---|---|
| `word` | `subj00_holdout2_word.npz` | `meg (N, 306, 250)` — isolated epochs, exactly `[0.0, 1.0]` s. No context, ever. |
| `sentence` | `subj00_holdout01_sentence.npz` | `meg (N_sent, 306, T)` — **the whole sentence**, plus `word_onsets_s (N_sent, W)` for **every word in it**, plus `sentence_sample_mask` |

`iter_windows()` cuts 250 samples at each onset and throws the rest away. **The
rest is still in the file.** The submission contract is row *order*, which is
fully determined by `holdout.metadata` — and we are free to build our own reader
over that same ordering that hands the model the sentence, the onset, and the
onsets of every neighbouring word.

So for the `sentence` half of the rows we legitimately have:

- **pre-onset baseline** (the reference model baseline-corrects over 0.5 s; on
  isolated epochs we cannot, on sentence rows we can)
- **post-onset context past 1 s**, up to the 3 s the reference model uses
- **the positions of the neighbouring words**, so a sentence can be decoded
  jointly rather than word by word

⚠️ **First action before committing to this: run `holdout.counts()`.** It needs
only subj00's two `.npz` files. If `sentence` is 10% of the rows, the context
bet is worth ~a tenth of what it looks like. If it is 80%, it is the most
valuable thing in this document. **We do not currently know which.**

---

## 4. The five bets, ranked by expected value per unit of work

| # | bet | why it should work | cost |
|---|---|---|---|
| **1** | **Replace 50-way CE with a semantic embedding head** trained contrastively against t5 embeddings, over the **open** vocabulary | It is what the 73.23% model does; it turns 121× class imbalance into a shared regression problem; it unlocks the **467k out-of-vocabulary words we currently discard** | medium |
| **2** | **Per-window robust instance norm + warmup/cosine at lr 3e-4** | Winner's largest holdout gain; and trial 3's flat loss at lr 1e-4 with no warmup is the textbook pre-norm-attention failure | **tiny** |
| **3** | **Phoneme CTC auxiliary head** | The phoneme content of these windows is decodable at 65.8% F1 — proven, on this dataset. The labels are already in the events TSV and cost nothing | small |
| **4** | **Multi-rate Conformer encoder** replacing the flat stack | Conformer beat SEANet by 9% on this data; Zipformer's U-Net multi-rate structure beats flat Conformer at *lower* cost | medium |
| **5** | **Sentence context + joint sentence decoding** | Free information sitting in the holdout file; the reference model's 3 s window is worth ~10% relative on its own | large, and gated on §3 |

Plus two that are not bets at all, just unclaimed points: **5-seed ensembling**
(+19.5% relative, measured) and **jitter TTA**.

---

## 5. The architecture

Name it `trial4`, model string `sentence_conformer`.

### 5.1 Shapes end to end

`B` = batch, `F` = frames. Full variant, word mode, 1 s core.

| stage | in | out |
|---|---|---|
| raw window | — | `[B, 306, 250]` @ 250 Hz |
| `RobustInstanceNorm` | `[B, 306, 250]` | `[B, 306, 250]` + scale side-channel `[B, 306]` |
| `FixedLowpassDecimate` (40 Hz, 81 taps, 250→100 Hz) | `[B, 306, 250]` | `[B, 306, 100]` |
| `SpatialAttention` (306 → 270) | `[B, 306, 100]` | `[B, 270, 100]` |
| `stem` 1×1 + BiasNorm | `[B, 270, 100]` | `[B, 192, 100]` |
| **stack A** — 3 Conformer blocks, d=192, dw k=31 (310 ms) | `[B, 192, 100]` | `[B, 192, 100]` |
| downsample ×2 | | `[B, 256, 50]` |
| **stack B** — 4 blocks, d=256, dw k=15 (300 ms) | `[B, 256, 50]` | `[B, 256, 50]` |
| downsample ×2 | | `[B, 384, 25]` |
| **stack C** — 4 blocks, d=384, dw k=9 (360 ms) | `[B, 384, 25]` | `[B, 384, 25]` |
| upsample ×2, add skip from B | | `[B, 256, 50]` |
| **stack D** — 3 blocks, d=256, dw k=15 | `[B, 256, 50]` | `[B, 256, 50]` |
| `AttentionPool` (single head, unit output) | `[B, 256, 50]` | `[B, 256]` |
| `semantic_head` | `[B, 256]` | `[B, 1024]` |
| `closed_set_head` | `[B, 256]` | `[B, 50]` |
| `phoneme_head` (off **stack A**, 100 Hz) | `[B, 192, 100]` | `[B, 100, 40]` |

Every depthwise kernel is chosen so its **span in milliseconds stays roughly
constant (~300 ms) across the rate change**, which is why the kernel shrinks as
the frame rate drops. A constant kernel size across a multi-rate stack silently
triples the receptive field per stack and is a common way to get a model that
only sees slow drift.

### 5.2 `RobustInstanceNorm` — the cheapest big win

Per channel, per window: subtract the **median**, divide by the **IQR/1.349**,
computed over that window's own time axis. Then clamp to ±5.

Why it matters here more than in speech: our holdout is a **held-out recording**
of the same subject. Head position drifts between recordings, sessions differ in
size by 30×, and SSS gain varies. Any statistic fitted on the training
recordings is the wrong statistic at test time. Per-window normalisation has no
train/test statistic at all.

⚠️ It also **destroys absolute amplitude**, which is real information. So keep
`log(IQR)` per channel as a `[B, 306]` side-channel, projected to `d_model` and
added to the stem output as a static bias. Nothing is thrown away; it is just
no longer allowed to set the scale of the network's input.

This replaces `ChannelStandardiser` fitted on the training split, and it
replaces every BatchNorm in the network (use BiasNorm or GroupNorm inside
blocks — trial 3 already reached the same conclusion for the same reason).

### 5.3 `SpatialAttention` — and the departure from the rank-73 whitener

Trials 1–3 feed the model 73 PCA components from a whitener fitted on the
training recordings. The winner fed **raw 306 channels** through a learned 1-D
projection and won. The reference model learns a **spatial attention layer
parameterised over sensor positions**. Both learn their spatial front end;
neither fixes it in advance.

Proposed: Défossez's parameterisation. For output channel `j` and sensor `i` at
2-D layout position `(x_i, y_i)`, the score is a learned sum of 2-D Fourier
harmonics evaluated at `(x_i, y_i)`; softmax over `i` gives the weights. 270
output channels. Parameter count is set by the number of harmonics, not by
306×270, so the layer is small and **smooth over the helmet** — which is the
right prior, since a neighbouring sensor sees a neighbouring field.

**Sensor positions are available:** `mne` 1.13.2 is installed, and
`mne.channels.read_layout("Vectorview-all")` returns 306 names with `(306, 4)`
positions.

> ⚠️ **The channel order almost certainly does not match.** `Vectorview-all`
> starts `MEG 0113, MEG 0112, MEG 0111` — that is **grad, grad, mag**. `CLAUDE.md`
> records the h5's `channel_types` as **`mag, grad, grad` repeating**. Sorting
> the layout names numerically (0111 before 0112 before 0113) recovers
> mag-first order, but **this must be verified, not assumed**: check that the
> 102 channels the h5 calls `mag` map onto the 102 layout names ending in `1`,
> and sanity-check that spatially adjacent sensors under the mapping are also
> the most correlated in the data. A wrong permutation here gives a model that
> trains fine and has a scrambled spatial prior — silent, as usual.

Fallback if the mapping cannot be established: a plain learned 1×1 conv
306 → 270, exactly what MEGConformer used. It loses the smoothness prior and
nothing else.

### 5.4 ⚠️ No per-session layer — session-adversarial instead

The reference model has a **subject-specific** 1×1 because subjects are known at
test time. Our analogue would be a per-session or per-recording layer, and
**it cannot work**: the Deep-track holdout is a recording we have never seen,
with no session id. A per-session parameter has nothing to load.

So invert it: make the encoder **session-invariant** rather than
session-conditioned.

- a small session classifier on the pooled embedding, trained through a
  **gradient reversal layer** (coefficient ramped 0 → 0.1 over the first 5
  epochs — ramping it from the start destabilises early training)
- instance norm already removes the easiest session cue (gain)

This is a genuine departure from the reference architecture, forced by the
evaluation protocol, and worth stating loudly because copying the subject layer
across would look right and do nothing.

### 5.5 The Conformer block

Standard Gulati block, macaron, half-step residuals, with two changes.

```
x = x + 0.5 * FFN_1(x)
x = x + MHSA(x)              # rotary relative positions
x = x + ConvModule(x)
x = x + 0.5 * FFN_2(x)
x = BiasNorm(x)
```

`ConvModule`: `BiasNorm → pointwise (×2) → GLU → depthwise k → BiasNorm →
SwooshR → pointwise → dropout`.

**Change 1 — keep the quadrature branch.** Trial 1's `QuadratureBranch` is the
one original idea in this repo with a real physiological argument: single-trial
MEG *phase* is not reproducible across repetitions of a word, induced *power*
is, and every linear filter is phase-sensitive. Put it inside the conv module as
a second depthwise path whose `sqrt(a² + b²)` output is concatenated before the
output pointwise conv. Cost is one extra depthwise conv per block.
`with_quadrature: false` remains the ablation.

**Change 2 — Zipformer's structural borrowings**, all of which are free:

| borrowed | instead of | why |
|---|---|---|
| U-Net multi-rate stacks (§5.1) | one flat frame rate | ICLR 2024: better *and* faster than flat Conformer |
| **BiasNorm** | LayerNorm | retains length information; removing it costs 0.08–0.18% WER |
| **SwooshR / SwooshL** | Swish | measured better |
| **attention-weight reuse** within a stack | recomputing per block | halves attention cost — and this machine is thermally bound |
| **ScaledAdam** | AdamW | faster convergence; scales each tensor's update by its own scale |

⚠️ ScaledAdam is the one I would hold back initially. It is a second variable
moving at the same time as the architecture, and trial 3's lesson was that
moving window, rate and model together makes the result unattributable. Land the
architecture on AdamW + warmup + cosine, then test ScaledAdam as its own run.

### 5.6 Sentence mode — one encoder, two input regimes

The same front end and encoder run over a whole sentence, `[B, 306, T_sent]`
with a validity mask. Then, for each word onset, a **query token** — a learned
vector rotated to the onset frame's rotary position — cross-attends over the
sentence's encoded frames (2 cross-attention layers) and produces that word's
pooled embedding.

- **Word mode** is the degenerate case: sentence = the 1 s epoch, one query at
  frame 0.
- Train with **context dropout**: with probability 0.5, truncate the sentence to
  the bare `[0, 1]` s core. One checkpoint then serves both holdout sources
  honestly, instead of a sentence model that has never seen an isolated epoch.
- The pre-onset margin is what makes baseline correction possible on sentence
  rows; the mask tells the model when there is none.

**Joint sentence decoding, as a post-hoc reranker.** Sentence mode gives a
`[W, 50]` probability matrix per sentence. A word-bigram prior estimated from
the training transcripts, run as a Viterbi pass over the sentence, reorders it.

⚠️ Be clear-eyed about this one: the metric is **top-10 of 50**, which is
permissive, so a reranker has to change the *top-10 set*, not just the argmax,
to earn anything. It is cheap and needs no retraining, so it is worth measuring
— but it is the last thing to build, not the first.

### 5.7 Parameter budget

| variant | blocks / widths | params |
|---|---|---|
| **full** | 3/4/4/3 at 192/256/384/256 | **≈26M** |
| **laptop** | 2/3/3/2 at 128/160/192/160 | **≈6M** |

Trial 3 was 4.3M and underfitting, so even the laptop variant is a capacity
increase, and the full variant is in the same range as the reference model's
transformer.

---

## 6. The loss — three heads, and why each is there

### 6.1 Primary: D-SigLIP against t5 embeddings

- **Target**: t5-large, **middle layer**, the word processed **on its own**
  (non-contextualised), 1024-d; multi-token words averaged. Precompute a lookup
  table for every word type in the corpus once — ~20k types × 1024 floats ≈ 80 MB.
  ⚠️ `transformers` and `sentencepiece` are **not installed**; this is a
  one-time install and a one-time download, on a machine with internet.
- **Loss**: sigmoid pairwise loss over the batch with learnable temperature and
  bias, and — this is the *D* — **drop pairs where two batch rows carry the same
  word**. With `the` at 41,143 occurrences, a batch of 256 contains it ~13
  times; naive in-batch contrastive treats those as negatives of each other,
  which is precisely the frequency artefact D-SigLIP was introduced to remove.
  Their ablation: 14.7 → 18.2.
- **Inference**: cosine similarity between the predicted embedding and the 50
  vocabulary embeddings → 50 scores → softmax → the submission's probabilities.

**Why this is bet #1.** Three separate reasons compound:

1. **It shares statistical strength across classes.** `people` has 341 examples.
   A 50-way softmax must learn its column from those 341 alone. An embedding
   regression learns `people` partly from `family`, `they`, `we` — everything
   nearby in t5 space.
2. **It opens the vocabulary.** 780,277 word events exist; only 313,068 (40.1%)
   are in the scored 50. Cross-entropy cannot use the other 467,209. A
   contrastive embedding head can use **all of them**, because the target is an
   embedding, not a class index. That is **2.5× the training data**, from data we
   already have on disk. The reference paper's own headline is that performance
   rises with training data, and that the model "can decode words absent from
   the training set."
3. It is what the 73.23% does.

### 6.2 Auxiliary: 50-way weighted CE

Keep it. `w ∝ 1/√n_c` (the winner's choice, and already this repo's
`class_weight_power: 0.5`), label smoothing 0.1, computed on in-vocabulary rows
only. It optimises the scored metric directly, and at inference the two heads'
probability vectors get **calibrated averaging** — weight chosen on validation,
not assumed.

### 6.3 Auxiliary: phoneme CTC

Off stack A's 100 Hz sequence, 39 phonemes + blank, CTC against the phoneme rows
of the events TSV for the word's span. Loss weight ~0.3.

Why: it is dense supervision (3–6 phonemes per window versus one label), it is
**free** — the labels are already in the TSV — and it is *proven decodable on
this exact dataset* at 65.8% F1. It forces the encoder to represent
acoustic-phonetic structure rather than memorising a 50-way lookup, which is the
failure mode a 121×-imbalanced classification loss actively encourages.

⚠️ Phoneme onsets come from `timeds`, the same column as words, for the same
reason: `timemeg` is off the 250 Hz grid 6% of the time.

Total: `L = L_dsiglip + 0.3·L_ce + 0.3·L_ctc + λ(t)·L_session_adv`.

---

## 7. Self-supervised pretraining — the long lever

~80 hours of continuous MEG exist for `sub-0`. The word windows use a fraction
of it. Pretrain the front end + encoder on **continuous stretches**, then
fine-tune the heads.

- **Objective: data2vec-style masked latent regression** against an EMA teacher.
  Chosen over wav2vec 2.0's quantised-codebook contrastive because there is no
  codebook to collapse and no negative sampling to tune, and it is stable at
  small batch — which is what a laptop or a single RTX 6000 can give.
- **Mask spans in time** (~10 frames, 6.5% of starts) **and mask whole
  sensors**. Sensor masking is the MEG-specific half: SSS makes the array
  genuinely rank-limited, so predicting a dropped sensor from its neighbours is
  a real task with a real answer.
- Long context: 10–30 s stretches, not 1 s windows. This is what MEG-XL reports
  buys data efficiency for brain-to-text.

⚠️ This is the most expensive item in the document and the least certain. It is
**stage 6 of 7** for that reason. Do not start here.

---

## 8. The data layer must change — no window cache

The current cache is 313k × 73 × 400 fp16 = 17 GB. The same cache at **306
channels** would be **77 GB**, against **80 GB free** on the internal disk. So
the front-end change alone kills the cache — and separately, a cache of pre-cut
windows forecloses sentence context (§5.6), the open vocabulary (§6.1), and SSL
(§7), all three.

**Replace it with on-the-fly windowing from the continuous recordings.**

- index the 162 events TSVs once into a compact table: `(recording, start_sample,
  word_type_id, sentence_idx, word_idx, duration, phoneme_spans)`
- read windows from the h5 with `h5py` slicing; one recording is
  306 × 332,250 × 4 B ≈ 407 MB, so keep a handful resident
- **recording-local batching** — sample K=8 resident recordings and interleave
  windows from all of them. Keeping only one resident is faster but makes every
  batch one session, which poisons in-batch contrastive negatives and makes the
  session-adversarial head trivial to fool.
- this is MEGConformer's "dynamic grouping loader" with a reason: their ablation
  gave it **+13.3% relative**, the largest non-normalisation gain they report.

⚠️ **The dataset is on an external volume and it is not mounted right now.** Any
plan that reads continuously from it depends on that drive being attached for
the whole run. That is a real operational risk for a 10-hour job and it belongs
in the go/no-go, not in a footnote.

⚠️ `iter_windows()` ordering rules from `CLAUDE.md` apply unchanged to whatever
custom holdout reader §5.6 needs. Row order comes from `holdout.metadata` and
nothing may sort, filter or reorder it. The count check in `predict.py` stays.

---

## 9. Augmentation, regularisation, optimisation

| knob | value | note |
|---|---|---|
| time jitter | ±100 ms | already have it; needs the pre-onset margin, which on-the-fly windowing gives for free |
| **sensor dropout** | drop whole sensors, p=0.1 | the MEG-native augmentation: SSS guarantees the array is redundant |
| bandstop masking | 4th-order IIR notch per band, p=0.4 | winner's MEGAugment |
| time masking | 2 spans, ≤180 samples | winner's MEGAugment |
| mixup | on the pooled embedding, targets mixed in t5 space | clean with an embedding head in a way it never is with a 50-way softmax |
| drop_path | 0 → 0.1, linear with depth | trial 3 disabled it because a 3-epoch run cannot see its benefit; a 30-epoch run can |
| label smoothing | 0.1 (CE head only) | |
| **lr** | **3e-4 peak**, 5% warmup, cosine → 1e-5 | ⚠️ trial 3's own Deferred list already names lr 3e-4 as "the highest-value next run" |
| weight decay | 0.05 | winner's 5e-2 |
| batch | 256, grad-accumulated if needed | in-batch contrastive **needs** a large batch; 64 gives 64 negatives |
| **weight EMA** | decay 0.999, evaluate the EMA | reliably worth ~a point, costs one extra copy of the weights |
| **ensemble** | **5 seeds, averaged probabilities** | measured **+19.5% relative on holdout** |
| **TTA** | average over 3–5 jitter offsets | same idea, one checkpoint |

⚠️ Note the winner found **augmentation removal cost +0.01% — nothing** on the
phoneme task. Augmentation is on this list because trial 2 overfitted from epoch
6, not because it is proven here. If it costs a day, skip it and ensemble
instead.

---

## 10. ⚠️ Compute — this design does not fit on the laptop

Scaling trial 3's measured 3.3 min compute per epoch by parameters × frames ×
samples:

| variant | params | frames | samples/epoch | ≈ compute/epoch | at duty 0.18 | 30 epochs |
|---|---|---|---|---|---|---|
| trial 3 (measured) | 4.3M | 40 | 277k | 3.3 min | ~24 min | — |
| **trial 4 laptop** | 6M | ~60 | 313k | ~8 min | ~45 min | **~22 h** |
| **trial 4 full** | 26M | ~60 | 780k | ~84 min | ~7.8 h | **~10 days** |
| trial 4 full, one RTX 6000 | 26M | ~60 | 780k | ~8 min | n/a | **~4 h** |

Rough, order-of-magnitude, and the RTX 6000 column assumes ~10× an M-series MPS
for this workload — but the conclusion is not sensitive to the factor:

> **The full design is a single 4-hour HPC job and a ten-day laptop job.**

Per `CLAUDE.md` §6 the HPC is not to be touched without an explicit instruction,
and login host, username and RDS path are all `<TO CONFIRM>`. **That is a
decision for Kamil, and it is the main thing this document needs an answer on.**
The staged plan below is built so that the first three stages — which carry most
of the expected gain — run on the laptop regardless.

---

## 11. The staged plan, with gates

Each stage changes **one** thing and is measured against the same session split.
Trial 3's post-mortem is the reason: window, rate and architecture all moved at
once and the result was unattributable.

| stage | change | runs on | gate to continue |
|---|---|---|---|
| **S0** | Run `holdout.counts()`. Verify the `mne` channel permutation. Measure the sentence/word split. | laptop, minutes | — |
| **S1** | **Instance norm + lr 3e-4 + warmup/cosine + EMA**, trial 2's architecture otherwise unchanged | laptop, ~1 night | beats trial 2's 34.7% |
| **S2** | **Semantic head**: D-SigLIP to t5 embeddings, in-vocabulary only, existing cache | laptop, ~1 night | beats S1 |
| **S3** | **On-the-fly loader** → open vocabulary (780k) + phoneme CTC | laptop, ~2 nights | beats S2 |
| **S4** | **Multi-rate Conformer encoder** (laptop variant) | laptop / HPC | beats S3 |
| **S5** | **Sentence context** + context dropout | HPC | beats S4 on the sentence rows specifically |
| **S6** | **SSL pretraining** | HPC | beats S5 |
| **S7** | 5-seed ensemble + jitter TTA + bigram reranker | either | — |

⚠️ **S1 and S2 are where I would put the money.** They are two nights of laptop
time, they change the two things the literature says matter most, and neither
needs the HPC, a new cache, or the external drive mounted for longer than a
training run.

⚠️ **Every gate is against a ~1.2-point noise band** (trial 2's measured
epoch-to-epoch variation) and, per `CLAUDE.md`, per-class recall at n≈34 for the
rarest class has SE ≈8.6 points. A 1-point "win" is not a win. Size the
validation split by `people`, not by session count.

---

## 12. Decisions, and what was rejected

| decision | why | rejected |
|---|---|---|
| Conformer, not plain transformer | +9.0% over SEANet on this dataset; the 2025 phoneme track was won with one | trial 3's pre-norm attention stack, which plateaued at 27.6% |
| Multi-rate (Zipformer-shaped) | better *and* cheaper than flat Conformer (ICLR 2024) | flat 100 Hz throughout — simpler, and what trial 3 did |
| Raw 306 + learned spatial front end | the winner used raw 306; the reference learns its spatial layer | the fitted rank-73 whitener — ⚠️ a real departure from trials 1–3, and it is what forces the cache rebuild |
| Per-window robust instance norm | +200% on holdout vs BatchNorm, measured | `ChannelStandardiser` fitted on the training split; BatchNorm anywhere |
| D-SigLIP to t5 embeddings as primary loss | it is what 73.23% does; +24% over plain CLIP in their ablation | 50-way CE alone, which is trials 1–3 and cannot use 60% of the data |
| Keep 50-way CE as auxiliary | it optimises the scored metric directly; calibrated averaging at inference | contrastive only, which leaves the metric to a cosine ranking nobody tuned |
| Phoneme CTC auxiliary | free labels, dense supervision, proven decodable at 65.8% F1 | no auxiliary — simpler, and gives up the cheapest regulariser available |
| **Session-adversarial, not a session layer** | the holdout is an unseen recording with no session id, so a per-session parameter has nothing to load at test time | copying the reference model's subject layer, which would look correct and do nothing |
| On-the-fly windowing | a 306-channel cache is 77 GB against 80 GB free; and a cache forecloses context, open vocabulary and SSL | rebuilding the window cache at 306 channels |
| lr 3e-4 + warmup + cosine | trial 3's own Deferred list names it as the leading suspect for its flat curve | lr 1e-4 with no schedule, kept in trials 1–3 for comparability |
| AdamW first, ScaledAdam later | one variable at a time; trial 3's lesson | ScaledAdam from the start, despite Zipformer's result |
| Keep `QuadratureBranch` | the repo's one original idea with a physiological argument, and it is cheap | dropping it — it stays ablatable |
| Bigram reranker last | top-10-of-50 is permissive, so it must move the top-10 *set* to earn anything | building it early, before the per-word model is good |

---

## 13. To verify before writing code

1. **`holdout.counts()`** — the sentence/word split. Gates §5.6 entirely.
2. **The channel permutation** between the h5's `(mag, grad, grad)` order and
   `mne`'s `Vectorview-all` layout. Gates §5.3's geometric prior.
3. **`transformers` + `sentencepiece` install and the t5-large download** —
   needs internet, one-time, then an 80 MB lookup table. Gates §6.1.
4. **Phoneme inventory** in the events TSV: the exact symbol set and whether it
   is the 39-phone set the 2025 task used. Gates §6.3.
5. **The external drive** — mounted, and stays mounted for the length of a run.
6. Whether the 50-word list's `it’s` (curly apostrophe) is normalised the same
   way in `vocab.py` and in the t5 lookup table. `vocab.py` currently does
   `.lower()` only.

## 14. Deferred

- ScaledAdam, as its own run after the architecture lands.
- Spherical-harmonic spatial attention instead of 2-D Fourier — the helmet is
  spherical, but this needs **3-D** sensor positions, which the h5 does not
  carry and the 2-D layout cannot supply.
- `RobustScaler` — superseded: §5.2 removes the fitted scaler entirely, so the
  known-broken mean/std placeholder in `ChannelStandardiser` stops mattering
  rather than getting fixed.
- The Moses secondary vocabulary. Trial 4 predicts primary only, as trials 1–3 do.
- `predict.py` for trial 4. Trial 3 still does not have one, so **trial 3 cannot
  produce a submission**; trial 4 must not inherit that.
- The Broad track. Nothing here is cross-subject, and the subject layer this
  design deliberately drops is exactly what Broad would need back.

---

## 15. Sources

- [The 2026 PNPL Competition: Word Classification and Efficient Cross-Subject Generalisation in LibriBrain100](https://arxiv.org/abs/2609.03231) — reference model scores: BAcc@10 73.23%, BAcc@1 25.60%, Moses 83.60%
- [Towards decoding individual words from non-invasive brain recordings](https://www.nature.com/articles/s41467-025-65499-0) (d'Ascoli et al., *Nature Communications* 2025) — [arXiv](https://arxiv.org/abs/2412.17829). The reference model: BrainModule + 16-layer transformer, D-SigLIP to t5-large embeddings
- [MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification](https://arxiv.org/abs/2512.01443) (de Zuazo, Saratxaga, Navas) — 2025 PNPL phoneme winner, 65.83% F1-macro
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100) (Gulati et al. 2020) — the macaron block
- [Zipformer: A faster and better encoder for automatic speech recognition](https://arxiv.org/abs/2310.11230) (ICLR 2024) — multi-rate U-Net, BiasNorm, SwooshR/L, ScaledAdam
- [Decoding speech perception from non-invasive brain recordings](https://www.nature.com/articles/s42256-023-00714-5) (Défossez et al. 2023) — spatial attention layer, subject layer, CLIP loss
- [Interpretable MEG Decoding of Perceived Speech](https://arxiv.org/abs/2608.01481) — spherical harmonics as an improvement on the 2-D Fourier spatial basis
- [MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](https://arxiv.org/abs/2602.02494) — long-context MEG pretraining
- [LibriBrain: Over 50 Hours of Within-Subject MEG](https://arxiv.org/abs/2506.02098) — the dataset paper
- `pnpl/competition/holdout.py` in the installed package — the two holdout sources and the canonical ordering
