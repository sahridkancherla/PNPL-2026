# Trial 23 — the final nights (14-16 Oct): anneal, new heads, tower blend, SAM, F0 fixed

The note: `handoff-docs/final-night-note-2026-10-14.md` ("Which optimiser,
model-size and architecture changes are worth running"). Built 14 Oct.

## Why

Best file `ens_t23_robust_ts_hubh_lone`, **70.849**: four transduced mixes of
the 23 members (blends 0.65 / 0.75 / 0.9, and 0.75 with SPW seeds 3-5)
averaged, then the holdout hub correction and I3's lone rows. Every idea of
the 13 Oct note failed its gate (`research-note-13oct.md`); the only thing
that has gained on the board is averaging over settings and seeds, and set B
(the same 23 families at fresh seeds) scored 69.00, so part of 70.85 is seed
luck.

The note says: no L-BFGS, no smaller main model, no new backbone or decoder.
Yes to (1) a short low-LR anneal on holdout-shaped fragments with weight
averaging, (2) output heads on another language model's middle layer,
(3) a training-free tower/anchor blend, (4) SAM, as a diversity member only.

User decisions (14 Oct): anneal + average on G2 s0-s2 and w1024 s0-s1;
head-only training on frozen G2 s0-s2, against GPT-2-medium layer 12 and
Qwen2.5-1.5B layer 14 (ungated, in place of the note's gated Llama-3.2-1B);
SAM as extra anneal members; one F0 re-run with the fixed recipe; no full
K-style retrains. Code goes into trial 23, not a new trial: every new member
must flow through trial 23's transduction, robust and lone tools. Deadline
15 Oct 23:59 AoE = Fri 16 Oct 12:59 BST: two nights.

## What went wrong with the 204M model (F0)

Read from `runs/trial10_f0_s0|s1/metrics.csv` and the trial 10/12 records.

| epoch | F0 val | F0 train | F0 train loss | F1 val | F1 train loss |
|---|---|---|---|---|---|
| 11 (F0's best) | **69.9** | 76.3 | 5.00 | 67.1 | 5.84 |
| 14 | 68.2 | 81.6 | 3.71 | 67.4 | 5.68 |
| 17 | 65.5 | 87.0 | 2.25 | 68.2 | 5.52 |
| 21 (F0 stopped) | 63.4 | **92.6** | **1.15** | 68.9 | 5.31 |

1. **It memorises.** After its validation peak the training loss falls
   5.0 -> 1.1 in ten epochs and training accuracy reaches 93%; validation
   drops 6.5 points. Seed 1 has the same shape. w1024 (53M) does it more
   slowly (loss 5.1 -> 2.4, val 70.0 -> 64.3); F1 (5.85M) never does.
   204M parameters were trained with the 723-participant recipe: lr 1e-4,
   feed-forward dropout 0, drop_path 0, no EMA, no augmentation.
2. **The kept checkpoint was never annealed.** Epoch 11 of a 50-epoch cosine
   is 0.89x the peak rate; early stopping fired at 21, so the decay never
   reached the weights that were kept.
3. **The organisers' recipe differs in one documented number:** the same
   16 x 1024 at **lr 1e-5** (patience 5) scored 73.23 on test; ours at 1e-4
   scored 70.4 / 68.9.
4. **The board's -8 is more than the recipe:** F0 never trained on
   holdout-shaped fragments (trial 13's +5 came later), and one seed carries
   about +-5 on this board. Capacity itself is not shown to hurt: w1024 gave
   +3.3 to the mix.

## What was built

| recipe | tool | members |
|---|---|---|
| 1, anneal + average | `cli/anneal.py --mode anneal` | `trial23_an_*`, fallbacks `trial23_anlast_*` |
| 4, SAM anneal | `cli/anneal.py --mode anneal --sam` | `trial23_sam_*`, `trial23_samlast_*` |
| 2, new heads | `cli/build_lm_embeddings.py`, then `cli/anneal.py --mode head` | `trial23_hg_*` (GPT-2), `trial23_hq_*` (Qwen), `*last_*` |
| 3, tower blend | `--tower-weight` on `cli/predict.py` and `cli/evaluate.py` | `trial19_spw_s0..2_tw07`, `trial20_spw_s3..5_tw07` |
| F0 fixed | trial 18's trainer, `configs/trial18/f0_fixed.yaml` | `trial18_f0fixed_s0` |

**The anneal** (`perword/config.py` and `perword/lit.py` knobs, all off by
default so every earlier checkpoint behaves as before): the parent's
(EMA) weights; AdamW at 0.1x the parent's rate, no warmup, decaying as
1 - sqrt(progress) to 0 (`schedule: one_minus_sqrt`); 4 epochs, ~20% of the
parents' 20-24; fragments filling half of every epoch beside the parent's
own isolated share (G2: 40% + 10%; w1024: 50%); BatchNorm frozen
(`freeze_batchnorm`), dropout on; one weights-only checkpoint per epoch, the
last 3 averaged into `best.ckpt` (buffers the last epoch's, which are the
parent's); the last epoch alone kept as the fallback member, "last" joined
to the run name's second word.

**SAM** (`perword/sam.py`): Foret's closure form around AdamW, rho 0.05, one
global gradient norm, `w` restored exactly before the base step. Lightning
hands `optimizer.step` a closure only without a GradScaler, so SAM runs at
precision "32" (`config.py` refuses 16-mixed). SAM evaluates each batch
twice; the second pass is neither logged nor stored.

**Heads**: `cli/build_lm_embeddings.py` encodes every word of the t5 table,
in its order (the two heads index both tables with one target row), each
word alone with a leading space for byte-level BPE tokenisers, right
padding, the mean over its real tokens, float32. `--mode head` adds Variant
K's second head to a frozen parent (`freeze_backbone`: the network runs as at
inference); only the new Linear and its SigLIP scale and bias train, lr 1e-3,
cosine to 0 over 5 epochs. `models/net.py`'s `second_embed_dim` lets the head
be 1536 wide for Qwen. The member ranks by both heads combined, as K2 does.

**Tower blend**: `Trial23Config.tower_weight` (0.5, the trained average, by
default) is the tower head's share of the combined score; another value is
another member, named `<run>_tw07` for 0.7.

**Scoring**: trial 23 had no per-word evaluator (`cli/evaluate.py` loaded the
continuous family only). `perword/evaluation.py` is trial 18's scorer on
trial 23's per-word code; `cli/evaluate.py` hands per-word checkpoints to it.
`cli/anneal.py` scores the parent on fragments, and both members on the
reference and fragment protocols with dumps; `anneal.json` holds the
settings and the sanity verdict (the note's: within 1 BAcc@10 point of the
parent on fragments, the mean of validation and test, or dropped).

**F0 fixed**: G2's recipe at 16 x 1024 x 16 (204.22M parameters), lr 1e-5,
feed-forward dropout 0.1, EMA 0.999, fragments 0.4 beside isolated 0.1,
patience 8 over a 50-epoch cosine. Line 34 of `configs/trial18/runs.txt`.

**Jobs and lists**: `pbs/trial23_anneal.pbs` (one line of
`configs/trial23/anneal.txt`, then both members' holdout CSVs),
`pbs/trial23_lm_tables.pbs` (both tables, then trial 16's hub check),
`pbs/trial23_tower.pbs` (the six SPW seeds at WEIGHT); every new member is a
line of `configs/trial23/newmembers.txt`, which the robust step's `INCLUDE`
regex reads (`pbs/trial23_cpu.pbs`; `MEMBERS_FILE` starts from set B).

## Checks (laptop, synthetic, short)

- Second head 19 wide beside 17 at batch sizes 1 and 25 tokens; every head
  parameter has a gradient; `load_trained` rebuilds the width. Old
  checkpoints load strictly and predict exactly as trial 22's code.
- Per-word evaluation: reports and dumps identical to trial 18's
  `cli/evaluate.py`; continuous evaluation identical to trial 22's.
- Tower weight 0.5 equals the plain prediction row for row; 0.7 differs.
- Schedule: 1 - sqrt from the peak with no warmup; the default cosine as
  before. SAM's step follows the gradient at w + rho g/|g| from w.
- `cli/anneal.py` at lr 0: all 64 tensors bit-identical to the parent and
  the holdout CSV equal. A real anneal moves every parameter and no
  BatchNorm buffer; `best.ckpt` is exactly the mean of the last epochs.
  Head mode leaves every parent tensor bit-identical.
- ⚠️ **Found by the lr-0 check: Lightning 2.6 never calls `model.train()`;**
  it keeps each module's mode around validation. Without an explicit
  `train()` before fit, the "frozen" BatchNorm statistics moved. Fixed in
  both `cli/anneal.py` and `on_fit_start`.
- The LM table equals each word encoded alone (padding changes nothing);
  `TargetTable` reads it with the t5 table's words; the hub check runs.
- The robust step's INCLUDE, MEMBERS_FILE and refusals in a dry run;
  `bash -n` on every job script; F0-fixed builds at 204.22M.

## Order of operations

Night 1 (14 -> 15 Oct): the Qwen download on a login node; then
`pbs/trial23_lm_tables.pbs`, anneal lines 1-5, the tower job, the F0 fix;
heads (lines 11-16) once the tables exist; SAM (lines 6-10) last.

Day 15: `cli/final_status.py` -- every new member's files, fragment scores
and verdict (a member more than 1 point below its parent is left out, with
the EXCLUDE it prints) -- then the robust rebuilds with INCLUDE as CPU jobs,
submitted in this order: anneal, heads, tower, all that passed, sam, f0fixed,
then the best rebuilt on set B too and averaged with A, a leave-one-group-out
of the best, its P7 lone version, a spare. A file is kept only if it beats
70.849.

Night 2 (15 -> 16 Oct): more of whatever won, with new lines in
`anneal.txt` and no new code -- the other families' parents and set B if the
anneal won; the winning table on more frozen backbones (and the full K2-style
retrain) if a head won; F0-fixed seeds 1-2 if it won; more SAM if SAM won.
If an anneal fails its sanity check, the note's second choice is to
recompute its BatchNorm statistics on training windows (not built).

## Results

Filled in as they arrive: the sanity verdicts, then the board.

Board, 15 Oct (robust + the group, against robust's 70.849). None beat it.

| file (`ens_t23_<name>_robust_ts_hubh_lone`) | members added | board |
|---|---|---|
| `f0fixed` | F0 fixed | 70.849 (tie, to the digit) |
| `tower` | SPW s0-5 at tower weight 0.7 | 70.599 |
| `towerswap` | SPW s0-2 replaced by their 0.7 versions | 69.849 |
| `heads` | 6 GPT-2 / Qwen heads on G2 | 70.313 |
| `all` | every group above and below | 69.599 |
| `noN` (13 Oct, for reference) | robust without N | 69.199 |
| `anneal` | 5 annealed G2 / w1024 | 69.163 |
| `sam` | 5 SAM anneals | 69.163 (same as anneal, to the digit) |

The anneal, a recipe the note ranked highest, cost 1.7 points as an
addition. The two exact ties (f0fixed = robust, sam = anneal) must be
checked for identical files before anything is read into them.

## The second night (15 -> 16 Oct): research and the portfolio

Nothing from the first night beat 70.849 (above). The user asked for anything
that could still raise the public score, with self-supervision on the holdout,
a Conformer, MEGState and a spherical-harmonics / temporal-filter front-end
named. What the research found (15 Oct):

- The 2025 phoneme track's winner was one compact Conformer (MEGConformer,
  arXiv 2512.01443) whose decisive trick, instance normalisation, lost 6-7
  points here (trial 15): 2025 scored 100-trial averages, this track single
  trials. MEGState's multi-resolution stem (arXiv 2512.17978) is already
  SPW's. The spherical-harmonics decoder (arXiv 2608.01481) replaces the
  spatial attention by a fixed SH basis -- which SSS has already imposed on
  this data (rank ~73-96; F3's fixed basis lost 7) -- and adds 25 branches
  with trainable temporal filters, the part built here.
- Test-time adaptation in the literature (entropy minimisation on norm layers,
  alignment, SSL-then-TTT) is EEG/sEEG; here AdaBN, re-centring and
  pseudo-label distillation each lost 3-5. Pretrained MEG foundation models
  reach 63 on this task (supervised ~73); MEG-XL's features failed P4.
- Subject-0-only all-data was never on the board (the 54.8 had 12 other
  subjects). Transduction's pull 2, neighbours and share floor were never
  varied.

Built and queued (`pbs/night2.sh`, one dependency chain per family: train ->
predict -> robust rebuild with `SANITY=1` (more than a point below G2 on
fragments is left out) and `SKIP_MISSING=1`):

| stage | members | new code |
|---|---|---|
| today | transduction pull 2 (local check), 6 neighbours, both; robust8 = robust + pull2 mixes averaged | robust step `PULL`, `NEIGHBOURS`, `SHARE_FLOOR`, `CHECK`, `MERGE` |
| configs | K-GPT2, K-Qwen (joint second heads), W-fixed (4 x 1024, lr 3e-5, dropout 0.1, EMA), All-0 G2 + K2, F0-fixed s1-2 | trial 18 `second_embed_dim` |
| ssl | G2 fine-tuned from masked-span reconstruction of every subject-0 recording + 40 subjects' holdout epochs | `cli/pretrain_encoder.py`, `init_encoder_from` |
| tf | merger to 25 branches, a 0.5 s FIR filter per branch | `WordEncoder.temporal_filter_kernel` |
| conformer | 4 Conformer blocks in place of the dilated convs | `models/conformer.py`, `WordEncoder.temporal` |
| tent | BatchNorm affine of G2/K2/K18 adapted by entropy minimisation on the holdout sentences, in their parents' place | `cli/tent.py` |
| all | every family above that passed, together | -- |

Checks (laptop, synthetic): the robust switches in a stubbed dry run; the
chains with a fake qsub; trial 18's second head 19 wide; the five configs;
the sanity filter's threshold; SSL pretraining runs and learns on real
holdout MEG (unexplained hidden variance 107% -> 91% in 200 tiny-model
steps), the fine-tune loads 41 encoder tensors; the Conformer and the FIR
encoder pass the shape check (distinct primes, padding junk changes nothing,
rows alone = rows batched within 5e-4, gradients everywhere; the FIR at its
identity start equals the plain encoder exactly) and train one epoch through
trial 18's trainer and predict through trial 23's; Tent moves only BatchNorm's
affine tensors and lowers the entropy.

Board, 15 Oct evening (transduction settings, against 70.849):

| file | setting | local check (transduced minus plain) | board |
|---|---|---|---|
| `pull2nb6` (`_ts_hubh`, no lone fill) | pull 2, 6 neighbours | not run | 70.842 (tie) |
| `nb6` | pull 1, 6 neighbours | not run | 70.147 |
| `robust8` | robust's four mixes + pull 2's four, averaged | pull 2's: -2.44, wins 5% | 69.747 |

Pull 2 alone failed its local check by 2.4 against pull 1's ~0, yet pull 2
with 6 neighbours ties the best on the board: the local check again does
not rank settings. Six neighbours alone lost 0.7.

Board, 16 Oct morning (against 70.849):

| file | what | board |
|---|---|---|
| `robustpn` | robust's 4 mixes + pull2nb6's 4, averaged | **71.432, new best** |
| `sslg2` | + SSL-G2 (3) | 70.849 (tie) |
| `wfixed` | + W-fixed (3) | 69.849 |
| `conf` | + Conformer (3) | 69.849 |
| `all0` | + All-0 G2/K2 (4 passed) | 69.599 |
| `f0fam` | + F0-fixed s0-2 | 69.449 |
| `kgq` | + K-GPT2/K-Qwen (3 passed) | 69.313 |
| `all` | every family together (43 members) | 67.849 |

No trained family helped; SSL-G2 tied, like F0-fixed. The only gain of the
two nights is, again, averaging transduction settings that each tie the best.
| `sslpn` | sslg2's 4 mixes + pull2nb6's 4, averaged (robustpn with SSL in the pull-1 half) | 71.032 (-0.40 against robustpn) |
| `robustpn3` | robust + pull2nb6 + pull 1.5 / 5 neighbours, 12 mixes averaged | 71.032 (-0.40 against robustpn) |

⚠️ The second night's `all` rebuild overwrote the first night's
`ens_t23_all_robust_ts_hubh_lone.csv` (69.599 on 15 Oct); the file on disk
is the 16 Oct one (67.849).

## Final picks (16 Oct, by 12:59 BST)

| # | file | public |
|---|---|---|
| 1 | `ens_t23_robustpn_robust_ts_hubh_lone` | 71.432 |
| 2 | `ens_t23_sslpn_robust_ts_hubh_lone` | 71.032 |
| 3 | `ens_t23_robustpn3_robust_ts_hubh_lone` | 71.032 |
| 4 | `ens_t23_robust_ts_hubh_lone` | 70.849 |
| 5 | `ens_t23_robustb_ts_hubh_lone` | 70.849 |
| 6 | `ens_t23_f0fixed_robust_ts_hubh_lone` | 70.849 |
| 7 | `ens_t23_sslg2_robust_ts_hubh_lone` | 70.849 |
| 8 | `ens_t23_pull2nb6_robust_ts_hubh` (no lone fill) | 70.842 |
| 9 | `ens_t23_blendavg_ts_hubh_lone` | 70.682 |
| 10 | `ens_t22_tc1b75_ts_hubh_lone` | 70.682 |
