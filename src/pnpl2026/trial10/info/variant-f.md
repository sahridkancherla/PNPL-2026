# Variant F — the d'Ascoli reference, reproduced

The competition paper (arXiv 2609.03231, Table 1) reports **73.23% BAcc@10**
for its adaptation of d'Ascoli et al. 2025 (PMC12658044) on LibriBrain100.
Trials 1–8 peaked at 46.6% on our own validation, and E0 scored ~38% on the
leaderboard (sentence mode) and ~28% (`--window-only`). Trial 10 rebuilds the
reference as faithfully as the two papers and brainmagick's code allow, then
measures how much of it survives the holdout's conditions.

## Why 46% and not 73%

Not one gap but several, and the biggest is not the model:

| | the reference (73.23%) | trials 7–8 (46.6%) |
|---|---|---|
| scored on | pnpl's standard **test partition** — Sherlock1 ses-12 run-2, MOCHATIMIT ses-3, TheMoth ses-30 | holdout-matched sequences from trial 1's session split |
| context tokens | **every word** of the sentence | scored words only, 1–9 per sentence |
| window | a full 3 s after every onset | stopped at the sentence's end (median 1.94 s); 9.6% alone with 1 s |
| input | 306 sensors, attention over sensor **positions**, subject layer | 73 fixed whitened components |
| scaling | RobustScaler **per recording**, clamp ±5, baseline first 0.5 s | robust norm per window |
| encoder | 10 conv layers, dilations 1–16 twice, BatchNorm, GLU every 2 | 5 layers, GroupNorm |
| sentence transformer | **16 × 1024 × 16 heads, ~201.5M** | 4 × 256, 3.2M |
| targets | t5-large middle layer for **every word** (~780k windows) | the 50 scored words (~280k) |
| loss, scoring | D-SigLIP; retrieval among the 50 t5 embeddings | SigLIP + closed-set CE blend |

⚠️ **The 73.23% is not a leaderboard number.** The paper gives none for the
reference. On the real holdout the organisers' own Baseline scores 44.4%,
while the top five teams score 72.7–76.5%. A faithful reproduction should beat
E0's 38% there, but matching 73% on the test partition will not by itself give
~70% on the leaderboard: the holdout serves scored words only, with truncated
windows. The holdout protocol below measures exactly that drop.

## What was built

Same subject, same h5 files; everything else is new and self-contained.

| piece | what | source |
|---|---|---|
| `data_processing/build_reference.py` | every subject-0 recording, all three pnpl partitions: 250 → 50 Hz (`resample_poly`), per-recording robust scaler (200 random 3 s segments), fp16, time-major; every word with its sentence id; sensor positions; F3's PCA | d'Ascoli; brainmagick `norm.py` |
| `cli/build_word_embeddings.py` | t5-large, middle layer (12 of 25), each word alone, sub-tokens averaged, **every word** in the cache | d'Ascoli |
| `models/merger.py` | `ChannelMerger`: 270 virtual channels, softmax over sensors, Fourier features of 2D position (2048), spatial dropout radius 0.2; `SubjectLayer` | brainmagick `common.py` |
| `models/encoder.py` | clamp ±5, baseline, merger, 1×1 270→270, subject layer, 10 conv layers (kernel 3, dilations 1,2,4,8,16 ×2, BatchNorm, GELU, skip, GLU every 2), 1×1 → 2h → GELU → 1024, single-head attention pool | brainmagick `clip_conv.yaml` |
| `models/transformer.py` | pre-norm, rotary over word index, **16 × 1024 × 16**, attention dropout 0.1 | d'Ascoli |
| `losses.py`, `lit.py` | D-SigLIP (t = 10, b = −10 learnable) over every word in the batch; retrieval BAcc@10 / @1 among the 50 words, Moses BAcc@10 among the Moses 50 | d'Ascoli |

**204.2M parameters, 98.7% in the sentence transformer** — the paper's are
203.7M and 98.8%. Encoder width 160 rather than brainmagick's 320 is what makes
the two agree; at 320 the encoder alone would be 7.7M.

Training follows the paper: 3 s windows from onset, every word a token, batch
64 sentences, AdamW lr 1e-4 (weight decay 0.01, AdamW's default), 1% warmup
then cosine over 50 epochs, early stopping on **validation-partition**
BAcc@10 with patience 10. No EMA and no augmentation; both exist as switches.

## The two protocols

`cli.evaluate` scores best.ckpt on pnpl's validation and test partitions, twice:

- **reference**: every word a token, full 3 s windows. On the test partition
  this is the paper's setting, so the target is ~73.2% BAcc@10, 25.6% BAcc@1,
  83.6% Moses BAcc@10
- **holdout**: what the competition holdout serves, as trial 7 measured it —
  scored words only, at most 9 per sentence, MEG stopping 1.1–1.7 s after the
  last word, 9.6% alone with 1 s. Reported with the sentence/isolated split

The gap between the two protocols, on the same recordings, is how much of the
reference survives the holdout's conditions.

## The queue

`configs/trial10/runs.txt`, five lines; each run changes one thing back
towards trials 7–8, so the table decomposes the 46 → 73 gap.

| run | change from F0 | answers |
|---|---|---|
| F0 ×2 seeds | — | does the reproduction reach ~73% on the test partition? |
| F1 | sentence transformer 4 × 256 (5.85M total) | how much is capacity? |
| F2 | scored words only as tokens, so 50-word targets | how much is training on, and reading context from, every word? |
| F3 | 73 fixed PCA components instead of 306 sensors + merger | how much is the input? |

## Results

Pending. `cli.collate` writes `runs/trial10_results.md` with the paper's row on
top; paste it here.

## Running it

All on CX3, in order. Steps 1–2 on a **login** node (they need the internet);
everything else is a job.

```bash
# 1. the six recordings the reference is selected and scored on
uv run python -m pnpl2026.cli.prepare --partition validation --subjects deep
uv run python -m pnpl2026.cli.prepare --partition test --subjects deep

# 2. transformers aside from the venv, and t5-large's weights (pbs/trial10_embed.pbs has both commands)

# 3. the cache, ~4 GB, then the all-word t5 table
qsub pbs/trial10_build.pbs
qsub pbs/trial10_embed.pbs          # after the build finishes

# 4. the queue; each run scores itself into runs/<run>/evaluation.json
qsub -J 1-5 pbs/trial10.pbs

# 5. the table, then submissions from a chosen run
uv run python -m pnpl2026.trial10.cli.collate
uv run python -m pnpl2026.trial10.cli.predict --ckpt runs/trial10_f0_s0/best.ckpt --window-only
uv run python -m pnpl2026.trial10.cli.predict --ckpt runs/trial10_f0_s0/best.ckpt
```

`predict` needs only the checkpoint and the holdout files. It resamples the
holdout to 50 Hz (`padtype="line"`, so epoch edges do not ring) and fits a
robust scaler on each holdout source's own MEG, the nearest equivalent of the
per-recording scaler (inputs only, no labels). In `--window-only` mode the
scaler sees the served 1 s windows and nothing else. ⚠️ The default mode reads
holdout MEG outside each word's 1 s window; as in trials 7–8, the rules do not
settle whether that is allowed.

⚠️ `pbs/trial10.pbs` asks for an **RTX6000** and trains in fp16 (`16-mixed`
in `configs/trial10/hpc.yaml`): Turing has no bf16. The loss and retrieval
scores run in float32 regardless. For an L40S, change `gpu_type` there and
`precision` to `bf16-mixed`, together; it should run roughly 2–3× faster.

## What was checked (2026-10-01)

- **shape-check skill**: five prime-dimension configurations (merger, linear,
  PCA, no initial/subject layer, no sentence transformer) and all four real
  configs at real dimensions on GPU in bf16 — batch of one, backward with no
  disconnected parameter, padding invariance, batch-composition invariance
- **units**: positions follow channel names, not order; merger weights sum to
  one over sensors and training bans a disc; the resampler is zero-phase
  (250 Hz sample 5k → 50 Hz sample k) and removes a 40 Hz tone; both scalers
  equal sklearn's `RobustScaler`; deduplicated SigLIP equals plain SigLIP on
  the kept rows; vocabulary columns are the official order, curly `it’s`
  included
- **end to end, synthetic recordings in pnpl's layout**, with a planted
  word-specific response: the builder put the response at exactly 0.3 s;
  F1 learned it to 98.8% test BAcc@10 (alignment holds from labels to
  retrieval); F0, F2 and F3 trained and scored; `--resume` continued from
  last.ckpt; `collate` read finished, running and unstarted runs. F0 in
  fp16 (`16-mixed`, the RTX6000's precision) trained without NaNs and
  learned as fast as in bf16 (validation 0.200, 0.544, 0.577 over 3 epochs)
- **t5 builder** on a tiny random T5 with the real tokenizer: batched, masked
  encoding equals encoding each word alone (max difference 1.7e-6)
- **predict on the real holdout files**, both modes: 960 rows, 101 columns,
  every row filled once. Every in-context sentence window equals the loader's
  canonical window for its row where the onset sits on the 50 Hz grid (495
  rows, correlation 1.0000); the other 373 sit 0.4 samples off-grid, the same
  rounding the training cache applies, and still match their own row best

## Cost

Timed on a laptop RTX 3060: F0 718 tokens/s, F1 1,622 tokens/s, F0 peak
4.6 GiB at 212 tokens. An epoch is ~750k tokens, ~4.7 PFLOP once sentence
padding is counted. Estimate on an L40S: **4.5–7.5 min/epoch, 4–6 h for 50
epochs**; on the RTX6000 the queue uses, roughly 2–3× that, so **F0 may need
a second 12 h submission** if early stopping does not end it first. Resubmit
the same line (`qsub -v RUN=<n> pbs/trial10.pbs`) and `--resume` continues
from last.ckpt with the schedule where it stopped. Each run
leaves last.ckpt (2.4 GB, with optimiser state) and best.ckpt (0.8 GB,
weights only) on RDS.

A batch of 64 sentences pads the transformer's grid to its longest sentence,
roughly 3.5× the real tokens. Bucketing sentences by length would cut F0's
time by about half; it is not done, to keep the batches as the paper's.

## Known risks

- ⚠️ **The first epoch can read exactly 20.00% BAcc@10 and 2.00% BAcc@1.** At
  initialisation the time-pooled word vector is almost all bias -- token-to-
  token variation was 0.4% of it -- so every word ranks the 50 candidates
  identically. Every smoke run sat there for its first epoch and then climbed
  (64-wide targets: 0.200, 0.467, 0.768, ... 0.935). Exactly chance after
  several epochs on real data, by contrast, means something is broken.
- The competition's exact adaptation is undocumented: encoder width, what
  "batch 64" counts, the subject layer. This follows d'Ascoli and brainmagick;
  expect 65–73%, not exactly 73.23%.
- The test partition is three recordings, so BAcc is noisy; hence two F0 seeds.
- `has_positions` depends on the h5 files carrying `channel_names`, which
  pnpl's serialiser writes. If they do not, the PBS preflight stops and says
  to set `spatial: linear`.
