# Trial 11 — one stream per sentence, a Zipformer-style multi-rate encoder

## The idea

Trials 7–9 encode every word's window **separately**, squash each to one
vector, and only then let the word vectors attend to each other. Trial 7's D8
suggests much of the context gain is words reading each other's overlapping
brain responses — which the squash happens before.

Trial 11 keeps the sentence as **one timeline** until the very end:

- the front end and trunk run **once per sentence**, not once per word, so
  each sample is encoded once rather than once per window that covers it
- words meet at **frame level**, at several rates, before anything is pooled
- the rates follow Zipformer (Yao et al. 2023): each stack drops to a lower
  frame rate, works there, and returns to 50 Hz. At 6.25 Hz a token spans
  160 ms — about one per word gap (median 0.21 s in the holdout)

Not the paper's block: each stack holds one **plain pre-norm transformer
block**, trial 8's. The local convolution the paper builds into its blocks
is already done by the trunk at full resolution, and its ablation puts each
block-level trick at only +0.1–0.3. One change at a time: the scaffold first.

## The network

```
 sentence MEG, first scored onset → end      isolated word: its 1 s
 [S, 73, 1200]  padded to 4.8 s, real for L_s samples
   │ robust instance norm, real samples only ────────► per-component scale
   │ (no baseline: the first 0.5 s IS word 1's response)          │
   │ low-pass 20 Hz, ÷5 → 50 Hz                                    │
 [S, 73, 240]                                                      │
   │ noise 0.1 · component dropout 0.1   (training)                │
   │ spatial 73→270 · initial 270→512                              │
   │ dilated conv trunk 1,2,4,8,16 → 256   (each frame: ±0.6 s shape)
 [S, 240, 256] ◄── + amplitude bias ───────────────────────────────┘
   │
   │  S1  50 Hz    block                               240 tokens
   │  S2  ÷2 → block at 25 Hz   → ×2 → bypass          120
   │  S3  ÷4 → block at 12.5 Hz → ×4 → bypass           60
   │  S4  ÷8 → block at 6.25 Hz → ×8 → bypass           30
   │  S5  ÷4 → block at 12.5 Hz → ×4 → bypass           60
   │  S6  ÷2 → block at 25 Hz   → ×2 → bypass          120
   │  final LayerNorm
 [S, 240, 256]  at 50 Hz
   │ per word: attention-pool its own 1 s from onset (50 frames)
 [N, 256]
   ├─ Linear → 50      closed-set, cross-entropy
   └─ Linear → 1024    t5 embedding, plain SigLIP
```

**Downsample ÷k**: a learned weighted average of each k frames (softmax over
k scalars), over real frames only. **Upsample ×k**: each token repeated k
times. **Bypass**: `x + c ⊙ (y − x)`, c a per-channel sigmoid starting at
0.5 — c near 0 skips the stack, so the 50 Hz detail always survives.

**Block**: LayerNorm → attention (8 heads × 32, rotary, padding mask) → add
→ LayerNorm → FFN 256→1024→256 → add. Rotary counts tokens at the stack's
own rate, so each stack learns its own timescale.

## The queue — `configs/trial11/runs.txt`, 2 runs × 3 seeds

| lines | run | `stack_factors` | question |
|---|---|---|---|
| 1–3 | `z1_multirate` | `[1, 2, 4, 8, 4, 2]` | the multi-rate stream |
| 4–6 | `z0_flat` | `[1, 1, 1, 1, 1, 1]` | the control: same blocks and bypasses, all at 50 Hz |

```bash
qsub -J 1-6 pbs/trial11.pbs
PYTHONPATH=src python -m pnpl2026.trial11.cli.collate     # runs/trial11_results.md
```

Two readings:

- **Z1 against Z0** — what the change of rate is worth. Clean: identical
  data, trunk, blocks and parameters up to 20 downsample weights.
- **Z0 against trial 8's E0 (0.466)** — what one shared stream is worth over
  per-word windows plus a word-level transformer. Not perfectly clean: see
  the departures below.

Read `sentence` and `isolated` apart, as in trials 7–8. An isolated word is a
one-word stream of 1 s, so the shared timeline cannot help it.

## Results — leaderboard, 2026-10-02

Seed 0 only, sentence mode (the default `predict`), Kaggle score:

| run | `stack_factors` | leaderboard |
|---|---|---|
| `z1_multirate` | `[1, 2, 4, 8, 4, 2]` | **0.37177** |
| `z0_flat` | `[1, 1, 1, 1, 1, 1]` | 0.33934 |

- **Z1 beats Z0 by 3.2 points**: the change of rate is worth something, on
  otherwise identical data, trunk and blocks.
- **Neither beats trial 8's E0** (~0.38 sentence mode on the leaderboard;
  trial 9's H0, the same network, scored 0.38024). Z1 is 0.8 points below it
  and Z0 4.1 below, so one shared stream per sentence did not beat per-word
  windows plus a word-level transformer here.
- Both are far below trial 10's F1 (0.58023, a different cache and front end).
- One seed each, so no error bar; seeds 1 and 2 are not submitted.

## Parameters

Measured from the built models.

| part | params |
|---|---|
| spatial + initial | 158,732 |
| amplitude projection | 18,944 |
| dilated trunk | 828,064 |
| 6 blocks × 789,760 | 4,738,560 |
| 5 bypasses × 256, final norm, downsample weights | 1,812 (Z1) / 1,792 (Z0) |
| readout pool + heads | 276,274 |
| **total** | **6,022,386 (Z1) · 6,022,366 (Z0)** |

## Where it departs from trial 8's E0

| | E0 | trial 11 | why |
|---|---|---|---|
| what is encoded | each word's window, up to 3 s | each sentence's stream, up to 4.8 s | the point of the trial |
| where words meet | word vectors, 4-block transformer | frames, at every rate | the point of the trial |
| baseline | 0–0.5 s per window | none | a stream's first 0.5 s is word 1's response; trial 6's C3 found no baseline +1.2 anyway |
| sentence splitting | ≤ 9 words | ≤ 9 words **and** onsets within 3 s | every sequence must fit one fixed-length stream; holdout sentences span < 3 s, so only sequences the holdout never serves are cut |
| jitter | per word | one shift per stream | words share a timeline |
| padding | to 3 s per window | to 4.8 s per stream | fixed, so GroupNorm statistics never depend on batch-mates |
| word masking | off | n/a | words are not tokens until the readout |

Everything else — cache, trunk, heads, losses, plain SigLIP, schedule,
noise, component dropout — is E0's.

## Checked, and how — 2026-10-02

| check | result |
|---|---|
| `Downsample` by hand, frames 0–7 with the last two padding, ÷4 | `[1.5, 4.5]`: the padding gets no weight |
| prime-dimension harness (S 3, N 7, C 5, d 24, 3 heads, 11 classes, embed 13), Z1 and Z0 | every output its expected shape |
| garbage past each stream's end | output unchanged (0.0) |
| a stream alone against the same stream in a batch | 1.8e-7 (Z1), 2.4e-7 (Z0) — float rounding |
| perturb stream 0 | only stream 0's words move |
| backward, both real configs | every parameter has a gradient (a ÷1 downsample used to hold an unused weight; removed) |
| batch of one isolated word | `[1, 50]`, finite |
| config guards | a span + tail + jitter longer than the stream, mismatched factor and depth lists, and word masking all raise at load |
| **`cli.train` end to end**, synthetic cache in the real layout (6 recordings, 73 components, a class-specific response planted 0.3 s after onset) | Z1 val BAcc@10 0.205 → 0.215 → 0.249 → 0.277 over 4 epochs; `--resume` continued from `last.ckpt` to 0.311; Z0 trains |
| **`cli.predict` on the real holdout files** (synthetic transforms) | 960 rows in canonical order, all sum to 1, both modes; isolated rows identical across modes (0.0); sentence rows differ, as context must make them |
| holdout stream lengths | 156 sentences, 1.34–4.12 s — all inside the 4.8 s padding |
| `cli.collate` on an empty runs folder, `bash -n pbs/trial11.pbs` | both pass |

⚠️ Not checked: anything on real MEG, and the walltime. `pbs/trial11.pbs`
asks for 3 h on an RTX6000 — an estimate: trial 8 early-stopped inside 2 h,
and trial 11 encodes each sample once but attends over 240 frames. A kill is
resumed by resubmitting the same line.

## Open questions

- If Z1 ≈ Z0, the rate change is not what matters; if both beat E0, the shared
  stream is. Both outcomes are informative.
- If trial 9's high-pass helps E0, trial 11's long streams are where a 0.5 Hz
  filter works best — the natural next combination.
- If trial 10's F3 shows 306 sensors beat our 73 components by a wide margin,
  trial 11 should move onto trial 10's front end before anything else.
