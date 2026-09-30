# Trial 8 — the architecture, drawn

Trial 7's D1 network with one change: every block of the sentence transformer
has its own **reach**, the number of words either side a word may read. The
queue, parameters and checks are in `reach.md`; this file is the picture.

## The whole network

```
═══════════════════════════════════════════════════════════════════════════════
 INPUT — one batch ≈ 64 sentences ≈ 330 words
═══════════════════════════════════════════════════════════════════════════════
 continuous MEG, 73 whitened components, 250 Hz
 ──────┬──────────┬─────┬──────────────────────────────┬──────────►
       w1         w2    w3                         sentence end
       ├─────────────────────────────────────────────────┤  w1: onset → min(+3 s, end)
                  ├──────────────────────────────────────┤  w2
                        ├────────────────────────────────┤  w3
 each word → [73, 750], zero-padded past its real length L_i (250–750)
 isolated words (9.6%): a one-word "sentence" with a 1 s window

═══════════════════════════════════════════════════════════════════════════════
 STAGE 1 · PER-WORD ENCODER — 1,006,508 params, shared by every word
═══════════════════════════════════════════════════════════════════════════════
 [N, 73, 750]                                                   N words, flat
   │ zero past L_i
   │ RobustInstanceNorm, real samples only ─────────► per-channel scale [N, 73]
   │ BaselineCorrect 0–0.5 s, re-zero padding                              │
   │ FixedLowpassDecimate: 81-tap FIR, 20 Hz, ÷5 → 50 Hz                    │
 [N, 73, 150]                                                               │
   │ GaussianNoise 0.1 · ComponentDropout 0.1 · jitter ±30 ms  (train only) │
   │ spatial   Conv1d 73 → 270 (1×1) + GELU                                 │
   │ initial   Conv1d 270 → 512 (1×1) + GELU                                │
 [N, 512, 150]                                                              │
   │ dilated conv trunk, kernel 3, hidden 160                               │
   │   L1  512→160  dilation 1    reach  3 frames                           │
   │   L2  160→160  dilation 2    reach  7          + GLU                   │
   │   L3  160→160  dilation 4    reach 15                                  │
   │   L4  160→160  dilation 8    reach 31          + GLU                   │
   │   L5  160→256  dilation 16   reach 63 frames ≈ 1.26 s                  │
 [N, 256, 150] → transpose → [N, 150, 256]                                  │
   │ frame transformer, depth 0: amplitude bias ◄──── scale projection ─────┘
   │                             + LayerNorm
   │ AttentionPool over real frames
 [N, 256]   ◄── ONE VECTOR PER WORD

═══════════════════════════════════════════════════════════════════════════════
 STAGE 2 · SENTENCE TRANSFORMER — 3,159,552 params — THE ONLY PART THAT CHANGES
═══════════════════════════════════════════════════════════════════════════════
   │ scatter onto a grid by (sentence, position)
 [S, W ≤ 9, 256]  + padding mask [S, W]
   │
   │  ┌─ block ℓ = 1…4 — 789,760 params each ────────────────────────────────┐
   │  │                                                                       │
   │  │  x ─┬─ LayerNorm ─ MHSA(8 heads × 32, rotary) ─ DropPath ─┐           │
   │  │     │                    ▲                                 │           │
   │  │     │              mask_ℓ = reach_ℓ band                    │           │
   │  │     │                     ∧ key is not padding              │           │
   │  │     │                     ∨ diagonal                        │           │
   │  │     └──────────────────────────────────────────────────── (+)          │
   │  │  x ─┬─ LayerNorm ─ Linear 256→1024 ─ GELU ─ Linear 1024→256 ─┐        │
   │  │     └─────────────────────────────────────────────────────── (+)       │
   │  └───────────────────────────────────────────────────────────────────────┘
   │ final LayerNorm
   │ gather back to flat words
 [N, 256]

═══════════════════════════════════════════════════════════════════════════════
 HEADS — per word — 276,018 params
═══════════════════════════════════════════════════════════════════════════════
   ├─ dropout 0.1 ─ Linear 256 → 50     closed-set logits   cross-entropy
   └─ dropout 0.1 ─ Linear 256 → 1024   t5 embedding        plain SigLIP
   inference: 50/50 blend → top 10 → BAcc@10

 TOTAL 4,442,334 — the same for every run (plus a 256-param mask token,
 unused at word_mask_p 0, kept so the count matches trial 7 D1 exactly)
```

## What a block's mask allows

`reach_ℓ` is an int (words either side) or `null` (the whole sentence). The
band is ANDed with the padding mask, and the diagonal is always restored so a
padding row never has zero visible keys, which would turn it NaN.

```
 reach null (full)              reach 1 (neighbours)
     w1 w2 w3 w4 w5                 w1 w2 w3 w4 w5
 w1   ●  ●  ●  ●  ●             w1   ●  ●  ·  ·  ·
 w2   ●  ●  ●  ●  ●             w2   ●  ●  ●  ·  ·
 w3   ●  ●  ●  ●  ●             w3   ·  ●  ●  ●  ·      w3 reads w2, w3, w4
 w4   ●  ●  ●  ●  ●             w4   ·  ·  ●  ●  ●
 w5   ●  ●  ●  ●  ●             w5   ·  ·  ·  ●  ●
```

## How reach compounds through the blocks

One tensor flows through; each block adds an update to every word's vector.
Under reach 1 a word reads its neighbours, but from block 2 on those
neighbours already hold THEIR neighbours, so information hops one word per
block. Following `w3` of a 5-word sentence:

```
            E1: [1, 1, null, null]                what w3's vector holds after it
 start      w3                                    its own MEG only
 block 1    w2 ← w3 → w4             reach 1      w3 + w2, w4
 block 2    w2 ← w3 → w4             reach 1      w3 + w2, w4 + (w1, w5 via w2, w4)
 block 3    w1 w2 w3 w4 w5           full         everything, read directly —
                                                  and every word is by now a
                                                  summary of its neighbourhood
 block 4    w1 w2 w3 w4 w5           full         a second global pass

            E2: [null, null, 1, 1]
 block 1–2  everything, raw          full         the whole sentence first
 block 3–4  neighbours only          reach 1      local refinement on top

            E3: [1, 1, 1, 1]
 after 4    at most ±4 words                      in a 9-word sentence, w1 and
                                                  w9 never meet
```

Measured, not just argued: perturbing `w1` of a 9-word sentence moves all nine
predictions under E0–E2 and only `w1`–`w5` under E3 (`reach.md`, checks).

## The code

| piece | file |
|---|---|
| the band mask | `attention_mask(valid, causal, n_times, device, reach)` in `../models/transformer.py` |
| one mask per block, built once per distinct reach | `FlatTransformer.forward` |
| validating the list | `resolve_reach` — also called by `Trial8Config`, so the PBS preflight catches a bad list |
| the config knob | `model_kwargs.sentence_reach` in `configs/trial8/*.yaml` |
