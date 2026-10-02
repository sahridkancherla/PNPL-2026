# Trial 9 — a 0.5 Hz high-pass on every window

## The idea

The slowest part of the MEG signal is dominated by things that are not the
word: head and body movement, and slow sensor drift. A high-pass at 0.5 Hz
removes that band and keeps the word response, which sits mostly at 1–8 Hz.
Combined with the existing 20 Hz low-pass, every window becomes **0.5–20 Hz**
instead of **0.1–20 Hz** (the recordings already carry a 0.1 Hz high-pass
from preprocessing).

Everything else is trial 8's E0: trial 7's D1 with plain SigLIP and no word
masking, which scored **0.466 ± 0.001**.

## Why on E0, and not on trial 10 or 11

E0 is the best measured network (0.466 validation; ~38% sentence mode and
~28% `--window-only` on the leaderboard), and with `highpass_hz: null` this
trial reproduces it bit for bit, so the filter is the only change. Trials 10
and 11 have no results yet; putting the filter on either would mix two
unknowns.

If H1 helps, trial 11 is its natural next home: its streams run up to 4.8 s,
where 0.5 Hz is a genuine filter, whereas most E0 windows are 1–3 s.

## Where it sits

```
 [N, 73, 750]  window, real for L_i samples (250–750), zero after
   │
   │  ┌─ WindowHighpass ─────────────── NEW, fixed, 0 parameters ────────────┐
   │  │                                                                      │
   │  │   real samples        odd reflection, 1 s each side                  │
   │  │      ╱╲    ╱             ╲╱  ╱╲    ╱  ╲╱                             │
   │  │  ───╱──╲──╱──   →    ────────╱──╲──╱────────   (a trend continues    │
   │  │                       ◄pad►  real  ◄pad►         straight through)   │
   │  │                                                                      │
   │  │   FFT  ×  |H(f)| = 1 / (1 + (0.5 / f)^4)  →  iFFT     zero phase     │
   │  │                                                                      │
   │  │   gain   0.1 Hz 0.00 · 0.25 Hz 0.03 · 0.5 Hz ~0.5 · 1 Hz 0.94 · 2 Hz 1.00
   │  │                                                                      │
   │  │   crop back to the window, re-zero past L_i                          │
   │  └──────────────────────────────────────────────────────────────────────┘
   │
   │  RobustInstanceNorm      ← now sees no drift in its median and MAD
   │  BaselineCorrect 0–0.5 s
   │  FixedLowpassDecimate 20 Hz, ÷5          → the band is now 0.5–20 Hz
   │  … trial 7's trunk, pool, sentence transformer and heads, unchanged
```

`|H|` is the gain of a 2nd-order Butterworth run forwards and backwards, the
standard zero-phase high-pass (`scipy.signal.sosfiltfilt`). Zero phase means
no latency in the response moves.

## Why it has to be built this way

A clean 0.5 Hz filter needs several seconds of signal. The holdout serves 1 s
isolated words and 1.7–4 s sentences, so filtering the continuous recording
for training would hand the model a transform it never sees at test time.
`WindowHighpass` filters every window on its own, inside the model, so
training, validation, `predict` and `predict --window-only` all get exactly
the same operation.

⚠️ On a 1 s window, 0.5 Hz is below the window's own frequency resolution.
There the filter acts mostly as **removal of the offset and the linear
trend**; on 3 s windows it is a genuine 0.5 Hz filter. The holdout has the
same mix of window lengths, so this does not create a mismatch, but it does
mean the isolated-word score tests a weaker version of the idea.

## The queue — `configs/trial9/runs.txt`

| lines | run | `highpass_hz` | question |
|---|---|---|---|
| 1–3 | `h1_highpass_05` | 0.5 | does removing < 0.5 Hz help? |
| 4–6 | `h0_control` | null | the control |

`h0_control` is **bit-identical to trial 8's E0** at the same initialisation
(checked below), so `qsub -J 1-3` alone can be read against E0's 0.466. Lines
4–6 only guard against drift between the two runs' environments.

**Adopt H1 if it beats H0 by more than the seed spread** (~0.002 in trial 8).
Read `isolated` too: unlike trial 8's reach, the filter acts on one-word
windows, so this is the first change since trial 7 that can move the
rules-safe `--window-only` score.

**What would argue against it:** trial 6's C3 found that *removing* baseline
correction scored +1.2, i.e. stripping slow offset content hurt there. If H1
loses, that is the likely reason: the slowest band carries word information,
not just movement.

## Parameters

**4,442,334**, unchanged: the filter's gain is a fixed, non-persistent buffer.

## Checked, and how — 2026-10-01

| check | result |
|---|---|
| against `sosfiltfilt(butter(2, 0.5 Hz), padtype="odd")` on the same windows, 200 windows of a synthetic recording (drift + 0.2, 3, 8 Hz + noise) | RMS difference 0.07–0.13 at 250–750 samples, on a signal of amplitude ~1 |
| against filtering the WHOLE recording, then cutting | RMS 0.28–0.53, versus 0.30–0.55 for scipy on the same windows and 2.5–2.7 for no filter |
| 3 + 8 Hz content kept | correlation with the clean components 0.84–0.91 |
| padding past `L_i` | exactly zero |
| a batch of mixed lengths vs each window alone | identical |
| the first draft's right-hand mirror index | **was unclamped** and indexed past the array on long windows; clamped, and the check above is after the fix |
| `h0_control` vs trial 8 E0, same seed | logits identical (difference 0.0) |
| `h1_highpass_05` vs E0, same seed | logits differ (max 1.06) |
| injected slow drift, mean absolute per-component offset | 1.133 before, 0.180 after |
| garbage past each window's end, both runs | output unchanged (0.0) |
| backward, H1 | every parameter but the unused mask token has a gradient |
| batch of one isolated word | `[1, 50]`, finite |
| `LitTrialNine._step` on fake data, real t5 table | loss 14.06, finite |
| cost | 0.29 s per 330-word batch on CPU; two FFTs on the GPU |
| `cli.collate` on an empty runs folder, `bash -n pbs/trial9.pbs` | both pass |

⚠️ Not checked: `cli.train` end to end, which needs `data/trial7_continuous/`
on RDS — the same gap as trial 8, where all twelve runs then trained cleanly.
