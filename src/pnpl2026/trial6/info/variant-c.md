# Variant C — the ablation queue, as implemented

Trial 6 is the "Variant C: changes from Variant B" design, turned into eleven
short runs against one reference. Every run is trial 5's network with one
change, capped at 20 epochs, selected on `val/bacc10`.

## The queue

`configs/trial6/runs.txt`, in array order. Each file carries only its change.

| # | run | change from C0 | question | adopt if |
|---|---|---|---|---|
| 1 | `c0_reference` | none: trial 5, 20 epochs | the reference | — |
| 2 | `c0_shuffled` | training labels shuffled | where is chance, measured? | — (the null) |
| 3 | `c1_depth2` | transformer depth 8 → 2 (3.29M) | is the transformer earning its 6.3M? | ≥ C0 |
| 4 | `c2_no_transformer` | depth 0 (1.71M) | the same, at the limit | within 0.5 pt of C0 |
| 5 | `c3_no_baseline` | `with_baseline: false` | is baselining removing signal? | > C0 |
| 6 | `c4_envelope_20_40` | stream B 20–40 Hz **alone** | does beta/low gamma carry words? | clearly > run 2 |
| 7 | `c5_envelope_40_100` | stream B 40–100 Hz **alone** | does the flat floor carry any? | clearly > run 2 |
| 8 | `c7_siglip_fixed` | SigLIP sums over candidates | does the semantic head help with gradient? | > C0 |
| 9 | `r1_regularise` | jitter ±30 ms, noise σ 0.1 | does the design's training row help? | > C0 |
| 10 | `w05_half_window` | window [0, 0.5] s | is anything after 0.5 s used? | ties C0 → no |
| 11 | `c6_lowpass_100` | 100 Hz lowpass, no decimation | the planned change, directly | > C0 by enough for 5x frames |

Small differences are noise. Per-class recall on the rarest validation class
has an SE of ~7 points (48 examples), and every run prints its own figure on a
`[data] rarest validation class` line at the start.

## Results — 2026-09-29

Best `val/bacc10` per run, one seed each. Δ is against C0, in points.

| run | BAcc@10 | Δ | rule | verdict |
|---|---|---|---|---|
| C0 | 0.351 | — | — | reference |
| shuffled | 0.204 | −14.7 | — | chance line sits at the nominal 0.20: no pipeline bias |
| C1 depth 2 | 0.355 | +0.4 | ≥ C0 | **adopt** — the same score with 41% of the parameters |
| C2 no transformer | 0.337 | −1.4 | within 0.5 | fails; C1 is the smaller model that ties |
| C3 no baseline | 0.363 | +1.2 | > C0 | **adopt**, the weakest of the adopts |
| C4 20–40 Hz alone | 0.204 | 0.0 over shuffled | clearly > null | **drop** — at chance exactly |
| C5 40–100 Hz alone | 0.224 | +2.0 over shuffled | clearly > null | small signal; only worth keeping if it adds to stream A |
| C7 SigLIP fixed | 0.351 | 0.0 | > C0 | not adopted; the per-head columns are unread |
| R1 regularise | 0.356 | +0.5 | > C0 | a tie |
| W05 0.5 s | 0.316 | −3.5 | ties → no | **keep 1 s** — the second half-second carries signal |
| C6 100 Hz | 0.341 | −1.0 | > C0, and by a lot | **reject** — worse, at 5x the cost |

⚠️ One seed per run, and seed-to-seed spread is unmeasured. Validation
sampling alone is roughly ±0.5 point per run (a rough estimate), so C3's +1.2
and C2's −1.4 are two or three SE, not certainties. W05 and C4 are the clear
results.

The 100 Hz question is settled against: raw high frequency (C6) hurt, the
20–40 Hz envelope carries nothing alone, and 40–100 Hz carries 2 points above
chance against stream A's 15.

## Running it

```bash
qsub -J 1-10 pbs/trial6.pbs                                    # runs 1-10, RTX6000
qsub -v RUN=11 -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S \
     -l walltime=08:00:00 pbs/trial6.pbs                       # C6 alone
```

**C9, the session probe**, reads C0's checkpoint, so it runs after run 1:

```bash
PYTHONPATH=src python -m pnpl2026.trial6.cli.session_probe \
    --ckpt runs/trial6_c0/best.ckpt \
    --config configs/base.yaml configs/trial6/trial6.yaml configs/trial6/c0_reference.yaml
```

It prints a verdict: keep the adversary only if session identity is linearly
decodable beyond a shuffled-label probe. Needs a GPU or a few CPU minutes.

**After the queue**: if C4 or C5 passes, `ab_combined.yaml` runs stream A plus
the envelopes (drop the band that failed). Adopted changes then go into a
trial 7 reference together.

## Where this departs from the design

| design says | implemented | why |
|---|---|---|
| C0 = "B with EMA" | C0 = trial 5 as it was | trial 5 already had the EMA (0.999) |
| training row: EMA, jitter, component dropout, noise | R1: jitter 0.1 → 0.03 s plus noise 0.1 | EMA and component dropout were already in B; the table never tested the rest |
| trunk "dilation 1,2,4,8,16", "receptive field ≥ 63" | unchanged: 1,2,4,8,1, reach 33 frames | trial 5 rejects 63 as overshooting 50 frames. ⚠️ C2's premise, "the trunk already sees the whole window", is therefore false |
| B's baseline "0–0.5 s" | C3 removes a **0–0.1 s** baseline | that is what trial 5 used |
| window 0.5, 1.0, 1.5 s | W05 only | the holdout serves exactly [0, 1] s; 1.5 s cannot be scored, and the cache stops at 1.28 s |
| stream A `[B, 306, 50]` | `[B, 73, 50]` | trial 5's input is trial 1's 73-component cache |
| stream B: Hilbert amplitude, 20 Hz, /5 | plus a log before the PCA | gain drift becomes an offset. `envelope_log: false` is the literal design |
| per-band PCA, ~32 components | fitted on 16,384 training windows, whitened | fixed buffers in the checkpoint, so predict applies the same map |
| stream B adds "~33k parameters" | 17,280 per two bands | the streams enter the 270-wide spatial map, not the 512 one |

Stream B reads the **raw** window, not stream A's normalised one: through the
normaliser, every envelope would be divided by broadband amplitude, and C4/C5
would no longer be about their band alone. For the same reason, C4 and C5 have
no amplitude side channel.

## Not implemented: C8

C8 (semantic targets from wav2vec 2.0 / HuBERT on each word's audio, trained
on every word) needs two things this repo does not have: the stimulus audio
per word, and a cache of **all** ~780k word windows rather than trial 1's 313k
in-vocabulary ones. Both are builds of their own before a model change.

## Two bugs inherited from trial 5, fixed here

Both break `--resume`, and both are still in `trial5/`.

- **Resume crashed.** Lightning restores the EMA from the checkpoint before it
  moves the model to the GPU, so the average stayed on the CPU and the first
  update raised a device mismatch. `on_fit_start` now moves it.
- **`last.ckpt` went stale.** On Lightning 2.6, `save_last=True` on a monitored
  checkpoint only writes when the metric improves, so a resume went back to the
  last *improvement*. A second, unmonitored callback now writes `last.ckpt`
  every epoch.

The EMA also averages parameters only now, not the fixed filter and PCA
buffers: averaging a constant only accumulates float32 rounding (~1e-6 after 70
steps).

## Cost, measured

Laptop RTX 3060, batch 512, synthetic data. HPC times are these scaled by the
GPUs' specs, not measured on them.

| run | params | step | peak memory | 20 epochs, RTX6000 |
|---|---|---|---|---|
| C0, C3, C7, R1, shuffled | 8.03M | 557 ms | 4.6 GB | ~70–90 min |
| C4, C5 | 8.00M | ≈ C0 (measured at batch 64) | ≈ C0 | ~70–90 min |
| C1 | 3.29M | 222 ms | 1.7 GB | ~30 min |
| C2 | 1.71M | 110 ms | 0.7 GB | ~20 min |
| W05 | 8.03M | 322 ms | 2.3 GB | ~45 min |
| C6 | 8.03M | ~3.3 s | ~22 GB | too big; **L40S ~3–4 h** |
