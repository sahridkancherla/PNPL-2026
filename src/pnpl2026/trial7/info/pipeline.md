# The trial-7 pipeline — stage order and every shape between stages

`N` words in a batch, `S` sequences, `W` the longest sequence, `L_i` word i's
real samples (250–750). Measured from the built models.

## Data

| stage | out |
|---|---|
| `build_continuous`: h5 → trial 1's standardiser and whitener → clip → fp16 | `meg_<key>.npy` `[73, n_times]`, `words_<key>.npz` |
| `build_sequences`: in-vocabulary words → sentences → runs of ≤ 9, 9.6% alone | `Sequence` per example |
| `SequenceWindows`: slice `[onset, min(onset + 750, end))`, zero-pad | `[n_words, 73, 750]` + `L_i` |
| `collate_tokens`: flatten sequences | windows `[N, 73, 750]`, lengths, labels, sequence, position `[N]` |

## Forward, D1

| stage | in | out | learned |
|---|---|---|---|
| zero past `L_i` | `[N, 73, 750]` | `[N, 73, 750]` | no |
| `RobustInstanceNorm`, real samples only | `[N, 73, 750]` | `[N, 73, 750]` + `[N, 73]` | no |
| `BaselineCorrect` 0–0.5 s, re-zero padding | `[N, 73, 750]` | `[N, 73, 750]` | no |
| `FixedLowpassDecimate` 20 Hz, /5 | `[N, 73, 750]` | `[N, 73, 150]` | no |
| `GaussianNoise`, `ComponentDropout`, frame mask | `[N, 73, 150]` | `[N, 73, 150]` | training only |
| `spatial`, `initial_linear` | `[N, 73, 150]` | `[N, 512, 150]` | yes |
| `DilatedConvTrunk` 1,2,4,8,16 | `[N, 512, 150]` | `[N, 256, 150]` | yes |
| `.transpose(1, 2)` | `[N, 256, 150]` | `[N, 150, 256]` | — |
| frame `FlatTransformer`, depth 0: amplitude bias + norm | `[N, 150, 256]` | `[N, 150, 256]` | yes |
| `AttentionPool`, real frames only | `[N, 150, 256]` | `[N, 256]` | yes |
| scatter to a grid, mask 15% (training) | `[N, 256]` | `[S, W, 256]` + `[S, W]` | — |
| sentence `FlatTransformer`, depth 4, key-padding mask | `[S, W, 256]` | `[S, W, 256]` | yes |
| gather | `[S, W, 256]` | `[N, 256]` | — |
| `ClosedSetHead`, `SemanticHead` | `[N, 256]` | `[N, 50]`, `[N, 1024]` | yes |

A frame is real when its centre, `5f`, is below `L_i`: 50 frames for a 1 s
window, 150 for 3 s.

## Per run

| run | trunk reach | frame depth | sentence depth | params |
|---|---|---|---|---|
| D1, D3, D4, D6, D8, D9 | 63 | 0 | 4 | 4.44M |
| D0 | 63 | 2 | 0 | 2.86M |
| D2 | 63 | 0 | 0 | 1.28M |
| D5a | 63 | 0 | 2 | 2.86M |
| D5b | 63 | 0 | 8 | 7.60M |
| D7 | 127 | 0 | 4 | 4.65M |

D1 against the design's budget: sentence transformer 3.16M (≈3.1M), heads
0.28M (≈0.28M), trunk 0.83M (≈1.0M, from the design's misread trunk).

## Checked, and how

| check | result |
|---|---|
| masked robust norm vs the same window unpadded | identical; padding stays zero |
| garbage past a window's end | output unchanged |
| a word encoded alone vs in a batch | identical |
| a sentence batched with longer ones | identical — the key-padding mask works |
| causal: change the last word | earlier words unchanged |
| D8: change a word's own window | its prediction unchanged, its neighbours' not |
| training masking | 15.0%, never a one-word sequence |
| SigLIP | matches −1/N Σ_i Σ_j; ~10 at initialisation, not ~10/N |
| sequences | every word once; ≤ 9 per run; never across sentences; tails in range |
| `build_continuous` on synthetic h5 + TSVs in the real layout | other subjects and unsplit sessions skipped; onsets exact; case-insensitive labels |
| builder vs `predict` transform | equal to float16 rounding |
| all 11 configs through `cli.train`, then a resume | all run; resume continues from the last epoch |
| gradient probe at initialisation | SigLIP 0.70x cross-entropy on the shared vector |
| `cli.predict`, both modes, on subject 0's **real** holdout files | 960 rows in order; isolated rows identical across modes; 6 sentence rows recomputed by hand match to 5e-7 |
| `pbs/trial7.pbs` locally with uv swapped for the venv | preflight, staging, seeded run name, training; a failed preflight stops the job |

⚠️ Not yet checked: anything on real MEG. The builder has never read a real h5
or events table — CLAUDE.md's documented layout is what it was tested against.
