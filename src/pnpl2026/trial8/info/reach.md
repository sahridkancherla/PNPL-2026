# Trial 8 — attention reach per sentence block

## The idea

Windowed attention on image grids (Swin) changes the window shape between
layers so information crosses window borders without every point attending to
every other. Trial 8 borrows the layer-by-layer half of that idea for the
sentence transformer: **each block gets a reach** — how many words either side
a word may read — and the schedule of reaches is the only thing that varies.

```
reach 1 (neighbours)            reach null (whole sentence)
    a  b  c  d  e                   a  b  c  d  e
c   ·  ●  ●  ●  ·               c   ●  ●  ●  ●  ●
```

With reach 1, information hops one word per block: after `[1, 1, ...]` a word's
vector holds its ±2 neighbourhood, and later full blocks read words that have
already absorbed their neighbours. The order of blocks is what differs between
E1 and E2 — the same ingredients, a different computation.

Not a cost saving. Sentences hold at most 9 words, so full attention costs
nothing; reach can only restrict what trial 7 already allows. The question is
whether a **locality prior** helps: trial 7's D8 found a word's neighbours
alone nearly match its own window (0.361 vs 0.364 on sentence words), so
neighbours carry much of the signal.

## Base: trial 7's D1, with two of its ablations folded in

| | trial 7 D1 | trial 8 | from |
|---|---|---|---|
| SigLIP | deduplicated | **plain** | D4, 0.467 vs 0.460; Sahrid's call |
| word-token masking | 15% | **off** | D6, 0.465; and a masked neighbour under reach 1 removes half a word's context, which would confound the comparison |
| sentence depth | 4 | 4 | D5b (depth 8) gained only +0.004 |
| everything else | — | unchanged | |

Because the base changed, E0 reruns it rather than borrowing D4 or D6.

## The queue — `configs/trial8/runs.txt`, 4 runs × 3 seeds

| lines | run | `sentence_reach` | reach after block 4 | question |
|---|---|---|---|---|
| 1–3 | `e0_full` | `[null, null, null, null]` | whole sentence | the control |
| 4–6 | `e1_local_then_full` | `[1, 1, null, null]` | whole sentence | neighbours first, then global |
| 7–9 | `e2_full_then_local` | `[null, null, 1, 1]` | whole sentence | global first, then local refinement |
| 10–12 | `e3_local_only` | `[1, 1, 1, 1]` | ±4 words | does anything beyond ±4 matter? |

**Adopt a schedule if it beats E0 by more than the seed spread** on
`val/bacc10`. `cli.collate` prints each run's gap to E0 with the larger seed
SD. Trial 7's seed SDs were 0.001–0.004, so ~0.01 is readable.

**Read `sentence` and `isolated` apart.** Reach cannot touch isolated words (a
one-word sentence has nothing to mask), so `val/bacc10_isolated` should match
across E0–E3 to within seed noise. If it does not, something other than reach
changed.

```bash
qsub -J 1-12 pbs/trial8.pbs
PYTHONPATH=src python -m pnpl2026.trial8.cli.collate    # runs/trial8_results.md
```

## Parameters

Reach is a mask, so all four runs are **4,442,334** — identical to trial 7 D1,
measured from the built models.

| part | params |
|---|---|
| per-word encoder | 1,006,508 |
| sentence transformer, 4 blocks × 789,760 + final norm | 3,159,552 |
| closed-set head | 12,850 |
| semantic head | 263,168 |
| mask token (unused at `word_mask_p: 0`, kept for the match) | 256 |

## Checked, and how — 2026-09-30

| check | result |
|---|---|
| band mask, 5 words, last one padding | `|i − j| ≤ 1` on real keys; padding row sees itself |
| no reach, no padding, not causal | mask is `None`: SDPA's fast path, as trial 7 |
| bad reach lists: wrong length, 0, `true`, 1.5 | all raise at config load, so the PBS preflight catches them |
| all four configs built through `load_trial8_config` | 4,442,334 each; dedup off; mask p 0 |
| same init, E0 vs E1–E3 | isolated word identical (0.0); 9-word sentence differs |
| perturb word 0 of a 9-word sentence | E0–E2 move all 9 words; **E3 moves words 0–4 only**; a 3-word sentence in the same batch never moves |
| backward, E1 | every parameter but the unused mask token has a gradient |
| batch of one isolated word | `[1, 50]`, finite |
| `LitTrialEight._step`, train and val, real t5 table | loss ~14 at init, finite — SigLIP ~10 + CE ~4, as trial 7 |
| `cli.collate` on an empty runs folder | writes the E0–E3 table |
| `pbs/trial8.pbs` | `bash -n` passes |

⚠️ Not checked: `cli.train` end to end. It needs `data/trial7_continuous/`,
which is on RDS and not on this machine. Every piece it calls is trial 7's,
unchanged, except the model and the config above.

⚠️ Walltime is trial 7's 2 h. Trial 7's D7 had a seed cut at epoch 18 by it;
`--resume` recovers, but a 12-line wave may need a second submission for the
seeds that do not early-stop in time.
