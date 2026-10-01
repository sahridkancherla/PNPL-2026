# Variant E — attention reach per sentence block, as implemented

Trial 7's D1 network with one new knob: each block of the sentence
transformer gets a **reach**, how many words either side a word may read.
`null` is the whole sentence, `1` is the immediate neighbours. The schedule of
reaches across the four blocks is the only thing that varies. The idea and the
checks are in `reach.md`, the network drawn end to end in `architecture.md`;
this file is the record of the queue and what it found.

## What carries over from Variant D

Everything about the data is trial 7's, unchanged: the same continuous cache,
the same holdout-matched sequences, the same validation windows. Trial 7's
preconditions, holdout measurements and rules question all still apply — see
`../../trial7/info/variant-d.md`.

The base network is D1 with two of its ablations folded in:

| | trial 7 D1 | trial 8 base | from |
|---|---|---|---|
| SigLIP | deduplicated | **plain** | D4, 0.467 vs 0.460 |
| word-token masking | 15% | **off** | D6, 0.465; a masked neighbour under reach 1 would also remove half a word's context |
| sentence depth | 4 | 4 | D5b, depth 8, gained only +0.004 |
| everything else | — | unchanged | |

Because the base changed, E0 reruns it rather than borrowing D4 or D6.

## Why reach

Trial 7's D8 found a word's neighbours alone nearly match its own window (0.361
against 0.364 on sentence words), so most of the context gain could come from
nearby words. Trial 8 asks whether steering attention towards neighbours — a
**locality prior** — helps, the way windowed attention (Swin) helps on image
grids.

It is not a cost saving: sentences hold at most 9 words, so full attention
costs nothing. Reach is a mask, not a weight, so every run has D1's 4,442,334
parameters.

## The queue

`configs/trial8/runs.txt`: 4 runs × 3 seeds, one wave on the 12-GPU limit.

| lines | run | `sentence_reach` | reach after block 4 | question |
|---|---|---|---|---|
| 1–3 | `e0_full` | `[null, null, null, null]` | whole sentence | the control |
| 4–6 | `e1_local_then_full` | `[1, 1, null, null]` | whole sentence | neighbours first, then global |
| 7–9 | `e2_full_then_local` | `[null, null, 1, 1]` | whole sentence | global first, then local refinement |
| 10–12 | `e3_local_only` | `[1, 1, 1, 1]` | ±4 words | does anything beyond ±4 matter? |

**Adopt a schedule if it beats E0 by more than the seed spread** on
`val/bacc10`. Read `sentence` and `isolated` apart: reach cannot touch an
isolated word, which has no neighbours to mask, so `isolated` should match
across E0–E3. If it does not, something other than reach changed.

## Results

All 12 runs complete, from `runs/trial8_results.md` via `cli.collate`. Cells
are mean ± SD of `val/bacc10` over 3 seeds; chance is 0.20.

| run | BAcc@10 | sentence | isolated | Δ vs E0 | rule | verdict |
|---|---|---|---|---|---|---|
| **E0 full** | **0.466 ± 0.001** | **0.480** | **0.337** | — | reference | trial 7's D1 + plain SigLIP + no masking |
| E1 local, then full | 0.464 ± 0.002 | 0.478 | 0.336 | −0.002 | > E0 + spread | not adopted — a tie |
| E2 full, then local | 0.463 ± 0.002 | 0.477 | 0.332 | −0.003 | > E0 + spread | not adopted — a tie |
| E3 local only, ±4 words | 0.463 ± 0.003 | 0.477 | 0.333 | −0.003 | > E0 + spread | not adopted — a tie, and the informative one |

**Adopt rule, as `cli.collate` computes it:** every gap to E0 is −0.002 or
−0.003, against seed SDs of 0.002–0.003. No schedule beats E0; none is
reliably worse either. **Nothing is adopted; E0 stays the base.**

**The sanity check holds.** `isolated` sits at 0.332–0.337 across all four,
inside its own seed SD of 0.007–0.013. Reach changed nothing it should not
have.

**E3 is the finding.** Restricting every block to its immediate neighbours,
so a word can never read anything more than four words away, costs nothing
measurable. Whatever context the sentence transformer uses lies within ±4
words — which fits D8's result that neighbours carry most of it. In a 9-word
sentence, the far end has nothing to add.

**The order of local and global blocks does not matter either.** E1 and E2
use the same ingredients in opposite orders and land 0.001 apart. The
sentence transformer is insensitive to how its attention is structured, so
there is nothing left to tune there.

**Combining D4 and D6 did not stack.** E0, which is D1 with both, scores
0.466 — the same as D4 (0.467) or D6 (0.465) alone. Neither small gain added
to the other.

**Runs converged faster than trial 7's.** Every seed early-stopped, best epoch
14–18 against D1's 24–31; none hit the walltime. Trial 7's D4 (plain SigLIP)
did the same, so plain SigLIP is the likely cause rather than reach.

**Overfitting is unchanged.** Training BAcc@10 is 0.59–0.62 against 0.46 on
validation, the same ~14-point gap as trial 7's D1.

**What this does not settle.** As in trial 7, the 46.6% depends on using MEG
outside each word's own 1 s window, which the rules do not settle. Under
`predict --window-only` every word is alone, so all four runs are identical and
score like the `isolated` split, about 34%.

## What to take forward

- Attention structure in the sentence transformer is exhausted as a lever:
  depth (trial 7's D5), causal masking (D3) and reach (here) either cost or
  tie. The bottleneck is elsewhere.
- E0 is the strongest model so far, 0.466, by a margin inside seed noise of
  trial 7's D4.
- Since context beyond ±4 words adds nothing, nothing argues for longer
  sequences or a bigger sentence transformer.

## Running it

```bash
qsub -J 1-12 pbs/trial8.pbs                              # needs trial 7's cache on RDS
PYTHONPATH=src python -m pnpl2026.trial8.cli.collate     # runs/trial8_results.md
```

`cli.predict` has trial 7's two modes; the default runs each holdout sentence
whole, and `--window-only` is the only mode known to be within the rules.

## Cost

Measured from the logs: every seed early-stopped between epochs 25 and 29,
inside trial 7's 2 h walltime with no resubmissions. All four runs cost the
same as trial 7's D1 per step — reach is a mask on a 9-word sequence.
