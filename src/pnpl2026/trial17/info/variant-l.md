# Variant L — G2 trained on holdout-matched runs

## Why

Copying the holdout's format is the only lever that has moved the
leaderboard by whole points: G2's lone 1 s words +5.8 over G0, G1's
fragments +5.3. Everything else since -- heads, word-pair fusion, per-window
normalisation, more ensemble members -- has lost.

And the public board is made of runs alone: a probe that reversed every lone
row's ranking left the score unchanged to three decimals (2026-10-07). So
every public point is on the holdout's 156 runs, and those runs are built in
a way no recipe has trained on:

| | the holdout's runs | G1's fragments | whole sentences |
|---|---|---|---|
| tokens | scored words ONLY | every word | every word |
| neighbours | adjacent, or ONE unscored word between (about a third of pairs) | adjacent | adjacent |
| length | 4-9 words (152 of 156) | 1-9 | the sentence |
| MEG | stops 1.13-1.68 s after the last onset | the same | 3 s per word |

## What L changes — one thing

Half of each epoch's training sentences become a **holdout run**
(`run_p: 0.5`): a chain of scored words with at most one unscored word
between neighbours and at most 0.8 s between onsets
(`data/sequences.py`, `holdout_chains`), a length drawn from the holdout's
histogram from 4 up (`fragment_min_words`), one placement uniform over every
long-enough chain, tokens = those scored words only, each keeping its t5
target, the MEG cut a tail of 1.13-1.68 s after the last onset
(`data/dataset.py`, `Shaping.holdout_run`). A sentence with no chain of 4+
stays whole. G2's tenth of lone words stays; the rest are whole sentences,
which keep supervising the unscored words.

Diffed: L differs from trial 13's G2 by `run_p` alone (plus the model's
registry name); per-window normalisation stays off, the baseline on, no EMA,
no jitter, no second head.

⚠️ The share of sentences that CAN supply a run is printed at the start of
every run (`[data] holdout runs of 4+ words possible in X% ...`). On the
synthetic cache it is 31%, so about 15% of items there are runs; real speech
has much shorter gaps, so expect more. If it reads below ~40% on CX3, the
change is weaker than intended -- say so before trusting a tie.

## Decision rule (fixed 2026-10-07, before any score)

The 3-seed L ensemble, compliant (`_ts`), against compliant G2 (63.41):

1. more than 2 points above: L replaces G2 in the mix
2. within 2 points: L joins the mix as one more group, the others' shares unchanged
3. more than 2 points below: L is dropped

A gap under about 1 point between files on the same rows is a tie (one row
of a single-row class moves the board by 2.000).

## Running it

From the repo root on CX3, after `git pull`.

```bash
uv run python -m src.pnpl2026.trial17.cli.training_scaler   # -> data/trial17_training_scaler.npz
qsub -J 1-3 pbs/trial17.pbs                                 # L x3, ~2-3 h each
uv run python -m src.pnpl2026.trial17.cli.collate           # -> runs/trial17_results.md

qsub -l select=1:ncpus=8:mem=32gb:ngpus=1:gpu_type=RTX6000 -l walltime=01:00:00 -N predict-17 -j oe -o logs/ <<'EOF'
cd "$HOME/PNPL-2026"; source "$HOME/.local/bin/env"
export UV_OFFLINE=1 HF_HUB_OFFLINE=1
for run in trial17_l_s0 trial17_l_s1 trial17_l_s2; do
    P="uv run --frozen python -m src.pnpl2026.trial17.cli.predict --ckpt runs/${run}/best.ckpt --holdout-dir data/holdout"
    $P
    $P --onset-shifts -1 0 1
done
EOF
```

## What was checked (2026-10-07, laptop, synthetic cache)

- `holdout_chains`: one skipped word joins a chain, two break it, a gap over
  the limit breaks it; `max_skipped 0` is adjacency only
- 3,000 draws: item shares as configured (runs scaled by how many sentences
  can supply one); every run scored-only, ascending, at most one skipped word
  and 0.8 s between neighbours, 4-9 words, tail in range
- the served item equals the drawn one: targets, columns and every window's
  length (cut at the last onset plus the tail) for 40 of 40 sentences
- smoke (25 s): one epoch, the run-share line printed, four protocols scored
  with dumps; compliant predict with onset shifts on the real holdout files;
  `collate`
- the network is trial 16's, unchanged (second head off), so no new tensor
  shapes; the shaped batches ran through it in the smoke run

## Results

### Local (2026-10-08), seed means

| run | frag val | FRAG test | ref test | lone words (isolated test) | train |
|---|---|---|---|---|---|
| trial 13 G2 | 38.44 | 31.80 | 69.32 | 27.5 / 28.9 / 31.3 (per seed) | -- |
| trial 13 G1 | 50.04 | 49.22 | 73.27 | 29.6 (s0) | -- |
| **L** | **53.37** ±1.4 | **46.85** ±3.8 | **72.22** ±2.2 | **35.07** ±0.2 | 71.31 |
| K0n (trial 16) | 33.35 | 31.76 | 66.39 | 31.59 | 65.79 |
| Kn (trial 16) | 34.73 | 30.44 | 68.77 | 33.32 | 68.36 |
| I3 (trial 15, lone-word specialist) | -- | -- | -- | **41.62** ±0.7 | 46.21 |

- L leads G2 locally everywhere: +15 on the fragment protocol, +3 on the
  full-context test, +6 on lone words. ⚠️ G1 looked the same locally
  (fragment 49, full context 73) and only tied G2 on the board; the fragment
  protocol is close to what L and G1 train on, so it flatters both. The
  board decides.
- Lone words, scored only on the private board and measured here in their
  exact format on thousands of words: I3 41.6 > L 35.1 > Kn 33.3 > K0n 31.6
  > G1/G2 ~29 > F1/w1024 ~23.

Pending: the compliant ensemble against G2 on the board.
