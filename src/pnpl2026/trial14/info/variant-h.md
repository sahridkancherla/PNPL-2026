# Variant H — G2 plus a trained 50-word head, judged on holdout-shaped chunks

Trial 13's G2 (isolated-word training) is the best leaderboard submission not
chosen by its score: **63.41** as a 3-seed ensemble. Trial 14 changes two
things on top of it.

## Part 1 — a 50-word head

**The problem.** G2 picks words by retrieval: its output vector is compared
with t5-large's embeddings of the 50 candidates and the most similar win. It
was trained only to match t5's vector for each word, never to rank the 50
candidates against each other, and the leaderboard metric asks exactly that:
is the true word in the top 10, with every word weighted equally. One visible
symptom, measured in the trial 10/12 submissions: a word's share of top-10
lists runs from 4.5% to 48% of rows where an even spread would be 20%.

**The head.** A dropout and one linear map from the sentence transformer's
output (`context`, 256 wide) to 50 logits, trained with **logit-adjusted
cross-entropy**: cross-entropy on `logits + tau * log(prior)`, the prior
counted from the training tokens. Training on the adjusted logits makes the
plain logits rank for balanced accuracy (Menon et al. 2021, "Long-tail
learning via logit adjustment") without class weights, whose 121x span
CLAUDE.md warns against. At scoring time the head's and retrieval's
log-probabilities are blended; `inference_closed_weight` is 0.5, and every
evaluation also scores a sweep of weights (0, 0.25, 0.5, 0.75, 1).

Every run starts from its seed's **G2 checkpoint**: the network's weights are
trial 13's key for key, and only the head is new.

Not verified: that t5 places common function words close together, the
explanation offered when this was proposed. The RDS drive was unreachable when
it was to be checked.

## Part 2 — the chunked protocol

Trial 13's fragment protocol copied the holdout most literally -- natural runs
of 4+ consecutive vocabulary words -- but such runs are rare in pnpl's
narrative recordings: **128 test words**, ±4-5 points of noise per run, too few
to rank anything. Its +17 for fragment training became +5 on the leaderboard.

The **chunked** protocol keeps the holdout's geometry on every scored word:

- 9.6% of vocabulary words are set aside and served alone with exactly 1 s
- the rest of every sentence, scored and unscored words alike, is cut in order
  into chunks whose lengths are drawn from the holdout's histogram (lengths
  1-9: 2, 0, 2, 24, 55, 37, 23, 11, 2)
- each chunk's MEG stops 1.13-1.68 s after its last onset, as the holdout's
- only vocabulary words are scored: **all ~2,700 test words**, each once

The holdout's tokens are consecutive words; so are a chunk's. The difference
is that a chunk's tokens include unscored words.

`pbs/trial14_calibrate.pbs` scores all 23 finished trial 10/12/13 checkpoints
under it and checks it against the leaderboard: over the **9 single runs**
with a score, over the **7 submitted ensembles** (rebuilt from their members'
per-word dumps), and over both together.

## The queue — `configs/trial14/runs.txt`, 3 configs × 3 seeds

Seed k starts from trial 13's G2 seed k. Fine-tuning: network at lr 3e-5, the
new head at 1e-3, at most 15 epochs, patience 5, G2's isolated-word items kept.

| run | trains | losses | scored as | answers |
|---|---|---|---|---|
| H0 | the head only (network frozen, BatchNorm statistics fixed) | head | 0.5 head + 0.5 retrieval | what a head is worth on G2's own features |
| H1 | head and network | D-SigLIP + head | 0.5 + 0.5 | does training the network for the head add to H0? |
| H2 | the network, no head | D-SigLIP | retrieval | the control: is any gain just more training? |

Read H0 and H1 against H2, by seed means. Confirm on the leaderboard with each
config's 3-seed ensemble.

## Running it

From the repo root on CX3. The two jobs are independent.

```bash
qsub pbs/trial14_calibrate.pbs          # no training; -> runs/trial14_calibration.md
qsub -J 1-9 pbs/trial14.pbs             # the 9 fine-tuning runs
uv run python -m pnpl2026.trial14.cli.collate      # -> runs/trial14_results.md
```

`trial14_results.md` reports `w*`, the blend weight best on the VALIDATION
chunks, and the test score at that weight. To submit a config's ensemble at
its own weight (0.5):

```bash
qsub -l select=1:ncpus=8:mem=32gb -l walltime=01:00:00 -N predict-14 -j oe -o logs/ <<'EOF'
cd "$HOME/PNPL-2026"; source "$HOME/.local/bin/env"
export PYTHONPATH="$PWD/src" UV_OFFLINE=1 HF_HUB_OFFLINE=1
for config in h0 h1 h2; do
    for seed in 0 1 2; do
        uv run --frozen python -m pnpl2026.trial14.cli.predict \
            --ckpt runs/trial14_${config}_s${seed}/best.ckpt --holdout-dir data/holdout
    done
done
EOF
S=submissions
for config in h0 h1 h2; do
    uv run python -m pnpl2026.trial14.cli.ensemble --out $S/trial14_${config}_ens_in_context.csv \
        --inputs $S/trial14_${config}_s0_in_context.csv $S/trial14_${config}_s1_in_context.csv $S/trial14_${config}_s2_in_context.csv
done
```

To use the validation-chosen weight instead, add `--closed-weight <w*>` to
predict; its files are then named `..._in_context_w<w*>.csv`. Never choose the
weight on the leaderboard. Record every score in
`configs/trial13/leaderboard.csv`, and every ensemble's members in
`configs/trial14/ensembles.csv`, so the next calibration can use them.

## What was checked (2026-10-05)

- **chunked protocol**, synthetic recordings in pnpl's layout: every eligible
  vocabulary word scored exactly once (391 of 391, against the fragment
  protocol's 78); chunks are consecutive words of one sentence, gaps only
  where a word was served alone; sizes follow the holdout's shape; tails
  1.13-1.68 s; isolated 9.7%, exactly 1 s; deterministic
- **logit adjustment**: tau 0 is plain cross-entropy, tau 1 is cross-entropy
  on logits + log prior; on a toy problem with a 100x long tail, balanced
  accuracy 0.66 plain against 0.79 adjusted
- **blend**: weight 0 is retrieval alone, 1 the head alone, and a run without
  a trained head is retrieval alone whatever the weight
- **shape-check skill**: primes and real dimensions in fp16, H0-H2 on real
  shaped and chunked batches; joint mode reaches every parameter; frozen
  mode reaches the head's two tensors only, and three optimiser steps move
  none of the network's weights or BatchNorm statistics. It caught one real
  bug: Lightning's `save_hyperparameters` raised on the NumPy prior. Fixed
- **smoke runs**, H0-H2 one epoch each from a G2-format parent on synthetic
  recordings: every run loaded its parent, trained and scored itself (three
  protocols, dumps). After the fix below, H0's best.ckpt keeps all 153 of the
  parent's network tensors identical; H1 and H2 changed 151. A trial 13
  checkpoint and H2 load as retrieval-only; H0 and H1 with their head; each
  head run's sweep ends equal retrieval and head alone
- ⚠️ the smoke run caught a second bug: Lightning restores submodule modes
  around validation, so frozen H0's BatchNorm statistics drifted despite the
  `train` override. Fixed by re-asserting eval mode in `on_train_batch_start`
- `collate` and `calibrate` (singles, ensembles rebuilt from dumps, an
  ensemble with no members present skipped) on the smoke runs
- **predict on the real holdout files**: at weight 0 trial 14 reproduces trial
  13's CSV for the same checkpoint (largest difference 1e-6, CSV rounding; the
  same top-10 set in all 960 rows); H1 writes its blend, `--closed-weight 0.25`
  changes the primary columns only; a head-less run refuses a weight above 0

## Results (2026-10-05) — negative on the leaderboard

BAcc@10, %. Local = chunked test, seed mean. LB = leaderboard, 3-seed
ensemble at the run's own weight; G2's 63.41 is the reference. The
leaderboard itself carries roughly ±2 points (960 rows, an estimate).

| run | local | LB | LB vs G2 |
|---|---|---|---|
| G2 (parent, trial 13) | 49.66 | 63.41 | — |
| H0 head on frozen G2 | 52.25 | 61.61 | −1.8 |
| H1 head + network | **55.11** | 61.59 | −1.8 |
| H2 network, no head | 51.32 | 59.63 | −3.8 |
| H1, head alone (w = 1) | 54.54 | 59.20 | −4.2 |
| H1 + G1, 6 members | — | 63.21 | −0.2 |

- **The head does not transfer.** On G2's exact network (H0) it costs 1.8;
  the more weight it gets, the lower the score (w 0: 63.41, 0.5: 61.6,
  1: 59.2). Locally it gained 2.6 (H0) to 5.5 (H1).
- **More fine-tuning hurt**: H2 is G2 plus up to 15 epochs selected on
  chunked validation, +1.7 locally, −3.8 on the leaderboard. H1 − H2 is +2.0
  on the leaderboard, in the local direction, but both sit below G2.
- **Validation-chosen weights** (`w*` 0.92 for H0, 0.75 for H1) scored no
  better on test than 0.5 locally; not submitted.
- **The chunked protocol inverts trial 13's singles**: G2 s0 is the lowest of
  23 runs locally (47.83) and scored 62.82; G1 s1 is among the highest
  (60.08) and scored 57.32. Calibration over the 23 older runs
  (`runs/trial14_calibration.md`): Spearman +0.47 over 9 singles, −0.29 over
  7 ensembles, before these two points were added. Top-1 agreement matches
  the holdout (0.326 local, 0.330 holdout), so the geometry is right and the
  ranking is not; why is open.
- G2 s0 alone (62.82) is within 0.6 of G2's ensemble.

### Ensembles of existing predictions, no training (2026-10-05)

| ensemble | members | LB |
|---|---|---|
| G1 + G2 | 6 | 64.47 |
| G2 + F1 + w1024 | 5 | 65.48 |
| **G1 + G2 + F1 + w1024** | 8 | **66.39** — best so far; up 12 places from 63.41 |
| G1 + G2 + G3 | 9 | 61.27 |
| G1 ens + G2 ens + (F1, w1024) ens, each a third | 8 | 64.92 |

Mixing recipes pays: adding trial 10/12's whole-sentence pair (F1, w1024)
gained about 2 on both G2 and G1 + G2. Adding G3, which combines G1's and
G2's switches, cost 3. Giving the F1 + w1024 pair a third of the vote
instead of a quarter scored 1.5 lower: more weight on two models is not more
diversity. Next: seeds 1-2 of F1 and w1024. Each step is near the leaderboard's ±2 noise, and the
best is the maximum of ~15 tries, so expect it to read a little high.
