# Hub correction — querybank normalisation, frozen from validation

## Why

Every model answers with the nearest of the 50 words' t5 vectors, and a few
words sit near the centre of that space. On uncertain rows those **hubs** land
in the top 10 whatever the true word: trial 16's hub check found the most
central word in 92-100% of simulated uncertain top-10 lists and others in
none; on the holdout, trial 10/12 models gave words 4.5-48% of top-10 places
against a fair 20%. Balanced top-10 accuracy pays for every place a hub
wastes. Hub correction is one plausible way the leaders reach 72-76 -- some
may use statistics over the whole holdout, which the rules allow and our
principle does not.

**Querybank normalisation** (Bogolin et al. 2022, arXiv 2112.12777) corrects
hubs with a bank of training-side queries, never other test queries. For 50
fixed candidate words its correction reduces to fixed numbers per word, so it
is fitted once on a validation partition's predictions and frozen: within the
holdout principle. Labels are never used.

## The two fits (`hubnorm.py`)

| fit | per word | effect |
|---|---|---|
| `zscore` | the bank's mean and spread of that word's scores; each score centred and scaled by them | the frozen version of column normalisation |
| `balance` | an offset, iterated until every word reaches the top 10 in its fair 20% of bank rows | targets the metric's top-10 places directly |

`balance_half` applies half of each offset. Run words and lone words get
separate banks (their scores differ: lone words have no context), split by
the dumps' isolated flag; a lone bank under 100 rows is not fitted.

## The rule (fixed 2026-10-07, before any score)

On the local test words, against no correction: run words take the fit with
the largest gain (none written if no fit gains; a warning under 0.5 points);
lone words -- scored on the private board only -- take a fit only if it gains
more than 1.0 (fewer words, more noise), else stay as they are. On the
board: one on/off pair, the corrected file against its parent.

## Running it

From the repo root on CX3 (login node, seconds). The bank and the check come
from the CSV's members' dumps; for the compliant 66.39 mix those are trial
14's calibration dumps:

```bash
C=runs/trial14_calibration
uv run python -m src.pnpl2026.trial17.cli.hubnorm \
    --csv submissions/ens_g1g2_f1w1024_ts.csv \
    --runs $C/trial13_g1_s0 $C/trial13_g1_s1 $C/trial13_g1_s2 \
           $C/trial13_g2_s0 $C/trial13_g2_s1 $C/trial13_g2_s2 \
           $C/trial10_f1_s0 $C/trial12_w1024_s0 \
    --holdout-dir data/holdout
# -> submissions/ens_g1g2_f1w1024_ts_hub.csv, runs/trial17_hubnorm.md
```

Members without chunked dumps (trials 15-17's runs) can use
`--bank-protocol fragment`: every trial's fragment protocol cuts identical
sequences, but its validation bank is small (~240 words).

## What was checked (2026-10-07, laptop)

- `balance` on a bank with four planted hubs: top-10 shares 15-78% become
  19.8-20.3%, the hubs get the largest offsets, half strength lands in
  between; `zscore` flattens the shares to 19-21%
- the CLI on fabricated chunked dumps (two runs; run and lone words with
  different planted hubs; unscored tokens): fitted on validation, checked on
  test (run words +2.7 with zscore, lone words +3.6 with balance), and applied
  to a CSV in the real holdout's layout: same index and columns, sentence rows
  corrected, Moses unchanged, rows distributions, no rank-10/11 ties

## Results

Pending: `runs/trial17_hubnorm.md` on the real dumps, and the on/off pair on
the board.
