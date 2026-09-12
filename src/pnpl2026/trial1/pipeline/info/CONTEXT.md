# pipeline/ — building the trial-1 cache

Turns 162 continuous recordings into one array of windows the model can read.
Every transform here is fitted on **training recordings only**.

## Router

| where | what |
|---|---|
| `../decimation.py` | 250 -> 125 Hz: `design_antialias_filter`, `decimate_recording`. Nothing fitted |
| `../preprocessing.py` | fitted transforms: `ChannelStandardiser`, `Whitener` |
| `maths/decimation.md` | why naive striding puts the alias at **parity** with the signal, why scipy's default filter is too weak, and the 56.25-62.5 Hz band given up to buy -80 dB |
| `maths/standardisation.md` | ⚠️ **opens with a TEMPORARY notice** -- the fitted scale is a known-broken placeholder. Then: the 466x argument, robust vs plain, where 1.349 comes from |

## Not written yet

Each document lands with the code it describes, so neither drifts ahead.

| planned | lands with |
|---|---|
| `maths/whitening.md` | `Whitener` |
| `maths/windowing.md` | the window cutter |

## The stage order

`decimate → fit standardiser → fit whitener → apply → cut windows → clip → fp16`

Fitting happens after decimation so the transforms are fitted on exactly the
data the model sees. Measured numbers for the dataset itself — N, rank, class
imbalance, bandwidth — live in `FORMAT.md` at the repo root, not here.
