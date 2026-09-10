# pipeline/ — building the trial-1 cache

Turns 162 continuous recordings into one array of windows the model can read.
Every transform here is fitted on **training recordings only**.

## Router

| where | what |
|---|---|
| `../preprocessing.py` | fitted transforms: `ChannelStandardiser`, `Whitener` |
| `maths/standardisation.md` | why median/IQR, where 1.349 comes from, the 466x argument |

## Not written yet

Each document lands with the code it describes, so neither drifts ahead.

| planned | lands with |
|---|---|
| `maths/whitening.md` | `Whitener` |
| `maths/decimation.md` | the FIR resampler |
| `maths/windowing.md` | the window cutter |

## The stage order

`decimate → fit standardiser → fit whitener → apply → cut windows → clip → fp16`

Fitting happens after decimation so the transforms are fitted on exactly the
data the model sees. Measured numbers for the dataset itself — N, rank, class
imbalance, bandwidth — live in `FORMAT.md` at the repo root, not here.
