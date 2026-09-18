# pipeline/ — building the trial-1 cache

Turns 162 continuous recordings into one array of windows the model can read.
Every transform here is fitted on **training recordings only**.

## Router

| where | what |
|---|---|
| `../preprocessing.py` | fitted transforms: `ChannelStandardiser`, `Whitener` |
| `maths/decimation.md` | aliasing, and why a 250 -> 125 Hz step needs a filter. No code implements this |
| `maths/standardisation.md` | ⚠️ **opens with a TEMPORARY notice** -- the fitted scale is a known-broken placeholder. Then: the 466x argument, robust vs plain, where 1.349 comes from |
| `terminology.md` | array dimensions and tensor axis naming conventions (`n_windows`, `n_channels`, `n_samples`) |

## Not written yet

Each document lands with the code it describes, so neither drifts ahead.

| planned | lands with |
|---|---|
| `maths/whitening.md` | `Whitener` |
| `maths/windowing.md` | the window cutter |

## The stage order

`fit standardiser → fit whitener → apply → cut windows → clip → fp16`

Everything runs at the native **250 Hz**, so a window is **400 samples** and
the cache is **18.3 GB**. Measured numbers for the dataset itself — N, rank,
class imbalance, bandwidth — live in `CLAUDE.md` at the repo root, not here.
