# ConvNeXt-style block for single-trial word decoding

**Goal**: a multi-branch residual convnet that beats `cnn_baseline` on BAcc@10
at **matched parameter count** (~989k), so a win is architectural rather than
capacity. Judged on the same session split as the baseline.

**Status**: in progress — design settled in outline, nothing implemented. No
training run has been made against any of this.

## Next action

Build the preprocessing cache, because everything downstream depends on it and
it is the only part with a verification target: rebuild one shard and compare
against the surviving `cache/v2` arrays on the old data volume, which were
built to the same design (73 components, 125 Hz, -0.32..1.28 s, 200 samples).

## Current state

**Measured, verified this session** (numbers now in `FORMAT.md`):

- `N = 313,068` in-vocabulary windows of 780,277 word events; `sub-0` only,
  162 recordings over 12 tasks; all 50 classes present.
- **Effective rank 73**, not 306 — SSS is in the processing chain and the
  noise-covariance eigenspectrum has a 135x cliff past ~74.
- **Class imbalance 121x** (`the` 41,143 / `people` 341).
- Bandpass is **0.1-125 Hz at fs=250** — no anti-alias headroom — and
  **9.97% of total power sits above 62.5 Hz**.
- Magnetometer/gradiometer scale ratio **21.6x**; 102 mags + 204 grads
  interleaved as `mag, grad, grad`.
- Recordings are ~806 s each, `float32`, `(306, ~201500)`.

**Assumed, not verified**: that the earlier pipeline's `decim: 2` was
anti-aliased. Its builder code was deleted, so treat `cache/v2` as a
plausibility check, not ground truth.

## Decisions

| decision | why | rejected alternative | why not |
|---|---|---|---|
| Whiten 306 -> 73 in the **data pipeline** | SSS leaves rank 73; a wider stem fits filters to directions that carry nothing | `Conv1d(306, D, k=1)` stem | 87 redundant directions available for amplifying numerical noise |
| **GroupNorm(8, D)**, pre-norm both sublayers | batch-independent (validation is by held-out session, where MEG statistics drift); normalises over time so the temporal profile survives | ConvNeXt's channels-last `LayerNorm` | normalises across channels *per timestep*, flattening the evoked amplitude profile — the signal |
| | | `BatchNorm1d` | batch statistics + session shift; also crashes on a final batch of 1 with `drop_last=False` |
| **mean+std pooling** first, attention pool as first ablation | zero params, no failure modes; std over time is band power and genuinely complementary | max pooling | peak in low-SNR MEG is often the noise; single-timestep gradient |
| | | committing to attention pooling now | cheapest thing in the model to ablate; confounds the read on whether the block works |
| **FIR zero-phase** anti-alias before decimating | 9.97% of power is above the new Nyquist | `scipy.signal.decimate` default (`ftype="iir"`) | frequency-dependent phase delay smears latency, which is the signal |
| | | naive `x[..., ::2]` | aliases 10% of power into the band |
| **sqrt** inverse-frequency class weights | full inverse frequency spans 121x, making the batch gradient depend on which rare tokens land in it | current full inverse frequency | almost certainly the cause of the `1e-3 -> 1e-4 -> 1e-6 -> 1e-5` walk in the git log |
| Store **-0.32..1.28 s** (200 samples), model takes the **125-sample [0,1]s core** | the holdout fixes the input length -- `predict.py` feeds `iter_windows()` straight to the model; the surrounding 75 samples are jitter budget only | the repo's current `0.0..1.0` as the *stored* window | no margin to jitter into |
| Jitter **+/-10 samples (+/-80 ms)**, crop offset 30..50, test at 40 | covers forced-alignment error plus trial-to-trial latency variation | +/-40 samples, the full margin | 320 ms moves a neighbouring word into the window centre -- words arrive ~400 ms apart |
| **Time jitter from day one**; other augmentation gated on the measured train/val gap | onset annotation has error and response latency varies trial to trial — jitter corrects a real mislabelling, not a hypothetical | full augmentation stack up front | `N=313k` is not scarce; unmeasured regularisation can mask underfitting |
| 4 branches, concat, 1x1 mixer | it is a **rank-3 factorisation of a dense k=25 conv** — 103k params where dense is 1.0M | splitting channels into 4 groups | drops the temporal basis to rank 1 per channel; 4x cheaper and meaningfully weaker |
| Dilation ramp **(1, 2, 4, 8)** across blocks | reaches full-window RF by block 4 without gridding | fixed `d=4` | dilated taps alias above ~31 Hz; the ramp plus the dense k=3/k=9 branches fill the gaps |
| `num_workers: 0`, augmentation on GPU | the 9.1 GB whitened cache sits in RAM; `__getitem__` becomes a slice | the current `num_workers: 8` | exists to hide HDF5-over-RDS latency that the cache removes |

## Constraints

- **Kamil commits, Claude does not.** Never run `git commit`; leave work in
  the working tree and say what is uncommitted. See `CLAUDE.md`.
- **Do not use the HPC unless explicitly told to.** See `CLAUDE.md`.
- The dataset is on an external volume. **That path must never reach code or
  config** — it is not mounted on the cluster or on a collaborator's machine.
- Standardiser, whitener and class weights must be **fitted on the training
  split only**, and re-derived rather than inherited from the old pipeline.
- Storage: `313k x 73 x 200` fp16 = **9.1 GB**, against 47.9 GB raw. Fits the
  64 GB the PBS scripts already request.

## To revisit before implementing

- **Step 1, the robust per-channel standardisation, is not yet clear to Kamil**
  (noted 2026-09-10). The whitening maths — covariance, eigendecomposition,
  `Lambda^-1/2`, the rank cut — landed. The median/IQR centre-and-scale did
  not. Re-explain before the cache builder is written, since a
  misunderstanding there silently corrupts everything downstream.

## Open questions

- **How much augmentation is actually needed?** Resolved by the train/val gap
  once a first run exists — not by argument.
- **`n_components`: 73, 50 or 30?** Components 30-73 hold ~5% of variance, and
  whitening promotes them to equal footing. Low variance is not low
  discriminability, but it is not high either. Resolved by a 3-point sweep.
- **What length does `iter_windows()` actually yield?** Inferred as 125 samples
  (`[0,1]s` at 125 Hz) from the repo's `Config` defaults, but unverified --
  `pnpl` is not installed locally. It fixes the model's input length, so
  confirm before building the cache.
- **FFN expansion 2 or 4?** At `N=313k`, 4 is affordable. Second knob.
- **Was `cache/v2` anti-aliased?** Resolved by rebuilding a shard and
  comparing; a mismatch concentrated above 30 Hz would say no.

## Pointers

- Competition, data, rank, imbalance, HPC: `FORMAT.md`
- Rules and the model-adding procedure: `CLAUDE.md`
- Metric implementation: `src/pnpl2026/metrics.py`
- Label-space trap: `src/pnpl2026/data/vocab.py`
- Old volume's surviving cache + `meta.npz`: `Neuro-Hackathon/cache/` on the
  external drive; its `README.md` documents the array contracts.
