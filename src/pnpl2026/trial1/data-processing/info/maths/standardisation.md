# Standardisation

> ## v1: the plain z-score
>
> **As of 2026-09-11, `ChannelStandardiser.fit` computes:**
>
> ```python
> centre = np.mean(pooled, axis=1)
> scale  = np.std(pooled, axis=1)      # apply: (x - centre) / scale
> ```
>
> A deliberate v1 choice: the familiar z-score, **not robust to artifacts**.
> The rest of this document is the argument for the robust alternative and
> when to switch.
>
> | option | formula | pro | con |
> |---|---|---|---|
> | **v1, current** | `(x - mean) / std` | simple, one line, nothing to tune | 10 samples at 1000 sigma in 200,000 inflate the scale **614%** |
> | robust | `(x - median) / (IQR / 1.3489795)` | the same artifact moves it **0.00%** | needs values held in memory, hence subsampling |
>
> **Which one matters here is testable, not arguable.** Fit both on the real
> recordings and compare the two `scale` arrays. Close agreement means the
> upstream `bads+headpos+sss` cleaning already removed the artifacts and v1 is
> simply correct. Channels differing by 5x or more are the artifacted sensors,
> and name themselves.
>
> ⚠️ Note v1 still **subsamples** (`scale_factor=50`), which a mean and a
> standard deviation do not require -- both stream exactly in two accumulators
> over every timepoint. Dropping the subsampling would make the estimate exact
> and delete the `scale_factor`/`seed` arguments entirely.

---

## Why per-channel is necessary

The 306 sensors are **two different instruments in two different units**: 102
magnetometers interleaved with 204 planar gradiometers as `mag, grad, grad`.
Measured on a 120 s chunk of `sub-0_ses-10`:

| | std |
|---|---|
| magnetometers | 3.43e-13 |
| gradiometers | 7.43e-12 |
| **ratio** | **21.6x** |

Covariance goes as the square of scale, so in `C = (1/T) X X^T`:

```
C_gg / C_mm  ~  21.6^2  ~  466
```

Every eigenvector of `C` would then point into the gradiometer subspace, and
the magnetometers would contribute ~0.2% of the total variance. **The rank-73
whitener would be fitted on 204 sensors and would silently ignore 102 of
them** -- a decision nobody made, arrived at by arithmetic.

Standardisation makes the diagonal of `C` all ones, so every sensor enters the
covariance on equal terms and the eigenvectors are free to use all of them.

---

## Why median and IQR, not mean and std

MEG retains blinks, movement artifacts and SQUID jumps even after SSS. Both
the mean and the standard deviation have a **breakdown point of 0** -- a
single sample can move either one arbitrarily far.

### The arithmetic

Take `k` outlier samples of magnitude `M` in a recording of `T` samples with
true scale `sigma`. Their contribution to the second moment is `k M^2 / T`, so
the estimated standard deviation becomes

```
sigma_hat  =  sigma * sqrt( 1 + k M^2 / (T sigma^2) )
```

With `k = 10`, `M = 1000 sigma`, `T = 200,000`:

```
sigma_hat / sigma  =  sqrt( 1 + 10 * 10^6 / 200,000 )  =  sqrt(51)  =  7.14
```

**Ten bad samples out of two hundred thousand inflate the standard deviation
7-fold.** Divide by that and every genuine fluctuation in the channel is
squashed to a seventh of its size -- the artifact sets the scale that was
supposed to remove it.

### Measured, on synthetic data with exactly that injection

| estimator | clean | with the jump | change |
|---|---|---|---|
| standard deviation | 1.0012e-13 | 7.1414e-13 | **+613%** |
| IQR / 1.349 | 1.0022e-13 | 1.0023e-13 | **+0.01%** |

The closed form above predicts 7.14x; the measurement gives 7.13x.

### Breakdown points

| statistic | breakdown point |
|---|---|
| mean, standard deviation | **0** -- one point suffices |
| median | 0.5 -- half the data must be corrupt |
| IQR | 0.25 |

---

## Where 1.349 comes from

For `Z ~ N(0,1)`, the 75th percentile is `Phi^-1(0.75) = 0.674490`, and by
symmetry the 25th is its negative. So

```
IQR(N(0,1))  =  2 * 0.674490  =  1.348980
```

Dividing an observed IQR by this makes the robust scale **numerically agree
with the standard deviation on clean Gaussian data**, while ignoring outliers.
It is a unit conversion, chosen so the two estimators are comparable -- nothing
more. Verified: on clean Gaussian channels the fitted scale matches the true
std to within 0.25%.

---

## Why fitted on training recordings only

`centre` and `scale` are statistics of the data. Computed over all sessions,
they encode each validation session's own distribution -- and the entire point
of holding out whole sessions is to measure transfer to a session the pipeline
has not adapted to.

The leak is mild, since no label is involved, but it is free to avoid.

---

## Practical: why the fit subsamples

**Because percentiles cannot be computed incrementally, and means can.**

| statistic | running state needed | memory |
|---|---|---|
| mean | a sum and a count | 2 numbers |
| covariance | a running `X X^T` | 749 KB, whatever `T` is |
| **median / percentile** | **every value, in order to sort** | **O(T) -- 16 GB** |

To know the median you must know which value is in the middle after sorting,
and you cannot know that until you have seen all of them. There is no small
running summary. This is exactly why `ChannelStandardiser.fit` subsamples and
`Whitener.fit` does not need to.


Percentiles need the **values**, not running sums, so a thinned copy is held in
memory. Over ~130 decimated training recordings (100,750 samples each):

| | timepoints | memory (float32) | percentile SE |
|---|---|---|---|
| all | 13,097,500 | 16.0 GB | 1.4e-4 |
| every 10th | 1,309,750 | 1.6 GB | 4.4e-4 |
| **every 50th** | **261,950** | **321 MB** | **9.8e-4** |
| every 100th | 130,975 | 160 MB | 1.4e-3 |

Default is 50. A standard error near `1e-3` on a percentile is far below
anything that affects a scale estimate.

### Measured cost of thinning

13,097,500 i.i.d. samples, five seeds each:

| subsample | kept | scale | spread |
|---|---|---|---|
| 1 | 13,097,500 | 0.99977 | 0.019% |
| 10 | 1,309,750 | 0.99909 | 0.109% |
| **50** | **261,950** | **0.99844** | **0.151%** |
| 100 | 130,975 | 0.99905 | 0.200% |
| 500 | 26,195 | 1.00273 | 0.957% |

No bias, only a slowly growing sampling error. 262k samples is far more than a
percentile needs; the other 12.8 million buy a third decimal place.

---

## ⚠️ Why the draw is RANDOM, not every nth

The table above assumes **independent** samples. MEG is not independent -- it
oscillates -- and `[:, ::50]` on 125 Hz data is a **2.5 Hz sampler**.

Any component at a multiple of 2.5 Hz then lands on the *same phase* every
time. Alpha, the largest rhythm in MEG, sits at ~10 Hz, and `10 / 2.5 = 4`
exactly.

Measured on noise plus a strong 10 Hz sinusoid:

| | scale | error |
|---|---|---|
| all samples (the truth) | 2.9106 | -- |
| every 50th, strided | 0.9399 | **-67.7%** |
| 262k random timepoints | 2.9739 | +2.2% |

```
sampled phases (first 6, radians): [0. 0. 0. 0. 0. 0.]
distinct phases across all 1000 strided samples: 2
```

**A 68% underestimate of the scale**, and everything divided by it would come
out 3x too large. Real alpha is not exactly 10.000 Hz and its phase drifts, so
the locking would be partial rather than total -- but the fix costs nothing, so
there is no reason to carry the risk.

`rng.choice(n_times, n_keep, replace=False)`.

### Why without replacement

Drawing `k` from `n` with replacement leaves about `n(1 - e^(-k/n))` distinct
values -- for `k=2015` from `n=100,750` that is ~1995, so ~1% of the sample is
repeats. Measured: 1998 distinct, against the predicted 1995.

Repeats are worse than merely wasteful. The pooled sample gets sorted to find
the percentiles, and a duplicated value is counted twice, so it carries double
weight in deciding where P25 and P75 fall -- variance with no information.
Measured spread of the scale estimate over 40 seeds:

| | spread |
|---|---|
| `replace=True` | 2.700% |
| `replace=False` | 2.578% |

~5% less variance, for free.

### Why the indices are NOT sorted

An earlier version sorted them, on the reasoning that a front-to-back read
would be kinder to the cache. **Measured, that is false:**

```
scattered indices : 1.82 ms per gather
sorted indices    : 1.89 ms per gather
```

The array is C-contiguous `(306, 100750)`, so gathering columns walks across
306 rows of 403 KB each and the cost is dominated by that row stride, not by
locality within a row. NumPy also builds a transposed `(2015, 306)` temporary
internally, so the access pattern is not the one the reasoning assumed.

Percentiles are order-independent, so the sort changed nothing functionally
either. It was removed.

### A note on `.base`

Fancy indexing also removes the view-retention footgun that strided slicing
carried, but **not** in the way a `.base is None` check would suggest:

| | own size | `.base` | pins the 123 MB parent? |
|---|---|---|---|
| `[:, ::50]` | 2.47 MB | the parent, 123 MB | **yes** |
| `[:, keep]` | 2.47 MB | a `(2015, 306)` temporary, 2.47 MB | no |

Advanced indexing on axis 1 builds a transposed temporary and returns a view
of *that*, so `.base` is not `None` -- but it is the same size as the result,
and the recording is released. Do not "fix" this by adding a `.copy()`.

---

## The two guards

**Non-finite -> raise.** NaN passes silently through every comparison, and no
choice of centre or scale repairs a channel whose samples are already NaN. It
means a broken recording, so it surfaces rather than propagating.

**Zero spread -> warn and zero the channel.** A flat channel would divide by
zero. Upstream bad-channel repair should prevent it, so it signals something
wrong; the channel is passed through as zeros rather than emitting infinities
into the covariance.

---

## What comes next

The output feeds `Whitener.fit`, which accumulates `C = (1/T) X X^T` over the
standardised recordings. `apply` therefore returns **float64** regardless of
input dtype: that covariance is a sum over millions of timepoints, and float32
loses meaningful precision over that many additions.
