# Standardisation

**Stage 1 of the cache pipeline.** Implemented as `ChannelStandardiser` in
`../../preprocessing.py`.

## The operation

For each channel `c` independently, two fitted numbers and one elementwise
transform:

```
x_c(t)  <-  ( x_c(t) - centre_c ) / scale_c
```

Nothing mixes between channels. 306 pairs of numbers, applied along each row.

```
centre_c  =  median over time of channel c
scale_c   =  ( P75 - P25 ) / 1.349
```

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

⚠️ **Thinning must `.copy()`.** `recording[:, ::50]` is a strided **view** that
keeps its whole parent alive through `.base`. Over 130 recordings that is
16.0 GB resident instead of 321 MB -- and it only bites when the caller passes
a generator, which is exactly how the cache builder calls it.

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
