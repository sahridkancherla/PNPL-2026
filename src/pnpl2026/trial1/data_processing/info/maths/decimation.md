# Decimation

Taking 250 Hz to 125 Hz. `../../decimation.py`.

## Why a filter is not optional

The recordings are bandpassed **0.1-125 Hz at fs=250** — right up to their own
Nyquist, with no anti-alias headroom left by the upstream chain. Measured on
`sub-0`, **9.97% of total power sits above 62.5 Hz**.

Downsampling by 2 makes 62.5 Hz the new Nyquist. Any content above it does not
disappear; it **folds** to `|f - 125|` and adds to whatever real signal already
occupies that frequency. 70 Hz lands on 55 Hz, 80 Hz on 45 Hz, 100 Hz on 25 Hz
— straight into the alpha and beta bands the decoding depends on.

Measured, a 10 Hz signal plus a 70 Hz contaminant:

| path | 10 Hz kept | 55 Hz alias | alias / signal |
|---|---|---|---|
| `x[:, ::2]` | 2.3433e+06 | 2.3433e+06 | **1.00** |
| `decimate_recording` | 2.3433e+06 | 1.1591e-03 | **4.95e-10** |

**Naive striding puts the alias at full parity with the signal.** Equal power,
inseparable, and no later stage can undo it.

The broadband version of the same check, on white noise:

| path | variance retained |
|---|---|
| `x[:, ::2]` | 1.0015 |
| `decimate_recording` | 0.4685 |

Naive striding *loses nothing*, which is precisely the problem — the upper half
of the band is still there, superimposed on the lower half. Retaining ~0.5 is
the correct outcome: half the band is gone because it was discarded rather than
folded.

---

## Why FIR and not IIR

`scipy.signal.decimate` defaults to `ftype="iir"`, an order-8 Chebyshev I.

An IIR filter has **frequency-dependent group delay**: it delays 40 Hz by a
different amount than it delays 5 Hz. Here **latency is the signal** — what
the model reads is the evoked response's timing relative to word onset, and a
filter that shifts each band differently smears that profile in a way no later
stage can correct.

A symmetric FIR has **linear phase**, so its only delay is a constant, whole
number of samples, and `resample_poly` compensates it exactly. The ripple and
tap cost are worth paying for that one property.

---

## ⚠️ scipy's default filter is too weak for this data

`resample_poly(x, 1, 2)` designs its own filter, and reading the source it is:

```python
half_len = 10 * 2                               # factor 2
firwin(2 * half_len + 1, 1.0 / 2, window=("kaiser", 5.0))
```

41 taps, and — the part that matters — **`firwin`'s cutoff argument is the
half-amplitude (-6 dB) point**, placed at exactly `1/2` of input Nyquist, which
is 62.5 Hz. The fold frequency itself. So the transition band straddles the
fold and the filter passes half the amplitude of the band it exists to remove.

Measured responses, all at fs=250, factor 2:

| design | taps | gain at 62.5 Hz | worst above it | edge each end |
|---|---|---|---|---|
| **`resample_poly` default** | 41 | **-6.0 dB** | **-6.0 dB** | 80 ms |
| 80 dB, passband to 56.25 Hz | **201** | **-79.5 dB** | **-79.4 dB** | **400 ms** |
| 80 dB, passband to 58 Hz | 279 | -79.4 dB | -79.4 dB | 556 ms |
| 100 dB, passband to 56 Hz | 247 | -98.2 dB | -98.2 dB | 492 ms |

**The 201-tap row is what is implemented.** Its actual response:

```
 10.00 Hz ->    0.00 dB      62.50 Hz ->  -79.53 dB
 40.00 Hz ->    0.00 dB      70.00 Hz ->  -93.05 dB
 56.25 Hz ->   -0.00 dB     100.00 Hz -> -106.25 dB
```

DC gain 1.000000, and a 10 Hz sinusoid survives with max error 2.4e-07 across
the valid core.

### The cost, stated plainly

`PASSBAND_FRACTION = 0.9` ends the passband at 56.25 Hz, so **56.25-62.5 Hz is
deliberately thrown away** rather than aliased. That band is a slice of the
9.97%. This is the only real knob in the module: a sharper filter keeps more of
it and costs taps and edge length. It is a 3-point sweep if it ever matters, and
the entry to change is `PASSBAND_FRACTION`, not the tap count.

### Sizing

Kaiser's standard estimate, with the transition width normalised to input
Nyquist:

```
n_taps  ~  (attenuation_db - 7.95) / (2.285 * pi * transition_norm)
```

For 80 dB across 56.25-62.5 Hz at fs=250: `(80 - 7.95) / (2.285 * pi * 0.05)`
= 200.8, forced odd to **201**. Odd length keeps the kernel symmetric about a
sample, so the compensated delay is a whole number rather than a half-sample
interpolation. The cutoff handed to `firwin` is the **centre** of the transition
band, 59.375 Hz, because that argument is the -6 dB point and not the stopband
edge.

---

## Edge transients

A 201-tap zero-phase filter needs 100 input samples — **400 ms** — of context at
each end. Inside that margin the output is convolved against padding rather than
data.

Against 806 s recordings this is negligible, but it is a **contract, not a
footnote**: `DecimatedRecording.invalid_edge_s` carries it, and the window
cutter must reject any window overlapping either margin. It already has to
reject windows falling off the ends for its own reasons, so this is one more
term in the same test.

---

## ⚠️ Filter the recording whole, never per window

The kernel is 201 samples and a stored window is 200.

Filtering windows independently would therefore give **every** window a
transient spanning essentially its entire length — the edge case would become
the only case. This is why decimation is stage 1, applied to the continuous
recording before anything is cut out of it.

---

## Why this is a function, not a `fit`/`apply` class

`ChannelStandardiser` and `Whitener` are classes because they hold statistics
**derived from the training split**: they must be fitted on train only,
persisted, and reapplied unchanged to the holdout.

The decimator holds no statistics. Its taps are a function of `sample_rate_in`,
`factor` and two constants — nothing about the data enters them. So there is no
leakage to avoid and nothing to persist beyond recording the taps as provenance.

`DecimatedRecording` exists for a narrower reason: **the sample rate must not
travel implicitly.** Word onsets are held in seconds and converted to indices
against a rate. If decimation returned a bare array, the caller would pass a
literal `125.0` downstream, and changing `factor` would silently misplace every
window by a constant multiple — with no error raised anywhere, training happily
to chance. Bundling the rate and the edge margin with the array makes that
mistake impossible to make quietly.

---

## What comes next

The output feeds `ChannelStandardiser.fit`. Fitting happens **after** this stage
so that both transforms are fitted on exactly the data the model will see.
