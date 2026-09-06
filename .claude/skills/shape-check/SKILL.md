---
name: shape-check
description: Verify that every tensor shape in a model or data pipeline is correct and consistent — layout conventions, divisibility constraints, loss target contracts, squeeze/unsqueeze, reduction axes, and silent broadcasting. Use before the first training run of new code, after refactoring a model or dataset, when a shape error appears, or when a model runs but produces nonsense.
---

# Shape check

Shape bugs split into two classes, and the second is what this procedure exists
for:

1. **Loud** — the shapes disagree and something raises. Annoying, cheap, fixed
   in minutes.
2. **Silent** — the shapes are *compatible but wrong*. Broadcasting quietly
   produces a plausible tensor, training runs, loss descends, and the result is
   meaningless. These cost days.

A model that trains to exactly chance is the classic signature of a silent
shape bug. Never attribute a chance-level result to "the signal is hard" until
this procedure has been run. Chance here is `metric_k / n_classes` for BAcc@k
(0.20 at k=10), not 1/50 — check the number against the right floor first.

---

## Step 1 — Establish the layout convention

Before reading any layer, write down the project's canonical layouts and check
every function against them.

For this project:

| tensor | layout | notes |
|---|---|---|
| MEG windows | `(B, C, T)` | `C = n_channels` (306 sensors), `T = (tmax - tmin) * sfreq` |
| conv / norm over time | `(B, C, T)` | `Conv1d`, `GroupNorm`, `BatchNorm1d` |
| attention / LayerNorm | `(B, T, C)` | channels-last |
| pooled features | `(B, D)` | after the temporal reduction |
| logits | `(B, n_classes)` | one word per trial; `n_classes = 50` |
| labels | `(B,)` | int64 word indices |
| class weights | `(n_classes,)` | inverse-frequency, from the train split only |

`C` and `T` both come from `Config`, so read them from the config rather than
assuming 306 and 250 — a change to `tmin`/`tmax`/`sfreq` moves `T` silently.

**Every `.transpose(1, 2)` is a boundary between two conventions.** List them
all and confirm each one sits exactly at a convention change — not one layer
early, not one late. A transpose in the wrong place produces a tensor of the
right *shape* whose axes mean the wrong thing, which is the worst case.

---

## Step 2 — Annotate every function with its shape contract

For each module's `forward`, write the input and output shape as a comment if
absent, then verify by reading — not by assuming the name is accurate.

```python
def forward(self, x):        # (B, C, T) -> (B, D)
```

Check specifically:

- **Does the output shape claimed match what the last operation produces?**
- **Do residual connections add tensors of identical shape?** `x + f(x)` will
  broadcast happily if `f` collapsed an axis to 1.
- **Do concatenations use the right `dim`,** and does the consumer's input width
  match the *sum* of the parts? Recompute the sum by hand.

---

## Step 3 — Check divisibility constraints

These raise loudly, but only for the configuration you happened to run. A
different preset breaks later, in someone else's run.

Enumerate every constraint and confirm it holds **for all configurations**, not
just the default:

- `GroupNorm(g, d)` → `d % g == 0`
- multi-head attention → `d % n_heads == 0`
- rotary embeddings → `(d // n_heads) % 2 == 0` (rotates feature *pairs*)
- grouped/depthwise conv → `in_ch % groups == 0` and `out_ch % groups == 0`
- multi-branch conv splitting width → `d % n_branches == 0`
- pooling/striding → does `T` stay non-zero after every downsample for the
  *shortest* input the pipeline can produce?

If a config knob (width, heads, branches) can be set independently, **resolve
the dependent value in code rather than documenting the constraint**. A helper
that picks the largest valid head count beats an assert that fires on someone
else's preset.

---

## Step 4 — Check loss target contracts

Every loss has a shape convention and they differ. Verify each call site:

| loss | logits | target |
|---|---|---|
| `cross_entropy` (this project) | `(B, K)` | `(B,)` int64 |
| `cross_entropy` (per-frame) | `(B, K, T)` | `(B, T)` int64 |
| `binary_cross_entropy_with_logits` | `(B, F, T)` | `(B, F, T)` float, same shape |
| contrastive / cosine | `(B, D)` normalised | `(B,)` labels |

`cross_entropy(..., weight=w)` needs `w` of shape `(K,)` on the same device as
the logits — a `(B,)` weight is also accepted by broadcasting rules in some
paths and silently weights the wrong axis.

Confirm:

- Class dimension is **axis 1** for `cross_entropy`, never last.
- Targets are `int64` for CE, `float` for BCE.
- Any lookup-table target (e.g. mapping frame classes to a feature matrix)
  produces the layout the loss wants — `table[targets]` on `(B, T)` gives
  `(B, T, F)` and needs a `permute` to `(B, F, T)`. **Verify the permutation
  rather than trusting it.**
- Sliced inputs get their targets sliced identically. If the model consumes a
  sub-window of the cached tensor, the per-frame targets must be sliced by the
  *same* indices.
- Augmentation that shifts inputs shifts targets by the same amount.

---

## Step 5 — Hunt silent broadcasting

The dangerous class. Grep for these patterns and check each by hand:

- **`(N,)` against `(N, 1)`** → produces `(N, N)`. Catastrophic and silent.
  Common in loss weighting and metric code.
- **`(B, 1, T)` against `(B, C, 1)`** → produces `(B, C, T)`. Legitimate for
  gating, a bug if one operand was supposed to be full-rank.
- **`keepdim`** — a reduction without `keepdim=True` followed by an arithmetic
  op is where most broadcasting bugs enter.
- **Indexing that drops an axis** — `x[:, 0]` versus `x[:, 0:1]`.
- **`squeeze()` with no argument** — removes *every* singleton axis, so it
  behaves differently when the batch size happens to be 1. **Always pass an
  explicit dim.** Same for `unsqueeze`: state the dim and say why.

---

## Step 6 — Check reduction axes

For every `mean`, `sum`, `std`, `max`, `argmax`, `softmax`, `topk`,
`logsumexp`:

- Is `dim` explicit? An implicit reduction over *all* elements is almost never
  what was meant.
- Is it reducing the axis intended — time, channels, or classes? Reducing over
  channels where time was meant yields the right shape and wrong content.
- For normalisation, does the reduction preserve the property it is supposed
  to? (Normalising per channel destroys relative channel scaling; normalising
  per trial preserves it. These have identical shapes.)

---

## Step 7 — Check alignment, not just shape

Two arrays can have matching shapes and still be misaligned — the error is in
*correspondence*, not dimension. Shape checks will not catch it, so check it
explicitly:

- Features built per-group and concatenated must be in the **same order** as
  the index array they describe.
- Masking by *index value* is wrong when items repeat; mask by **position**.
- Slicing a source array by `[i:i+step]` while writing to a destination of
  different length silently truncates or raises at the final block — clamp the
  end (`j = min(i+step, n)`).
- Predictions and labels gathered in separate loops must use identical
  iteration order and batching.

---

## Step 8 — Run the harness with distinctive prime dimensions

Static reading misses things. Instantiate the model and push a tensor through.

**Use small, distinct primes for every axis** — `B=3, C=5, T=7`, not
`B=32, C=64, T=64`. If two axes share a size, a transposed or broadcast error
produces a valid shape and the test passes. Distinct primes make every such bug
raise.

```python
import torch
from pnpl2026.models import REGISTRY, build_model

# Distinct primes, and every model in the registry -- not just cnn_baseline.
for name in REGISTRY:
    model = build_model(name, n_channels=5, n_classes=7)
    # T=53 is prime and survives cnn_baseline's two MaxPool1d(4) stages
    # (53 -> 13 -> 3). A smaller prime like 11 pools to length 0 and raises.
    logits = model(torch.randn(3, 5, 53))
    print(name, tuple(logits.shape))    # expect (3, 7)
```

Verify each output against the contract by hand. Then:

- **Backward pass** — `loss.backward()`, then assert no parameter has
  `grad is None`. A `None` gradient means that branch is disconnected from the
  loss, which shape checks alone will not reveal.
- **Batch size 1** — catches `squeeze()` bugs that hide at larger batches.
- **Shortest and longest real input length** — catches downsampling that
  reaches zero and positional tables that are too small.
- **Loss actually computes** on the harness tensors, with real target dtypes.

---

## Step 9 — Leave assertions behind

Convert what you verified into cheap permanent checks at module boundaries:

```python
B, C, T = x.shape
assert C == self.n_in, f"expected {self.n_in} channels, got {C}"
```

Assert on the *semantically meaningful* axes — channel counts, class counts,
sequence lengths that must match a companion array. Skip asserts inside hot
loops; boundaries are enough.

---

## Output

Report:

1. **Confirmed contracts** — one line per module, `input -> output`.
2. **Defects found**, each with file:line, why it is wrong, and whether it is
   loud or silent. Silent ones first.
3. **Unverifiable claims** — anything that could not be checked without running
   the real data, stated as such rather than assumed correct.

Do not report a pipeline as shape-correct if only the default configuration was
exercised. Say which configurations were checked.
