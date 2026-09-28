# trial5/models/ — the network

A dilated convolution trunk feeding a **flat** transformer. Channels-first up
to the trunk, channels-last from the transformer on; `net.py` owns the one
transpose between them.

## Router

| where | what |
|---|---|
| `../frontend.py` | `RobustInstanceNorm`, `BaselineCorrect`, `FixedLowpassDecimate` — fixed, unlearned, applied identically to train and holdout |
| `../trunk.py` | `DilatedConvTrunk` — the tokeniser, and `ReceptiveFieldError` |
| `../transformer.py` | `FlatTransformer` — rotary pre-norm blocks. **79% of the parameters** |
| `../heads.py` | pooling, `SemanticHead`, `ClosedSetHead`, `SessionAdversary` |
| `../norms.py` | `BiasNorm`, `SwooshR/L`, `DropPath` — only `DropPath` is on by default |
| `../net.py` | `TrialFiveNet` — assembles the above |
| `../__init__.py` | `REGISTRY` and `build_model` |

## The stack, with shapes

```
[B,  73, 250]   whitened components, 1 s at 250 Hz
[B,  73, 250]   RobustInstanceNorm  -> also returns log(sigma) per channel
[B,  73, 250]   BaselineCorrect     0.0-0.1 s, NOT the reference's 0-0.5 s
[B,  73,  50]   FixedLowpassDecimate  20 Hz, / 5  -> 50 Hz
[B, 270,  50]   spatial      VARIANT B: 1x1 off the PCA components, + GELU
[B, 512,  50]   initial_linear
[B, 256,  50]   DilatedConvTrunk   5 layers, dilations 1,2,4,8,1, reach 33
      transpose(1, 2)  <- the only one
[B,  50, 256]   FlatTransformer    depth 8, heads 8, rotary
[B, 256]        AttentionPool
                -> logits [B, 50] · embedding [B, 1024] · session [B, n_sess]
```

## Three things that are easy to get wrong here

**The trunk must not out-reach the window.** An unreset ladder at depth 5 gives
a reach of 63 frames against a 50-frame window, so the last layer convolves
mostly with padding and the division of labour with the transformer collapses.
`DilatedConvTrunk` raises `ReceptiveFieldError` rather than letting that train.
`dilation_period: 4` is the fix, and it is the default.

**Baseline correction cannot use pre-onset samples.** The cache has them; the
holdout does not — `iter_windows()` serves pre-cut `[0, 1]` s windows. See the
docstring in `frontend.py`.

**`spatial` and `initial_linear` each carry a GELU.** Two linear maps in a row
collapse into one, so without the activation these are a single 73 -> 512 map
wearing two names.
