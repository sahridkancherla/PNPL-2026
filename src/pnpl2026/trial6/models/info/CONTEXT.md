# trial6/models/ — the network

Trial 5's dilated conv trunk and flat transformer, with Variant C's switches.
Every switch defaults to trial 5's behaviour, so C0 is trial 5's network.

## Router

| where | what |
|---|---|
| `../net.py` | `TrialSixNet` — the two streams, their concatenation, the rest of trial 5 |
| `../envelope.py` | stream B: `BandEnvelope`, `FittedProjection`, `EnvelopeStream`, and `fit_envelope_projections` |
| `../frontend.py` | fixed front end and the three training-only augmentations, `GaussianNoise` new |
| `../trunk.py` | `DilatedConvTrunk` — unchanged, with `ReceptiveFieldError` |
| `../transformer.py` | `FlatTransformer` — fused attention; depth 0 is valid |
| `../heads.py` | pooling, `SemanticHead`, `ClosedSetHead`, `SessionAdversary` — unchanged |
| `../norms.py` | `BiasNorm`, `SwooshR/L`, `DropPath` — unchanged |
| `../__init__.py` | `REGISTRY` and `build_model` |

Stages and shapes, per run, are in `../../info/pipeline.md`.

## Three things that are easy to get wrong here

**Stream B reads the raw window.** Fed stream A's normalised one, every
envelope would be divided by broadband amplitude, and C4/C5 would stop asking
about their band alone.

**The envelope PCA is fitted, not learned.** `cli.train` fits it on training
windows before step one; a checkpoint carries it. An unfitted projection
raises rather than passing zeros, which would train to exactly chance.

**Window length is checked.** Every layer accepts any length, so `forward`
refuses one that is not the configured `T`: a 1 s holdout window fed to W05's
0.5 s model would otherwise run and score the wrong span.
