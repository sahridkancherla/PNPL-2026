# trial7/models/ — the network

A per-word encoder — Variant B's front end and trunk over a masked, variable
window — then a transformer across the words of a sentence.

## Router

| where | what |
|---|---|
| `../encoder.py` | `PerWordEncoder` — one word's window to one vector, padding masked throughout |
| `../net.py` | `TrialSevenNet` — the grid of words, masking, D8's per-word masking, both heads |
| `../transformer.py` | `FlatTransformer` with `attention_mask`: key padding and causal |
| `../frontend.py` | fixed front end; `RobustInstanceNorm` takes a validity mask |
| `../heads.py` | masked `AttentionPool`, `ClosedSetHead`, a one-layer `SemanticHead` |
| `../trunk.py`, `../norms.py` | unchanged from trial 6 |
| `../__init__.py` | `REGISTRY` and `build_model` |

Stages and shapes are in `../../info/pipeline.md`.

## Three things that are easy to get wrong here

**Padding must stay zero, and statistics must skip it.** A median over a
half-padded window is dragged towards zero. The norm uses `nanmedian` over
real samples; everything after re-zeroes the padding; the pool gives padding
frames no weight.

**A query with no visible key is NaN.** `attention_mask` always lets a position
see itself, so padding rows stay finite. Their outputs are never read.

**A masked word is not scored.** Training replaces 15% of words with the mask
token so the others stop leaning on any one neighbour; scoring the masked ones
would train prediction from context alone. D8 is the run that does score them.
