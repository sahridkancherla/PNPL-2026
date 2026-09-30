# trial8/models/ — the network

Trial 7's network, unchanged except that each sentence block has its own
attention reach.

## Router

| where | what |
|---|---|
| `../transformer.py` | `FlatTransformer(reach=...)`, `attention_mask(..., reach)`, `resolve_reach` — **the one change** |
| `../net.py` | `TrialEightNet` — passes `sentence_reach` to the sentence transformer |
| `../encoder.py`, `../frontend.py`, `../heads.py` | unchanged from trial 7 |
| `../trunk.py`, `../norms.py` | unchanged from trial 6 |
| `../__init__.py` | `REGISTRY` (`trial8_net`) and `build_model` |

Shapes are trial 7's: `../../../trial7/info/pipeline.md`.

## Easy to get wrong here

**Reach is a mask, not a weight.** Every schedule has the same 4,442,334
parameters; only which words attend to which changes.

**A padding query still sees itself.** The band mask is ANDed with the key
padding mask, then the diagonal is restored, or padding rows turn NaN.

**Reach compounds.** Each block at reach 1 extends a word's view by one word
either side, so `[1, 1, 1, 1]` reaches at most 4 — the ends of a 9-word
sentence never meet. The frame transformer ignores reach (default: full).

**The mask token is dead weight at `word_mask_p: 0`.** It stays so the
parameter count matches trial 7 exactly; it receives no gradient.
