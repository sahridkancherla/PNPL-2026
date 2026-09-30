"""Trial 7: Variant D, matched to the holdout.

A per-word encoder, then a transformer across the words of a sentence, trained
on windows and sentences shaped like the holdout's. Self-contained by design;
nothing here imports from another trial. See `info/CONTEXT.md` for the router
and `info/variant-d.md` for the design as built.
"""
