"""Trial 9: trial 7's Variant D, with a 0.5 Hz high-pass on every window.

A per-word encoder, then a transformer across the words of a sentence, trained
on windows and sentences shaped like the holdout's. Self-contained by design;
nothing here imports from another trial. It reads trial 7's continuous cache,
which trial 7's builder writes. See `info/CONTEXT.md` for the router and
`info/highpass.md` for the design and the run queue.
"""
