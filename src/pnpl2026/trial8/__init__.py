"""Trial 8: trial 7's Variant D, with an attention reach per sentence block.

A per-word encoder, then a transformer across the words of a sentence, trained
on windows and sentences shaped like the holdout's. Self-contained by design;
nothing here imports from another trial. It reads trial 7's continuous cache,
which trial 7's builder writes. See `info/CONTEXT.md` for the router and
`info/reach.md` for the design and the run queue.
"""
