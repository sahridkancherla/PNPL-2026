"""Trial 10: Variant F -- the d'Ascoli reference, reproduced on LibriBrain100.

The competition paper reports 73.23% BAcc@10 for its adaptation of d'Ascoli
et al. (2025), on LibriBrain100's standard TEST partition. This trial rebuilds
that model from the paper and the brainmagick code it descends from, trains it
the same way, and scores it the same way -- then also under holdout
conditions. Self-contained; nothing here imports from another trial. See
`info/CONTEXT.md` for the router and `info/variant-f.md` for the design.
"""
