"""Trial 23: the 13 Oct research note -- cross-subject information without the timing echo.

The note (`handoff-docs/cross-subject-note-2026-10-13.md`) explains why trial
22's cross-subject votes lost (they re-voted the sentence timing every file
shares) and ranks what could still add two points:

    0   a labelled testbed: subject 0's Sherlock validation/test chapters,
        heard by 12 cached broad subjects too (`cli/testbed_check.py`,
        `cli/testbed.py`), which gates every cross-subject method
    P1  a timing-residualized, cross-fitted 39-subject graph, propagated
        from the mix's posteriors (`crossfeatures.py`, `cli/cross_graph.py`)
    P3  cross-fitted LDA prototypes per subject; P8a their likelihood product
    P6  a boundary / short-sentence specialist (`cli/boundary.py`,
        `cli/finetune.py`, boundary features in the per-word network)
    P7  lone words: more specialist seeds, onset and sensor-dropout TTA
    P2  a row-identification encoder (`cli/row_encoder.py`)
    P5  a skip-aware sequence prior by forward-backward (`cli/sequence_prior.py`)
    P4  MEG-XL as a frozen feature extractor (`cli/megxl_features.py`)
    P8b session re-centering (`cli/recentre.py`)

A copy of trial 22's code. Nothing here imports from another trial. See
`info/CONTEXT.md`.
"""
