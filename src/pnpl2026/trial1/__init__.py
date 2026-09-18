"""Trial 1 -- whitened 73-component pipeline with a multi-branch residual net.

A *trial* is one complete self-contained attempt: its own preprocessing, its
own model, its own configs. Trials do not import from one another, and code is
copied between them rather than shared, so that a finished trial stays
reproducible when the next one changes its mind about the window, the rank or
the sample rate.

Shared, trial-independent machinery -- the metric, the competition label
space, the run config -- stays in `pnpl2026` above this package.
"""
