"""A copy of trial 10's cache builder, for its constants and functions only.

Trial 15 builds no cache: it reads trial 10's (`pnpl2026.trial10.cli.build_reference`).
`cli.wordpairs` reads the same cache's `words_<key>.npz` tables.
`cli.predict` resamples the holdout with this module's constants, so the two
can never disagree on the rate.
"""
