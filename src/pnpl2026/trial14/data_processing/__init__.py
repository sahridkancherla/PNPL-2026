"""A copy of trial 10's cache builder, for its constants and functions only.

Trial 14 builds no cache: it reads trial 10's (`pnpl2026.trial10.cli.build_reference`).
`cli.predict` resamples the holdout with this module's constants, so the two
can never disagree on the rate.
"""
