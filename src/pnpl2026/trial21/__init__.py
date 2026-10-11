"""Trial 21: train on the holdout's own session and format.

Two parts, from the 2026-10-10 architecture review
(`handoff-docs/architecture-review-2026-10-10.md`):

    A   transductive self-distillation (`cli/distil.py`): a trained member of
        either family is fine-tuned briefly on its own training items plus
        the holdout's epochs, where the target is the best ensemble's
        hub-corrected distribution. The encoder and its BatchNorm adapt to
        the holdout session; the sentence stage sees the holdout's sparse
        list-word tokens with consistent targets. Nothing fitted on labels
    B   the hybrid (`stem: dascoli`): the continuous family with the per-word
        family's encoder as its stem, warm-started from a trained G2

A copy of trial 20's code (2026-10-11), plus trial 18's per-word family under
`perword/` so one distillation loop and one `cli.predict` drive both
families. Nothing here imports from another trial. See `info/CONTEXT.md`.
"""
