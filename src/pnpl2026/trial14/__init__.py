"""Trial 14: Variant H -- G2 plus a trained 50-word head, judged on holdout-shaped chunks.

Trial 13's G2 (isolated-word training, 63.41 on the leaderboard as a 3-seed
ensemble) scores words by retrieval: cosine against t5-large's embeddings of
the 50 candidates, trained only to match t5. Trial 14 adds a 50-way head on
the same vector, trained with logit-adjusted cross-entropy for balanced
accuracy, and blends the two. It starts every run from a G2 checkpoint.

It also replaces trial 13's fragment protocol (128 test words) with the
CHUNKED protocol: every sentence cut into holdout-shaped chunks, every scored
word scored. A copy of trial 13's code; it reads trial 10's cache and t5
table and loads trial 10/12/13 checkpoints. Nothing here imports from another
trial. See `info/CONTEXT.md`.
"""
