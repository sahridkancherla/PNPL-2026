"""Holdout inference -> submission CSV (deep track).

    uv run python -m pnpl2026.cli.predict --ckpt runs/<name>/best.ckpt

Write this once and do not restructure it later. LibriBrainCompetitionHoldout
owns the canonical row ordering -- the host builds the label CSV by iterating
the *same* loader, so indices align by construction. Never sort, shuffle, or
reorder the outputs of iter_windows(): probabilities must be concatenated in
exactly the order yielded.
"""

from __future__ import annotations

import argparse

from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

from ..config import DATA_DIR, SUBMISSIONS_DIR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or SUBMISSIONS_DIR / "submission_deep.csv"

    holdout = LibriBrainCompetitionHoldout(track="deep", data_path=DATA_DIR)

    # TODO: load LitWordClassifier from args.ckpt, eval mode, no_grad.
    primary = []
    for meg, _meta in holdout.iter_windows(batch_size=args.batch_size):
        # meg: (B, 306, 250) float32 -- model must return (B, 50) probabilities.
        raise NotImplementedError
    # primary_probs = np.concatenate(primary)

    write_submission(
        out,
        indices=holdout.indices,
        primary_probs=None,  # TODO: primary_probs
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
