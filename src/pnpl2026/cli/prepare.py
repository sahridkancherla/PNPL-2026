"""Download / cache LibriBrain data.

Run on the HPC *login node* -- compute nodes have no outbound internet.

    uv run python -m pnpl2026.cli.prepare                      # the train partition
    uv run python -m pnpl2026.cli.prepare --partition validation --subjects deep
    uv run python -m pnpl2026.cli.prepare --partition test --subjects deep

pnpl's partitions for subject 0: train is 156 recordings; validation and test
are three each (Sherlock1 ses-11/12 run-2, MOCHATIMIT ses-2/3, TheMoth
ses-29/30). Trial 10 scores on the test partition, as the competition paper's
reference does. `--subjects deep` keeps the validation and test downloads to
subject 0 -- without it they include all 32 other subjects' Sherlock1 sessions.
"""

from __future__ import annotations

import argparse

from pnpl.datasets import LibriBrain100
from pnpl.tasks import WordClassification

from ..config import DATA_DIR, Config


def parse_args():
    """Command-line arguments for the download."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        default="train",
        choices=("train", "validation", "test"),
        help="pnpl's standard partition to download",
    )
    parser.add_argument(
        "--subjects",
        default="all",
        help="pnpl subject selector: 'all', 'deep' (subject 0) or an id",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LibriBrain100(
        data_path=DATA_DIR,
        task=WordClassification(tmin=cfg.tmin, tmax=cfg.tmax),
        partition=args.partition,
        subjects=args.subjects,
    )
    print(f"data ready under {DATA_DIR} ({args.partition}, subjects={args.subjects})")


if __name__ == "__main__":
    main()
