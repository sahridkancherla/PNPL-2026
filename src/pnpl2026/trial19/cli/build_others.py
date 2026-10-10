"""Build the other subjects' cache: LibriBrain100's broad-track recordings, as subject 0's.

    PYTHONPATH=src python -m pnpl2026.trial19.cli.build_others

Every recording of every subject but 0 that pnpl's manifest lists -- the 32
broad-track subjects, Sherlock1 sessions 11 and 12, ~44 min each -- through
the same steps as trial 10's reference cache (`data_processing/build_reference.py`,
`build_others`): 250 -> 50 Hz with an anti-alias filter, a robust scaler per
recording, a word table. Keys carry `sub-<id>`; subject 0's channel order is
required and its sensor positions are copied, so the merger reads every
subject's sensors alike.

A run trains on this cache with `others_cache_dir: data/trial19_others`. Its
local validation is then not clean -- these people heard subject 0's
validation and test stories -- so such a run trains a fixed number of epochs
(`cli/train.py`).

⚠️ The recordings must be downloaded first, on a LOGIN node (the default
`--subjects all` brings the other subjects):

    uv run python -m src.pnpl2026.cli.prepare --partition validation
    uv run python -m src.pnpl2026.cli.prepare --partition test

⚠️ Heavy: reads every h5 in full. pbs/trial19_build_others.pbs runs it as a CPU
job. A killed build resumes: recordings already written are reused.
"""

from __future__ import annotations

import argparse

from ...config import DATA_DIR, REPO_ROOT
from ..data_processing.build_reference import build_others


def parse_args():
    """Command-line arguments for the others' build."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root holding the task folders")
    parser.add_argument("--reference-dir", default="data/trial10_reference", help="subject 0's cache, repo-relative")
    parser.add_argument("--out-dir", default="data/trial19_others", help="repo-relative output")
    parser.add_argument("--force", action="store_true", help="rebuild recordings already written")
    return parser.parse_args()


def main():
    """Build the cache, then summarise it."""
    args = parse_args()
    manifest = build_others(args.data_dir, REPO_ROOT / args.out_dir, REPO_ROOT / args.reference_dir, force=args.force)

    #how much MEG the other subjects add, against subject 0's ~80 h
    records = manifest["recordings"]
    hours = sum(record["n_times"] for record in records) / manifest["sample_rate"] / 3600
    words = sum(record["n_in_vocab"] for record in records)
    print(f"[others] {len(records)} recordings of {len(manifest['subjects'])} subjects: {hours:.1f} h, "
          f"{words:,} in-vocabulary words; {len(manifest['skipped'])} skipped")
    if not records:
        raise SystemExit("[others] nothing built: download the other subjects first (see this module's docstring).")


if __name__ == "__main__":
    main()
