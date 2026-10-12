"""Build a reference cache at a chosen rate: trial 10's build, with `--sfreq`.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.build_reference --sfreq 100 \\
        --out-dir data/trial20_reference_100hz

Trial 10's cache is 50 Hz (`data/trial10_reference`); trial 19's model reads
any rate the config names, and the 10 Oct note's change 4 asks for 100 Hz.
Everything else is trial 10's build (`data_processing/build_reference.py`):
250 Hz h5 -> polyphase resample -> a robust scaler per recording -> word
tables, positions, the training PCA, a manifest.

⚠️ Heavy: reads every h5 in full. pbs/trial22_build.pbs runs it as a CPU job.
A killed build resumes: recordings already written are reused.
"""

from __future__ import annotations

import argparse

from ...config import DATA_DIR, REPO_ROOT
from ..data_processing.build_reference import build_reference


def parse_args():
    """Command-line arguments for the build."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root holding the task folders")
    parser.add_argument("--out-dir", default="data/trial20_reference_100hz", help="repo-relative output")
    parser.add_argument("--sfreq", type=float, default=100.0, help="the cache's rate; 250 must be a multiple-friendly ratio")
    parser.add_argument("--force", action="store_true", help="rebuild recordings already written")
    return parser.parse_args()


def main():
    """Build the cache, then summarise it."""
    args = parse_args()
    manifest = build_reference(args.data_dir, REPO_ROOT / args.out_dir, force=args.force, sfreq=args.sfreq)
    records = manifest["recordings"]
    hours = sum(record["n_times"] for record in records) / manifest["sample_rate"] / 3600
    print(f"[build] {len(records)} recordings at {manifest['sample_rate']:g} Hz, {hours:.1f} h -> {REPO_ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
