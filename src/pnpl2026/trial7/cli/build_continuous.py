"""Build trial 7's continuous cache from the downloaded recordings.

    PYTHONPATH=src python -m pnpl2026.trial7.cli.build_continuous \\
        --config configs/base.yaml configs/trial7/trial7.yaml

⚠️ Needs, on the same filesystem:
    data/<task>/derivatives/...   the raw download, from `pnpl2026.cli.prepare`
    data/trial1_cache/            for transforms.npz and manifest.json

Reads every recording in full, so it runs as a CPU batch job
(pbs/trial7_build.pbs), never on a login node. A killed build resumes: pairs
already written are reused unless --force is passed.
"""

from __future__ import annotations

import argparse

from ...config import DATA_DIR, REPO_ROOT
from ..config import load_trial7_config
from ..data_processing.build_continuous import build_continuous


def parse_args():
    """Command-line arguments for the continuous build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument(
        "--trial1-cache",
        default="data/trial1_cache",
        help="trial 1's cache, repo-relative, for its transforms and split",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="the raw download, repo-relative; defaults to the repo's data/",
    )
    parser.add_argument("--force", action="store_true", help="rebuild existing recordings")
    return parser.parse_args()


def main():
    """Build the cache and summarise what it holds."""
    args = parse_args()
    cfg = load_trial7_config(*args.config)

    manifest = build_continuous(
        data_dir=REPO_ROOT / args.data_dir if args.data_dir else DATA_DIR,
        trial1_cache=REPO_ROOT / args.trial1_cache,
        out_dir=cfg.resolved_continuous_dir(),
        sample_rate=float(cfg.sfreq),
        force=args.force,
    )

    #the totals are what to compare against trial 1's 312,370 windows
    recordings = manifest["recordings"]
    for split in ("train", "val"):
        chosen = [r for r in recordings if r["split"] == split]
        print(
            f"[build] {split}: {len(chosen)} recordings, "
            f"{sum(r['n_in_vocab'] for r in chosen):,} in-vocabulary words, "
            f"{sum(r['n_sentences'] for r in chosen):,} sentences"
        )

    #say plainly what is missing, since it moves the validation set
    if manifest["absent_from_trial1"]:
        print(
            f"[build] {len(manifest['absent_from_trial1'])} of trial 1's recordings are "
            f"not in this download: {manifest['absent_from_trial1']}"
        )
    for skipped in manifest["skipped"]:
        print(f"[build] skipped {skipped['key']}: {skipped['reason']}")


if __name__ == "__main__":
    main()
