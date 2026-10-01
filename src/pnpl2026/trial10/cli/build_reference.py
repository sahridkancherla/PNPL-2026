"""Build trial 10's reference cache from the downloaded recordings.

    PYTHONPATH=src python -m pnpl2026.trial10.cli.build_reference \\
        --config configs/base.yaml configs/trial10/trial10.yaml

⚠️ Needs every subject-0 recording on the same filesystem: the train
partition from `pnpl2026.cli.prepare`, plus the validation and test partitions
(`--partition validation --subjects deep`, then `test`). Without those six the
cache builds, but there is nothing to select on or score.

Reads every recording in full: pbs/trial10_build.pbs runs it as a CPU job. A
killed build resumes; recordings already written are reused unless --force.
"""

from __future__ import annotations

import argparse

from ...config import DATA_DIR, REPO_ROOT
from ..config import load_trial10_config
from ..data_processing.build_reference import build_reference


def parse_args():
    """Command-line arguments for the reference build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
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
    cfg = load_trial10_config(*args.config)

    manifest = build_reference(
        data_dir=REPO_ROOT / args.data_dir if args.data_dir else DATA_DIR,
        out_dir=cfg.resolved_cache_dir(),
        force=args.force,
        seed=cfg.seed,
    )

    #pnpl's split, as the reference was trained, selected and scored on it
    recordings = manifest["recordings"]
    for partition in ("train", "validation", "test"):
        chosen = [r for r in recordings if r["partition"] == partition]
        print(
            f"[build] {partition:10s}: {len(chosen):3d} recordings, "
            f"{sum(r['n_words'] for r in chosen):,} words, "
            f"{sum(r['n_in_vocab'] for r in chosen):,} in-vocabulary"
        )

    #the partitions the reference needs; say plainly if any is empty
    present = {r["partition"] for r in recordings}
    for partition in ("validation", "test"):
        if partition not in present:
            print(
                f"[build] WARNING: no {partition} recordings. Download them on a login node:\n"
                f"    uv run python -m pnpl2026.cli.prepare --partition {partition} --subjects deep"
            )

    print(
        f"[build] sensor positions: {'yes' if manifest['has_positions'] else 'NO'}"
        + ("" if manifest["has_positions"] else " -- set spatial: linear in the config")
    )
    for skipped in manifest["skipped"]:
        print(f"[build] skipped {skipped['key']}: {skipped['reason']}")


if __name__ == "__main__":
    main()
