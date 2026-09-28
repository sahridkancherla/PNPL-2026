"""Build the trial-1 window cache from the downloaded recordings.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.

    PYTHONPATH=src python -m pnpl2026.trial1.cli.build_cache \
        --config configs/base.yaml configs/trial1.yaml

⚠️ Needs the downloaded LibriBrain cache to already exist -- run
`pnpl2026.cli.prepare` first, on a machine with outbound internet. This step
reads all 162 recordings twice to fit the transforms and once more to cut
windows, so it is long and I/O bound, and it writes ~18 GB.
"""

from __future__ import annotations

import argparse
import json

from ...config import DATA_DIR
from ..config import load_trial1_config
from ..data_processing.build_cache import build_cache


def parse_args():
    """Command-line arguments for the cache build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even though a manifest already exists in the cache dir.",
    )
    return parser.parse_args()


def main():
    """Fit the transforms on the training sessions and write every shard."""
    args = parse_args()
    cfg = load_trial1_config(*args.config)
    out_dir = cfg.resolved_cache_dir()

    #rebuilding silently would discard a cache that a finished run's numbers
    #depend on, and it costs hours to recreate
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(
            f"{manifest_path} already exists. Pass --force to rebuild, or point "
            f"cache_dir at a new directory."
        )

    print(f"[cache] reading recordings from {DATA_DIR}")
    print(f"[cache] writing shards to     {out_dir}")
    print(
        f"[cache] rank {cfg.n_components}, bounds {tuple(cfg.window_bounds)} s "
        f"at {cfg.sfreq} Hz"
    )

    manifest = build_cache(
        data_dir=DATA_DIR,
        out_dir=out_dir,
        sample_rate=float(cfg.sfreq),
        n_components=cfg.n_components,
        val_sessions=cfg.val_sessions or None,
        val_fraction=cfg.val_fraction,
        bounds=tuple(cfg.window_bounds),
        seed=cfg.seed,
    )

    #the clipped fraction is the one number that says whether clip_sigma was
    #sane; a large value means the cast to float16 is losing real signal
    print(
        f"\n[cache] {manifest['n_windows']} windows over "
        f"{len(manifest['shards'])} shards"
    )
    print(f"[cache] train sessions {manifest['train_sessions']}")
    print(f"[cache] val sessions   {manifest['val_sessions']}")
    print(f"[cache] worst shard clipped {manifest['max_clipped_fraction']:.2%} of values")
    print(json.dumps({k: v for k, v in manifest.items() if k != "shards"}, indent=2))


if __name__ == "__main__":
    main()
