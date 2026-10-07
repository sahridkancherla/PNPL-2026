"""Fixed scaling statistics from the training recordings, for compliant prediction.

    PYTHONPATH=src python -m pnpl2026.trial17.cli.training_scaler

The holdout principle (CLAUDE.md): nothing computed across the holdout as a
whole. Until 2026-10-07 every prediction scaled the holdout with a median and
IQR pooled over all of a holdout source's epochs. The reference cache scaled
each training recording by its own per-channel median and IQR
(`scaler_<key>.npz`, fitted at 50 Hz); this takes, per channel, the median of
those medians and the median of those IQRs over the TRAINING recordings, and
writes

    data/trial17_training_scaler.npz     median [306], iqr [306]

which `cli.predict --scaling training` applies to every holdout item, frozen.

It also prints how much the training recordings differ from that fixed
scaler: each recording's IQR as a ratio of the fixed one, and its median's
offset in fixed IQRs. Near 1 and near 0 mean a fixed scaler costs little even
for models trained on per-recording scaling (G1, G2, F1, w1024); I1/I2
standardise every window themselves and barely notice.

Light: reads ~156 small files, seconds on a login node.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ...config import REPO_ROOT
from ..config import Trial17Config


def parse_args():
    """Command-line arguments for the fixed scaler."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=Trial17Config.cache_dir, help="repo-relative reference cache")
    parser.add_argument("--partition", default="train", help="pnpl partition the statistics come from")
    parser.add_argument("--out", default="data/trial17_training_scaler.npz", help="repo-relative output")
    return parser.parse_args()


def percentiles(values):
    """5th, 50th and 95th percentiles as one line.

    ARGS:
        values (np.ndarray)

    RETURNS:
        str
    """
    low, middle, high = np.percentile(values, [5, 50, 95])
    return f"{low:.3f} / {middle:.3f} / {high:.3f}"


def main():
    """Read every training recording's scaler, take medians, report, write."""
    args = parse_args()
    cache_dir = REPO_ROOT / args.cache_dir
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))

    #one median and one IQR per channel per recording, training partition only
    medians, iqrs = [], []
    for record in manifest["recordings"]:
        if record["partition"] != args.partition:
            continue
        with np.load(cache_dir / f"scaler_{record['key']}.npz") as stored:
            medians.append(stored["median"].astype(np.float64))
            iqrs.append(stored["iqr"].astype(np.float64))
    if not medians:
        raise SystemExit(f"No {args.partition} recordings in {cache_dir}.")

    #[n_recordings, 306]; axis 0 is the recording axis, a median per channel
    medians, iqrs = np.stack(medians, axis=0), np.stack(iqrs, axis=0)
    fixed_median = np.median(medians, axis=0)
    fixed_iqr = np.median(iqrs, axis=0)

    #how far each recording's own scaler sits from the fixed one
    ratio = iqrs / fixed_iqr[None, :]
    offset = np.abs(medians - fixed_median[None, :]) / fixed_iqr[None, :]
    print(f"[training_scaler] {len(medians)} {args.partition} recordings, {medians.shape[1]} channels")
    print(f"[training_scaler] recording IQR / fixed IQR, 5/50/95%:            {percentiles(ratio)}")
    print(f"[training_scaler] |recording median - fixed| / fixed IQR, 5/50/95%: {percentiles(offset)}")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, median=fixed_median, iqr=fixed_iqr, n_recordings=np.int64(len(medians)),
             partition=np.array(args.partition))
    print(f"[training_scaler] wrote {out_path}")


if __name__ == "__main__":
    main()
