"""Build per-sample phoneme labels for the phoneme head: one int8 per cache sample.

    PYTHONPATH=src python -m pnpl2026.trial20.cli.build_phoneme_labels

For every subject-0 recording in the reference cache, the event table's
phoneme rows (`timeds` onset, `duration`, `segment` through pnpl's
TIMIT -> ARPAbet map, stress digits stripped) are painted onto the cache's
sample grid: class 1-39 for the ARPAbet symbols in pnpl's order, 0 for
silence and for samples no phoneme covers. Written as
`data/trial20_phonemes/phon_<key>.npy` [n_times] int8, plus a manifest.

The head reads these at the stem's steps, shifted by the brain's lag
(`models/continuous.py`, `step_label_indices`). Light: event tables only, no
h5; a minute or two on a login node.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ...config import DATA_DIR, REPO_ROOT
from ..data.dataset import load_manifest
from ..data_processing.build_reference import (
    EVENTS_KIND_COLUMN,
    EVENTS_ONSET_COLUMN,
    EVENTS_SEGMENT_COLUMN,
    discover_recordings,
)
from .build_phoneme_embeddings import PHONEME_KIND, phoneme_symbol

#the events' duration column, seconds
EVENTS_DURATION_COLUMN = "duration"

#class 0: silence, and every sample no phoneme covers
SILENCE_CLASS = 0


def parse_args():
    """Command-line arguments for the label build."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root holding the task folders")
    parser.add_argument("--cache-dir", default="data/trial10_reference", help="repo-relative reference cache")
    parser.add_argument("--out-dir", default="data/trial20_phonemes", help="repo-relative output")
    return parser.parse_args()


def phoneme_classes():
    """The class list: silence first, then pnpl's ARPAbet symbols.

    RETURNS:
        list of str, index = class
    """
    from pnpl.datasets.constants import ARPABET

    return ["SIL"] + list(ARPABET)


def paint_labels(events_path, n_times, sample_rate, class_of, mapping):
    """Every sample's phoneme class for one recording.

    ARGS:
        events_path (Path): the events TSV
        n_times (int): the cache's samples for this recording
        sample_rate (float): the cache's rate
        class_of (dict): symbol -> class
        mapping (dict): pnpl's TIMIT -> ARPAbet table

    RETURNS:
        [n_times] int8, and the share of samples covered by a phoneme
    """
    events = pd.read_csv(events_path, sep="\t")
    rows = events[events[EVENTS_KIND_COLUMN] == PHONEME_KIND]
    labels = np.full(n_times, SILENCE_CLASS, dtype=np.int8)
    covered = 0
    for onset, duration, segment in zip(rows[EVENTS_ONSET_COLUMN], rows[EVENTS_DURATION_COLUMN], rows[EVENTS_SEGMENT_COLUMN]):
        symbol = phoneme_symbol(segment, mapping)
        if symbol is None or symbol not in class_of or not np.isfinite(onset) or not np.isfinite(duration):
            continue

        #[start, stop) on the sample grid, rounded as the cache's onsets are
        start = int(round(float(onset) * sample_rate))
        stop = int(round((float(onset) + float(duration)) * sample_rate))
        start, stop = max(start, 0), min(max(stop, start + 1), n_times)
        if start >= n_times:
            continue
        labels[start:stop] = class_of[symbol]
        covered += stop - start
    return labels, covered / max(n_times, 1)


def main():
    """Paint every recording's labels and write the manifest."""
    args = parse_args()
    from pnpl.datasets.constants import TIMIT_TO_ARPABET

    classes = phoneme_classes()
    class_of = {symbol: index for index, symbol in enumerate(classes)}
    cache_dir = REPO_ROOT / args.cache_dir
    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache_dir)
    sample_rate = float(manifest["sample_rate"])

    #the event table of each cached recording, by key
    events_of = {found["key"]: found["events_path"] for found in discover_recordings(args.data_dir)}
    written, shares = [], []
    for record in manifest["recordings"]:
        events_path = events_of.get(record["key"])
        if events_path is None:
            print(f"[phonemes] {record['key']}: no event table under {args.data_dir}; skipped")
            continue
        labels, share = paint_labels(events_path, int(record["n_times"]), sample_rate, class_of, TIMIT_TO_ARPABET)
        np.save(out_dir / f"phon_{record['key']}.npy", labels)
        written.append(record["key"])
        shares.append(share)

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump({"sample_rate": sample_rate, "classes": classes, "recordings": written}, handle, indent=2)
    print(f"[phonemes] {len(written)} recordings at {sample_rate:g} Hz, {len(classes)} classes; "
          f"phonemes cover {100 * np.mean(shares):.1f}% of samples (min {100 * min(shares):.1f}%) -> {out_dir}")
    if not written:
        raise SystemExit("[phonemes] nothing written: is the data root right?")

    #⚠️ speech fills most of a recording; almost nothing covered means the
    #phoneme rows' timing columns were not read
    if np.mean(shares) < 0.05:
        raise SystemExit(f"[phonemes] phonemes cover only {100 * np.mean(shares):.1f}% of samples: "
                         f"check the phoneme rows' {EVENTS_ONSET_COLUMN} and {EVENTS_DURATION_COLUMN} columns.")


if __name__ == "__main__":
    main()
