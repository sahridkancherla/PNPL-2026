"""
built with AI. unchecked
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ...data.splits import session_split
from ...data.vocab import is_target, to_id
from .preprocessing import ChannelStandardiser, Whitener
from .storage import clip_and_cast
from .windowing import WINDOW_BOUNDS, cut_windows

# Verified against real recordings on 2026-09-14. The h5 also carries a `times`
# dataset and the events table several other time columns.
H5_DATASET_KEY = "data"

# ⚠️ `timeds`, NOT `timemeg`. The events table carries both: `timemeg` is the
# original 1 kHz timeline and lands off the 250 Hz sample grid 6% of the time,
# which is a random sub-sample shift per trial. `timeds` is the downsampled
# timeline and matches the h5 exactly. There is no `onset` column.
EVENTS_ONSET_COLUMN = "timeds"
EVENTS_KIND_COLUMN = "kind"
EVENTS_SEGMENT_COLUMN = "segment"

#the events row type that marks a spoken word, per CLAUDE.md section 2
WORD_KIND = "word"

#one timepoint in this many is held for the standardiser fit
STANDARDISER_SUBSAMPLE = 50


@dataclass
class RunPaths:
    """Where one recording's two files live, and how it is identified.

    ATTRS:
        meg_path (Path): the continuous [306, n_times] h5
        events_path (Path): the events TSV beside it
        session (str): session id, the unit the train/val split works on
        task (str): task name, e.g. "Sherlock1"
        run (str): run id within the session
    """

    meg_path: Path
    events_path: Path
    session: str
    task: str
    run: str

    @property
    def key(self):
        """A filename-safe identifier, unique across the 162 recordings."""
        return f"ses-{self.session}_task-{self.task}_run-{self.run}"


def discover_runs(data_dir):
    """Find every recording that has both an h5 and a matching events TSV.

    ARGS:
        data_dir (Path): the cache root, from `config.DATA_DIR`

    RETURNS:
        runs : list of RunPaths, sorted by session then task then run
    """
    #the serialised folder carries the processing chain in each filename
    meg_paths = sorted(Path(data_dir).glob("*/derivatives/serialised/*_meg.h5"))
    if not meg_paths:
        raise FileNotFoundError(f"No *_meg.h5 found under {data_dir}")

    runs = []
    for meg_path in meg_paths:
        #split "sub-0_ses-10_task-Sherlock1_run-1_proc-...._meg" into its fields
        parts = dict(
            piece.split("-", 1)
            for piece in meg_path.stem.split("_")
            if "-" in piece
        )

        #the events file drops the proc chain, so rebuild its name from the key
        events_dir = meg_path.parent.parent / "events"
        stem = f"sub-{parts['sub']}_ses-{parts['ses']}_task-{parts['task']}"
        events_path = events_dir / f"{stem}_run-{parts['run']}_events.tsv"

        #a recording with no event table cannot produce a single labelled window
        if not events_path.exists():
            continue

        runs.append(
            RunPaths(
                meg_path=meg_path,
                events_path=events_path,
                session=parts["ses"],
                task=parts["task"],
                run=parts["run"],
            )
        )

    if not runs:
        raise FileNotFoundError(f"No recording under {data_dir} had an events table")
    return runs


def load_recording(meg_path):
    """Read one continuous recording into memory.

    ARGS:
        meg_path (Path): the h5 holding a single [306, n_times] array

    RETURNS:
        recording : [n_channels, n_times] float32, continuous
    """
    with h5py.File(meg_path, "r") as handle:
        #the key is an assumption, so say what was actually there when it misses
        if H5_DATASET_KEY not in handle:
            raise KeyError(
                f"{meg_path.name} has no '{H5_DATASET_KEY}' dataset; "
                f"found {list(handle.keys())}. Fix H5_DATASET_KEY."
            )
        return handle[H5_DATASET_KEY][:]


def load_word_events(events_path):
    """Read the in-vocabulary word onsets and labels from one events table.

    Only the 50 scored words survive. Labels come from `vocab.to_id`, never
    from the task's own numbering -- the two orderings disagree and mixing
    them scores at chance.

    ARGS:
        events_path (Path): the events TSV for one recording

    RETURNS:
        onsets_s : [n_words] onsets in seconds from the start of the recording
        labels : [n_words] competition class ids, aligned with `onsets_s`
    """
    events = pd.read_csv(events_path, sep="\t")

    #name the missing column rather than raising a bare KeyError deep in pandas
    required = {EVENTS_ONSET_COLUMN, EVENTS_KIND_COLUMN, EVENTS_SEGMENT_COLUMN}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(
            f"{events_path.name} lacks {sorted(missing)}; "
            f"found {list(events.columns)}. Fix the EVENTS_* constants."
        )

    #word rows only; the table also holds phonemes and silences
    words = events[events[EVENTS_KIND_COLUMN] == WORD_KIND]

    #restrict to the 50 scored words before anything else touches them
    is_scored = words[EVENTS_SEGMENT_COLUMN].map(is_target)
    scored = words[is_scored]

    onsets_s = scored[EVENTS_ONSET_COLUMN].to_numpy(dtype=np.float64)
    labels = np.array(
        [to_id(word) for word in scored[EVENTS_SEGMENT_COLUMN]], dtype=np.int16
    )
    return onsets_s, labels


def build_cache(
    data_dir,
    out_dir,
    sample_rate=250.0,
    n_components=73,
    val_sessions=None,
    val_fraction=0.1,
    bounds=WINDOW_BOUNDS,
    seed=0):
    """Fit the transforms on the training split and write one shard per run.

    Recordings are re-read from disk on each pass rather than held in memory:
    all 162 at float32 is ~40 GB. Paths are what get iterated, so the passes
    cannot silently exhaust one another the way a generator would.

    ARGS:
        data_dir (Path): the cache root holding the task folders
        out_dir (Path): where shards, transforms and the manifest are written
        sample_rate (float): samples per second of the recordings
        n_components (int): whitener rank, 73 is the measured effective rank
        val_sessions (list or None): sessions to hold out, else taken from the
            end until they cover `val_fraction` of the windows
        val_fraction (float): held-out share of WINDOWS, not of sessions
        bounds (tuple): window bounds in seconds relative to word onset
        seed (int): seed for the standardiser's timepoint draw

    RETURNS:
        manifest (dict): what was written, and the split it was written under
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(data_dir)

    #pass 0 reads only the event tables, which are small, so the split can be
    #sized by window count rather than by how many sessions happen to be held
    events_by_key = {run.key: load_word_events(run.events_path) for run in runs}

    #one session id per window, which is what session_split expects
    window_sessions = []
    for run in runs:
        onsets_s, _ = events_by_key[run.key]
        window_sessions.extend([run.session] * len(onsets_s))

    train_window_idx, _ = session_split(window_sessions, val_sessions, val_fraction)

    #collapse the per-window split back to the set of training sessions
    train_sessions = {window_sessions[i] for i in train_window_idx}
    train_runs = [run for run in runs if run.session in train_sessions]
    val_runs = [run for run in runs if run.session not in train_sessions]

    # --- AI, 2026-09-24: progress reporting on a job that runs for hours. ---
    def announce(paths, label):
        """Yield recordings while printing which one is being read.

        ARGS:
            paths (list): RunPaths to read, in order
            label (str): the phase name, for the log line

        RETURNS:
            generator of [n_channels, n_times] recordings
        """
        #flushed, because this output is redirected to a log on a long job
        for i, run in enumerate(paths, 1):
            print(f"[cache] {label} {i}/{len(paths)} {run.key}", flush=True)
            yield load_recording(run.meg_path)

    print(f"[cache] {len(runs)} recordings, {len(train_runs)} of them training",
          flush=True)
    # --- end AI ---

    #both transforms are fitted on training recordings only, re-reading each
    standardiser = ChannelStandardiser.fit(
        announce(train_runs, "standardiser fit"),
        scale_factor=STANDARDISER_SUBSAMPLE,
        seed=seed,
    )
    whitener = Whitener.fit(
        announce(train_runs, "whitener fit"),
        standardiser,
        n_components=n_components,
    )

    shards, total_windows, clipped_fractions = [], 0, []
    for run_index, run in enumerate(runs, 1):
        onsets_s, labels = events_by_key[run.key]

        # --- AI, 2026-09-24 ---
        #the running window total is what says whether the build is on track
        print(
            f"[cache] shard {run_index}/{len(runs)} {run.key} "
            f"({len(onsets_s)} words, {total_windows} windows so far)",
            flush=True,
        )
        # --- end AI ---

        #standardise then whiten before cutting, so a window is already in the
        #73-component space and the shard is 4x smaller than a sensor-space one
        recording = load_recording(run.meg_path)
        whitened = whitener.apply(standardiser.apply(recording))

        windows, kept = cut_windows(whitened, onsets_s, sample_rate, bounds)

        #kept re-aligns the labels; the full array would be shifted by every
        #window that fell off an end of the recording
        stored, clipped_fraction = clip_and_cast(windows)
        run_labels = labels[kept]

        np.save(out_dir / f"windows_{run.key}.npy", stored)
        np.save(out_dir / f"labels_{run.key}.npy", run_labels)

        shards.append(
            {
                "key": run.key,
                "session": run.session,
                "split": "train" if run.session in train_sessions else "val",
                "n_windows": int(len(kept)),
                "n_dropped": int(len(onsets_s) - len(kept)),
                "clipped_fraction": clipped_fraction,
            }
        )
        total_windows += len(kept)
        clipped_fractions.append(clipped_fraction)

    #the transforms travel with the cache; the holdout must reuse these exact
    #arrays rather than refit, or it is served statistics it never saw
    np.savez(
        out_dir / "transforms.npz",
        centre=standardiser.centre,
        scale=standardiser.scale,
        matrix=whitener.matrix,
        eigenvalues=whitener.eigenvalues,
        shrinkage=whitener.shrinkage,
    )

    manifest = {
        "n_windows": total_windows,
        "n_samples_per_window": int(round((bounds[1] - bounds[0]) * sample_rate)),
        "n_components": n_components,
        "sample_rate": sample_rate,
        "bounds": list(bounds),
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted({run.session for run in val_runs}),
        "max_clipped_fraction": max(clipped_fractions) if clipped_fractions else 0.0,
        "shards": shards,
    }
    with open(out_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    return manifest
