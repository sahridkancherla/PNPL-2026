"""Build trial 7's continuous cache: whole recordings, whitened, plus word tables.

Trial 1's cache holds pre-cut windows of in-vocabulary words only, 1.28 s after
onset, with no sentence ids. Variant D needs windows of any length up to 3 s,
cut per word and per sentence, so this stores each recording WHOLE instead:

    meg_<key>.npy      [73, n_times] float16 -- standardised, whitened, clipped
    words_<key>.npz    one row per word event, in onset order:
                       onset_sample, sentence, label, word
    manifest.json      recordings, their split, and what was skipped

Windows are then sliced on the fly. ~4.6 GB for every recording of subject 0,
against trial 1's 18.3 GB of overlapping windows.

⚠️ The transforms and the split are TRIAL 1's, read from its cache, never refit.
predict.py applies the same transforms to the holdout, and the split must stay
the one every earlier trial was scored on, or no comparison with them holds.

⚠️ Heavy: reads every h5 in full. Run it as a CPU batch job -- pbs/trial7_build.pbs
-- never on a login node.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ...data.vocab import is_target, to_id

#verified against real recordings on 2026-09-14, per CLAUDE.md
H5_DATASET_KEY = "data"

#⚠️ `timeds`, not `timemeg`: timemeg is the 1 kHz timeline and lands off the
#250 Hz grid 6% of the time. timeds matches the h5 exactly
EVENTS_ONSET_COLUMN = "timeds"
EVENTS_KIND_COLUMN = "kind"
EVENTS_SEGMENT_COLUMN = "segment"
EVENTS_SENTENCE_COLUMN = "sentenceidx"
WORD_KIND = "word"

#the deep track is subject 0; other subjects' files can share the data tree
SUBJECT = "0"

#clip threshold in whitened sigmas, the same as trial 1's cache and predict.py
CLIP_SIGMA = 20.0

#time samples transformed at once, which bounds float64 memory to ~0.5 GB
CHUNK_SAMPLES = 200_000


def discover_recordings(data_dir):
    """Find every subject-0 recording with both an h5 and an events table.

    ARGS:
        data_dir (Path): the download root, from `config.DATA_DIR`

    RETURNS:
        list of dict: meg_path, events_path, session, task, run, key -- sorted
    """
    #the serialised folder carries the processing chain in each filename
    meg_paths = sorted(Path(data_dir).glob("*/derivatives/serialised/*_meg.h5"))

    recordings = []
    for meg_path in meg_paths:
        #"sub-0_ses-10_task-Sherlock1_run-1_proc-..._meg" -> its fields
        parts = dict(piece.split("-", 1) for piece in meg_path.stem.split("_") if "-" in piece)

        #⚠️ other subjects' recordings sit in the same tree once the validation
        #or test partitions are downloaded, and must never enter subject 0's
        if parts.get("sub") != SUBJECT:
            continue

        #the events file drops the proc chain, so rebuild its name from the key
        stem = f"sub-{parts['sub']}_ses-{parts['ses']}_task-{parts['task']}_run-{parts['run']}"
        events_path = meg_path.parent.parent / "events" / f"{stem}_events.tsv"
        if not events_path.exists():
            continue

        recordings.append(
            {
                "meg_path": meg_path,
                "events_path": events_path,
                "session": parts["ses"],
                "task": parts["task"],
                "run": parts["run"],
                #the same key trial 1's shards use, so the two caches line up
                "key": f"ses-{parts['ses']}_task-{parts['task']}_run-{parts['run']}",
            }
        )
    return recordings


def load_transforms(trial1_cache):
    """Read trial 1's fitted standardiser and whitener.

    ARGS:
        trial1_cache (Path): trial 1's cache directory

    RETURNS:
        centre : [n_sensors] float64
        scale : [n_sensors] float64
        matrix : [n_components, n_sensors] float64
    """
    stored = np.load(Path(trial1_cache) / "transforms.npz")
    return (
        np.asarray(stored["centre"], dtype=np.float64),
        np.asarray(stored["scale"], dtype=np.float64),
        np.asarray(stored["matrix"], dtype=np.float64),
    )


def to_components(recording, centre, scale, matrix, clip_sigma=CLIP_SIGMA):
    """Standardise, whiten and clip a whole recording, chunk by chunk.

    The same arithmetic as trial 5's `predict.to_components`, which is what
    the holdout goes through: per-sensor affine, then the whitening matrix.

    ARGS:
        recording (np.ndarray): [n_sensors, n_times] float32, continuous
        centre (np.ndarray): [n_sensors]
        scale (np.ndarray): [n_sensors]
        matrix (np.ndarray): [n_components, n_sensors]
        clip_sigma (float): clip threshold after whitening

    RETURNS:
        components : [n_components, n_times] float16
    """
    n_times = recording.shape[1]
    components = np.empty((matrix.shape[0], n_times), dtype=np.float16)

    #float64 for the affine and the matmul: the 21x magnetometer/gradiometer
    #scale gap makes float32 marginal here
    for start in range(0, n_times, CHUNK_SAMPLES):
        chunk = recording[:, start : start + CHUNK_SAMPLES].astype(np.float64)
        standardised = (chunk - centre[:, None]) / scale[:, None]
        whitened = matrix @ standardised

        #clipped before the cast, since float16 overflows to inf at 65504
        components[:, start : start + CHUNK_SAMPLES] = np.clip(
            whitened, -clip_sigma, clip_sigma
        ).astype(np.float16)
    return components


def load_word_table(events_path, sample_rate):
    """Every word event of one recording, in onset order.

    ARGS:
        events_path (Path): the events TSV
        sample_rate (float): samples per second of the recording

    RETURNS:
        dict of arrays, one row per word:
            onset_sample (int64), sentence (int64, -1 where the table has
            none), label (int16, competition id or -1), word (str)
    """
    events = pd.read_csv(events_path, sep="\t")

    #name the missing column rather than raising a bare KeyError in pandas
    required = {EVENTS_ONSET_COLUMN, EVENTS_KIND_COLUMN, EVENTS_SEGMENT_COLUMN, EVENTS_SENTENCE_COLUMN}
    missing = required - set(events.columns)
    if missing:
        raise KeyError(f"{events_path.name} lacks {sorted(missing)}; found {list(events.columns)}")

    #word rows only, sorted by onset: the table also holds phonemes
    words = events[events[EVENTS_KIND_COLUMN] == WORD_KIND]
    words = words.sort_values(EVENTS_ONSET_COLUMN, kind="stable")

    segments = words[EVENTS_SEGMENT_COLUMN].astype(str).to_numpy()

    #labels from vocab.to_id, NEVER from a task's own numbering: the two
    #orderings disagree and mixing them scores at chance
    labels = np.array(
        [to_id(word) if is_target(word) else -1 for word in segments], dtype=np.int16
    )

    #a word with no sentence id becomes a sentence of its own downstream
    sentence = words[EVENTS_SENTENCE_COLUMN].to_numpy(dtype=np.float64)
    sentence = np.where(np.isfinite(sentence), sentence, -1).astype(np.int64)

    onset_sample = np.round(
        words[EVENTS_ONSET_COLUMN].to_numpy(dtype=np.float64) * sample_rate
    ).astype(np.int64)

    return {
        "onset_sample": onset_sample,
        "sentence": sentence,
        "label": labels,
        "word": segments,
    }


def build_continuous(data_dir, trial1_cache, out_dir, sample_rate=250.0, force=False):
    """Write the continuous cache for every recording trial 1's split covers.

    ARGS:
        data_dir (Path): the download root holding the task folders
        trial1_cache (Path): trial 1's cache, for its transforms and split
        out_dir (Path): where the continuous cache is written
        sample_rate (float): samples per second of the recordings
        force (bool): rebuild recordings that already exist

    RETURNS:
        manifest (dict): what was written, and what was skipped and why
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #trial 1's split, read rather than re-derived, so every trial scores on
    #the same validation sessions
    with open(Path(trial1_cache) / "manifest.json", "r", encoding="utf-8") as handle:
        trial1_manifest = json.load(handle)
    train_sessions = set(trial1_manifest["train_sessions"])
    val_sessions = set(trial1_manifest["val_sessions"])
    centre, scale, matrix = load_transforms(trial1_cache)

    #the transforms travel with the cache: predict.py reads them from here, so
    #the holdout is always put through exactly what built these recordings
    shutil.copy2(Path(trial1_cache) / "transforms.npz", out_dir / "transforms.npz")

    recordings, skipped = [], []
    for found in discover_recordings(data_dir):
        #a session trial 1 never saw has no split, and guessing one could put
        #a validation session into training
        if found["session"] in train_sessions:
            split = "train"
        elif found["session"] in val_sessions:
            split = "val"
        else:
            skipped.append({"key": found["key"], "reason": "session not in trial 1's split"})
            continue

        meg_file = out_dir / f"meg_{found['key']}.npy"
        words_file = out_dir / f"words_{found['key']}.npz"

        #an existing pair is reused unless forced, so a killed build resumes
        if force or not (meg_file.exists() and words_file.exists()):
            with h5py.File(found["meg_path"], "r") as handle:
                recording = handle[H5_DATASET_KEY][:]
            np.save(meg_file, to_components(recording, centre, scale, matrix))
            np.savez(words_file, **load_word_table(found["events_path"], sample_rate))

        #read back what is on disk, so the manifest describes the files
        n_times = int(np.load(meg_file, mmap_mode="r").shape[1])
        with np.load(words_file) as table:
            n_words = int(len(table["label"]))
            n_in_vocab = int((table["label"] >= 0).sum())
            n_sentences = int(len(np.unique(table["sentence"][table["sentence"] >= 0])))

        recordings.append(
            {
                "key": found["key"],
                "session": found["session"],
                "task": found["task"],
                "run": found["run"],
                "split": split,
                "n_times": n_times,
                "n_words": n_words,
                "n_in_vocab": n_in_vocab,
                "n_sentences": n_sentences,
            }
        )
        print(
            f"[build] {found['key']:40s} {split:5s} {n_times / sample_rate / 60:5.1f} min "
            f"{n_words:6d} words {n_in_vocab:6d} in-vocab {n_sentences:5d} sentences"
        )

    #recordings trial 1 had but this download lacks -- pnpl's train partition
    #omits six of the 162
    present = {recording["key"] for recording in recordings}
    absent = sorted(shard["key"] for shard in trial1_manifest["shards"] if shard["key"] not in present)

    manifest = {
        "sample_rate": sample_rate,
        "n_components": int(matrix.shape[0]),
        "clip_sigma": CLIP_SIGMA,
        "train_sessions": sorted(train_sessions),
        "val_sessions": sorted(val_sessions),
        "recordings": recordings,
        "skipped": skipped,
        "absent_from_trial1": absent,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
