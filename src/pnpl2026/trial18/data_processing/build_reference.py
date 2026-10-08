"""Build trial 10's reference cache: every subject-0 recording, 306 sensors, 50 Hz.

d'Ascoli et al.'s preprocessing, applied once to whole recordings:

    h5 [306, n] at 250 Hz, already 0.1-125 Hz bandpassed
      -> resample_poly, down 5      the anti-alias FIR is the low-pass: the
                                    paper's 40 Hz band ends at 25 Hz at 50 Hz
      -> RobustScaler PER RECORDING per-channel median and IQR from 200 random
                                    3 s segments, as brainmagick's BatchScaler
    meg_<key>.npy      [n_times, 306] float16, TIME-MAJOR, scaled, not clamped
    words_<key>.npz    every word: onset_sample (50 Hz), sentence, word,
                       primary and moses columns (-1 when unscored)
    scaler_<key>.npz   the median and IQR the recording was scaled by
    positions.npy      [306, 2] sensor positions in [0, 1], for the merger
    pca.npz            73-component PCA of the training recordings, for F3
    manifest.json      recordings, pnpl partition, channel names, totals

The clamp to +/-5 and the 0.5 s baseline happen at read time, per window.

⚠️ The partition of each recording is pnpl's, from its manifest -- never
re-derived. The reference was selected on the validation partition and scored
on the test partition, and so is this trial.

⚠️ Time-major on purpose: a 3 s window is then one contiguous 92 KB read
rather than 306 strided ones.

⚠️ Heavy: reads every h5 in full. pbs/trial10_build.pbs runs it as a CPU job.

Trial 18 adds `build_others`: the same processing for every OTHER subject's
recordings (LibriBrain100's 32 broad-track subjects, Sherlock1 sessions 11
and 12), into a cache of their own whose keys carry `sub-<id>` and whose
records name their subject. Subject 0's cache is not touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from ..data.words import Vocabulary, normalise_word

#verified against real recordings on 2026-09-14, per CLAUDE.md
H5_DATASET_KEY = "data"
H5_CHANNEL_NAMES = "channel_names"

#⚠️ `timeds`, not `timemeg`: timemeg is the 1 kHz timeline and lands off the
#sample grid 6% of the time. timeds matches the h5 exactly
EVENTS_ONSET_COLUMN = "timeds"
EVENTS_KIND_COLUMN = "kind"
EVENTS_SEGMENT_COLUMN = "segment"
EVENTS_SENTENCE_COLUMN = "sentenceidx"
WORD_KIND = "word"

#the deep track is subject 0; other subjects' files can share the data tree
SUBJECT = "0"

#250 Hz in, 50 Hz out, as the paper resamples
SAMPLE_RATE_IN = 250.0
DOWNSAMPLE = 5
SAMPLE_RATE_OUT = SAMPLE_RATE_IN / DOWNSAMPLE

#brainmagick's BatchScaler: 200 segments per recording, here of a word's 3 s
SCALER_SEGMENTS = 200
SCALER_SEGMENT_SAMPLES = 150

#a dead channel's IQR would divide by zero; nothing real is this small after
#the h5's own scaling, so it only ever catches flat channels
IQR_FLOOR = 1e-12

#F3's input: the effective rank CLAUDE.md measured, and the clamp it sees
PCA_COMPONENTS = 73
PCA_SAMPLES_PER_RECORDING = 4000
CLAMP = 5.0


def discover_recordings(data_dir, subjects=(SUBJECT,)):
    """Find every recording of the chosen subjects with both an h5 and an events table.

    ARGS:
        data_dir (Path): the download root, from `config.DATA_DIR`
        subjects (collection or None): subject ids kept; None keeps every
            subject but 0

    RETURNS:
        list of dict: meg_path, events_path, subject, session, task, run,
        key -- sorted. Subject 0's keys are trial 10's; others' start sub-<id>
    """
    meg_paths = sorted(Path(data_dir).glob("*/derivatives/serialised/*_meg.h5"))

    recordings = []
    for meg_path in meg_paths:
        #"sub-0_ses-10_task-Sherlock1_run-1_proc-..._meg" -> its fields
        parts = dict(piece.split("-", 1) for piece in meg_path.stem.split("_") if "-" in piece)

        #⚠️ the validation and test downloads bring other subjects too
        subject = parts.get("sub")
        if subject is None or (subjects is None and subject == SUBJECT):
            continue
        if subjects is not None and subject not in subjects:
            continue

        stem = f"sub-{parts['sub']}_ses-{parts['ses']}_task-{parts['task']}_run-{parts['run']}"
        events_path = meg_path.parent.parent / "events" / f"{stem}_events.tsv"
        if not events_path.exists():
            continue

        #subject 0 keeps trial 10's keys, so its cache stays readable as built
        key = f"ses-{parts['ses']}_task-{parts['task']}_run-{parts['run']}"
        if subject != SUBJECT:
            key = f"sub-{subject}_{key}"
        recordings.append(
            {
                "meg_path": meg_path,
                "events_path": events_path,
                "subject": subject,
                "session": parts["ses"],
                "task": parts["task"],
                "run": parts["run"],
                "key": key,
            }
        )
    return recordings


def pnpl_partitions():
    """pnpl's own partition for every subject-0 recording.

    RETURNS:
        dict: (session, task, run) -> "train" | "validation" | "test"
    """
    from pnpl.datasets.libribrain100.manifest import RUN_RECORDS

    return {
        (record.session, record.task, record.run): record.partition
        for record in RUN_RECORDS
        if record.subject == SUBJECT
    }


def downsample(recording):
    """250 Hz to 50 Hz with a polyphase anti-alias filter.

    resample_poly is zero-phase, so input sample 5k lands on output sample k:
    an onset at t seconds is at round(t * 50) both before and after.

    ARGS:
        recording (np.ndarray): [n_channels, n_times] at 250 Hz

    RETURNS:
        [n_channels, ceil(n_times / 5)] float32 at 50 Hz
    """
    return resample_poly(recording, 1, DOWNSAMPLE, axis=1).astype(np.float32)


def fit_robust_scaler(recording, generator):
    """Per-channel median and IQR from random 3 s segments of one recording.

    brainmagick fits sklearn's RobustScaler per recording on 200 sampled
    segments; this is the same statistic, computed directly.

    ARGS:
        recording (np.ndarray): [n_channels, n_times] at 50 Hz
        generator (np.random.Generator): for the segment starts

    RETURNS:
        median : [n_channels] float64
        iqr : [n_channels] float64, floored at IQR_FLOOR
    """
    n_times = recording.shape[1]
    span = SCALER_SEGMENT_SAMPLES

    #a recording shorter than the segments is used whole
    if n_times <= span * 2:
        sample = recording
    else:
        starts = generator.integers(0, n_times - span, SCALER_SEGMENTS)
        sample = np.concatenate([recording[:, start : start + span] for start in starts], axis=1)

    #axis=1 is time: one statistic per channel
    q25, median, q75 = np.percentile(sample.astype(np.float64), [25, 50, 75], axis=1)
    return median, np.maximum(q75 - q25, IQR_FLOOR)


def load_word_table(events_path):
    """Every word of one recording, in onset order, at 50 Hz.

    ARGS:
        events_path (Path): the events TSV

    RETURNS:
        dict of arrays, one row per word: onset_sample (int64, 50 Hz),
        sentence (int64, -1 where the table has none), word (normalised str),
        primary and moses (int16 columns, -1 when unscored)
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

    #a segment that was punctuation only normalises to "" and is not a word
    normalised = np.array([normalise_word(word) for word in words[EVENTS_SEGMENT_COLUMN]])
    real = normalised != ""
    words, normalised = words[real], normalised[real]

    primary, moses = Vocabulary("primary"), Vocabulary("moses")
    sentence = words[EVENTS_SENTENCE_COLUMN].to_numpy(dtype=np.float64)

    return {
        #round(t * 50): the onset on the 50 Hz grid, at most 10 ms off
        "onset_sample": np.round(words[EVENTS_ONSET_COLUMN].to_numpy(dtype=np.float64) * SAMPLE_RATE_OUT).astype(np.int64),
        "sentence": np.where(np.isfinite(sentence), sentence, -1).astype(np.int64),
        "word": normalised,
        "primary": np.array([primary.label(word) for word in normalised], dtype=np.int16),
        "moses": np.array([moses.label(word) for word in normalised], dtype=np.int16),
    }


def channel_positions(channel_names):
    """2D sensor positions in [0, 1] from MNE's Neuromag 306 layout.

    brainmagick's PositionGetter does the same from `mne.find_layout`; the h5
    carries names, not an MNE info, so the layout is read by name.

    ARGS:
        channel_names (list): as stored in the h5, e.g. "MEG0113"

    RETURNS:
        [n_channels, 2] float32, or None when a name is not in the layout
    """
    import mne

    layout = mne.channels.read_layout("Vectorview-all")

    #MNE spells them "MEG 0113"; the h5 may not have the space
    lookup = {name.replace(" ", ""): index for index, name in enumerate(layout.names)}
    try:
        rows = [lookup[name.replace(" ", "")] for name in channel_names]
    except KeyError:
        return None

    #the centre of each sensor's box, then min-max to [0, 1] per axis
    boxes = layout.pos[rows]
    centres = boxes[:, :2] + boxes[:, 2:] / 2
    span = centres.max(axis=0) - centres.min(axis=0)
    return ((centres - centres.min(axis=0)) / span).astype(np.float32)


def read_channel_names(handle):
    """The h5's channel names, or None if it does not carry them.

    ARGS:
        handle (h5py.File): an open recording

    RETURNS:
        list of str, or None
    """
    if H5_CHANNEL_NAMES not in handle.attrs:
        return None
    raw = handle.attrs[H5_CHANNEL_NAMES]
    text = raw.decode() if isinstance(raw, bytes) else str(raw)
    return [name.strip() for name in text.split(",")]


def build_reference(data_dir, out_dir, force=False, seed=0):
    """Write the reference cache for every subject-0 recording pnpl partitions.

    ARGS:
        data_dir (Path): the download root holding the task folders
        out_dir (Path): where the cache is written
        force (bool): rebuild recordings that already exist
        seed (int): for the scaler's segments and the PCA's samples

    RETURNS:
        manifest (dict)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    partitions = pnpl_partitions()
    generator = np.random.default_rng(seed)

    names, recordings, skipped = None, [], []

    #F3's PCA: float64 moments of clamped training samples, accumulated here
    #so the fit needs no second pass over 156 recordings
    pca_count, pca_sum, pca_outer = 0, None, None

    for found in discover_recordings(data_dir):
        partition = partitions.get((found["session"], found["task"], found["run"]))
        if partition is None:
            skipped.append({"key": found["key"], "reason": "not in pnpl's manifest"})
            continue

        meg_file = out_dir / f"meg_{found['key']}.npy"
        words_file = out_dir / f"words_{found['key']}.npz"
        scaler_file = out_dir / f"scaler_{found['key']}.npz"

        #an existing triple is reused unless forced, so a killed build resumes
        if force or not (meg_file.exists() and words_file.exists() and scaler_file.exists()):
            with h5py.File(found["meg_path"], "r") as handle:
                recording = handle[H5_DATASET_KEY][:]
                these_names = read_channel_names(handle)

            #⚠️ every recording must list its sensors in the same order, or
            #one set of positions would describe the wrong sensors
            if these_names is not None:
                if names is None:
                    names = these_names
                elif these_names != names:
                    raise ValueError(f"{found['key']} lists its channels in a different order")

            downsampled = downsample(recording)
            median, iqr = fit_robust_scaler(downsampled, generator)
            scaled = (downsampled - median[:, None]) / iqr[:, None]

            np.save(meg_file, np.ascontiguousarray(scaled.T).astype(np.float16))
            np.savez(words_file, **load_word_table(found["events_path"]))
            np.savez(scaler_file, median=median, iqr=iqr)
        elif names is None:
            #a resumed build still needs the names, from any recording
            with h5py.File(found["meg_path"], "r") as handle:
                names = read_channel_names(handle)

        meg = np.load(meg_file, mmap_mode="r")
        with np.load(words_file) as table:
            n_words = int(len(table["primary"]))
            n_in_vocab = int((table["primary"] >= 0).sum())
            n_sentences = int(len(np.unique(table["sentence"][table["sentence"] >= 0])))

        #PCA samples from TRAINING recordings only, clamped as the model sees
        if partition == "train":
            rows = generator.integers(0, meg.shape[0], PCA_SAMPLES_PER_RECORDING)
            sample = np.clip(np.asarray(meg[np.sort(rows)], dtype=np.float64), -CLAMP, CLAMP)
            pca_count += len(sample)
            pca_sum = sample.sum(axis=0) if pca_sum is None else pca_sum + sample.sum(axis=0)
            pca_outer = sample.T @ sample if pca_outer is None else pca_outer + sample.T @ sample

        recordings.append(
            {
                "key": found["key"],
                "session": found["session"],
                "task": found["task"],
                "run": found["run"],
                "partition": partition,
                "n_times": int(meg.shape[0]),
                "n_words": n_words,
                "n_in_vocab": n_in_vocab,
                "n_sentences": n_sentences,
            }
        )
        print(
            f"[build] {found['key']:40s} {partition:10s} {meg.shape[0] / SAMPLE_RATE_OUT / 60:5.1f} min "
            f"{n_words:6d} words {n_in_vocab:6d} in-vocab {n_sentences:5d} sentences"
        )

    #positions only when every recording named its sensors and MNE knows them
    positions = channel_positions(names) if names is not None else None
    if positions is not None:
        np.save(out_dir / "positions.npy", positions)

    #PCA-whitening: training mean, leading eigenvectors, 1/sqrt(eigenvalue)
    if pca_count > 1:
        mean = pca_sum / pca_count
        covariance = (pca_outer - pca_count * np.outer(mean, mean)) / (pca_count - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:PCA_COMPONENTS]
        np.savez(
            out_dir / "pca.npz",
            mean=mean.astype(np.float32),
            components=eigenvectors[:, order].T.astype(np.float32),
            scale=(1.0 / np.sqrt(np.maximum(eigenvalues[order], 1e-12))).astype(np.float32),
            explained=np.float64(eigenvalues[order].sum() / np.maximum(eigenvalues, 0).sum()),
        )

    manifest = {
        "sample_rate": SAMPLE_RATE_OUT,
        "n_channels": 306,
        "layout": "time-major [n_times, n_channels], per-recording robust-scaled, not clamped",
        "channel_names": names,
        "has_positions": positions is not None,
        "recordings": recordings,
        "skipped": skipped,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def build_others(data_dir, out_dir, reference_dir, force=False, seed=0):
    """Write a cache of every other subject's recordings, processed as subject 0's.

    Only recordings pnpl's LibriBrain100 manifest lists are taken. All of them
    are training data here, whatever pnpl's own partition (kept as
    `pnpl_partition`): they are subject 0's validation and test STORIES, heard
    by other people.

    ARGS:
        data_dir (Path): the download root holding the task folders
        out_dir (Path): where the others' cache is written
        reference_dir (Path): subject 0's cache, for its channel order and
            sensor positions
        force (bool): rebuild recordings that already exist
        seed (int): for the scalers' segments

    RETURNS:
        manifest (dict)
    """
    from pnpl.datasets.libribrain100.manifest import RUN_RECORDS

    out_dir, reference_dir = Path(out_dir), Path(reference_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)

    #subject 0's sensors: every other recording must list them in this order
    with open(reference_dir / "manifest.json", encoding="utf-8") as handle:
        reference = json.load(handle)
    names = reference["channel_names"]

    #pnpl's own partition for each other subject's recording
    listed = {
        (record.subject, record.session, record.task, record.run): record.partition
        for record in RUN_RECORDS
        if record.subject != SUBJECT
    }

    recordings, skipped = [], []
    for found in discover_recordings(data_dir, subjects=None):
        pnpl_partition = listed.get((found["subject"], found["session"], found["task"], found["run"]))
        if pnpl_partition is None:
            skipped.append({"key": found["key"], "reason": "not in pnpl's manifest"})
            continue

        meg_file = out_dir / f"meg_{found['key']}.npy"
        words_file = out_dir / f"words_{found['key']}.npz"
        scaler_file = out_dir / f"scaler_{found['key']}.npz"

        #an existing triple is reused unless forced, so a killed build resumes
        if force or not (meg_file.exists() and words_file.exists() and scaler_file.exists()):
            with h5py.File(found["meg_path"], "r") as handle:
                recording = handle[H5_DATASET_KEY][:]
                these_names = read_channel_names(handle)

            #⚠️ one set of positions describes every subject's sensors
            if names is not None and these_names is not None and these_names != names:
                raise ValueError(f"{found['key']} lists its channels unlike subject 0's cache")

            downsampled = downsample(recording)
            median, iqr = fit_robust_scaler(downsampled, generator)
            scaled = (downsampled - median[:, None]) / iqr[:, None]
            np.save(meg_file, np.ascontiguousarray(scaled.T).astype(np.float16))
            np.savez(words_file, **load_word_table(found["events_path"]))
            np.savez(scaler_file, median=median, iqr=iqr)

        meg = np.load(meg_file, mmap_mode="r")
        with np.load(words_file) as table:
            n_words = int(len(table["primary"]))
            n_in_vocab = int((table["primary"] >= 0).sum())
            n_sentences = int(len(np.unique(table["sentence"][table["sentence"] >= 0])))
        recordings.append(
            {
                "key": found["key"],
                "subject": found["subject"],
                "session": found["session"],
                "task": found["task"],
                "run": found["run"],
                "partition": "train",
                "pnpl_partition": pnpl_partition,
                "n_times": int(meg.shape[0]),
                "n_words": n_words,
                "n_in_vocab": n_in_vocab,
                "n_sentences": n_sentences,
            }
        )
        print(
            f"[others] {found['key']:48s} {meg.shape[0] / SAMPLE_RATE_OUT / 60:5.1f} min "
            f"{n_words:6d} words {n_in_vocab:6d} in-vocab"
        )

    #subject 0's positions, so the merger reads every subject's sensors alike
    has_positions = (reference_dir / "positions.npy").exists()
    if has_positions:
        np.save(out_dir / "positions.npy", np.load(reference_dir / "positions.npy"))

    manifest = {
        "sample_rate": SAMPLE_RATE_OUT,
        "n_channels": 306,
        "layout": "time-major [n_times, n_channels], per-recording robust-scaled, not clamped",
        "channel_names": names,
        "has_positions": has_positions,
        "subjects": sorted({record["subject"] for record in recordings}, key=lambda subject: int(subject)),
        "recordings": recordings,
        "skipped": skipped,
    }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
