"""Read holdout-shaped sequences out of the continuous cache, as streams.

Trial 11 serves each sequence as ONE stream: its MEG from the first word's
onset to the sequence's end, zero-padded to one length, plus every word's
onset as an offset into it. Trials 7-9 served a separate window per word.

⚠️ The end is what makes this the holdout's problem and not an easier one. In
a holdout sentence the MEG stops 1.1-1.7 s after the last scored word, so the
stream does too; an isolated word gets exactly its 1 s.

A batch holds S streams and a flat list of N words, each with the index of
its stream and its offset in samples. The model reads each word off its own
stream; nothing crosses from one stream to another.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_NAME = "manifest.json"


def load_manifest(cache_dir):
    """Read the continuous cache's manifest.

    ARGS:
        cache_dir (Path or str): the directory `cli.build_continuous` wrote

    RETURNS:
        manifest (dict)
    """
    path = Path(cache_dir) / MANIFEST_NAME

    #say what is missing and how to make it, rather than a bare FileNotFound
    if not path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {cache_dir}. Build the continuous cache first:\n"
            f"    python -m pnpl2026.trial7.cli.build_continuous --config ..."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ContinuousRecordings:
    """One split of the continuous cache: recordings and their word tables.

    Recordings stay on disk as memory maps, opened lazily in whichever process
    reads them -- opened before a DataLoader forks, the handles would be shared.

    ATTRS:
        cache_dir (Path)
        records (list): the manifest's entries for this split
        tables (list): one word table per recording, as dicts of arrays
        n_times (list): samples per recording
    """

    def __init__(self, cache_dir, split):
        """
        ARGS:
            cache_dir (Path or str): the continuous cache
            split (str): "train" or "val", as the builder recorded it
        """
        self.cache_dir = Path(cache_dir)
        manifest = load_manifest(cache_dir)
        self.sample_rate = float(manifest["sample_rate"])
        self.n_components = int(manifest["n_components"])

        #the split was fixed at build time from trial 1's; honour it
        self.records = [r for r in manifest["recordings"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"No recordings marked split={split!r} in {cache_dir}.")

        #word tables are small; held outright, without the word strings
        self.tables = []
        for record in self.records:
            with np.load(self.cache_dir / f"words_{record['key']}.npz") as stored:
                self.tables.append(
                    {name: stored[name] for name in ("onset_sample", "sentence", "label")}
                )
        self.n_times = [int(record["n_times"]) for record in self.records]
        self._handles = {}

    def meg(self, recording):
        """The memory-mapped [n_components, n_times] array of one recording.

        ARGS:
            recording (int): index into `records`

        RETURNS:
            np.memmap, float16
        """
        if recording not in self._handles:
            key = self.records[recording]["key"]
            self._handles[recording] = np.load(self.cache_dir / f"meg_{key}.npy", mmap_mode="r")
        return self._handles[recording]

    def __getstate__(self):
        """Drop open memory maps when pickled into a worker; it reopens its own."""
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state


class SequenceStreams(Dataset):
    """Holdout-shaped sequences, each one stream sliced from its recording.

    ATTRS:
        recordings (ContinuousRecordings)
        sequences (list): of sequences.Sequence
        stream_samples (int): the padded stream length
        tail_range (tuple): (low, high) samples, for tails drawn per read
        jitter_samples (int): one-sided start jitter, 0 outside training
    """

    def __init__(self, recordings, sequences, stream_samples, tail_range, jitter_samples=0):
        """
        ARGS:
            recordings (ContinuousRecordings): where the MEG is
            sequences (list): of Sequence, from `build_sequences`
            stream_samples (int): every stream is zero-padded to this length
            tail_range (tuple): (low, high) samples kept after the last onset
            jitter_samples (int): one-sided jitter; pass 0 for validation
        """
        self.recordings = recordings
        self.sequences = sequences
        self.stream_samples = stream_samples
        self.tail_range = tail_range
        self.jitter_samples = jitter_samples

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        """Slice one sequence's stream.

        ARGS:
            index (int): position in `sequences`

        RETURNS:
            stream : [n_components, stream_samples] float16, zero past `length`
            length : int, real samples in the stream
            offsets : [n_tokens] int64, each onset in samples from the stream start
            labels : [n_tokens] int64 class ids
            isolated : [n_tokens] bool
        """
        sequence = self.sequences[index]
        meg = self.recordings.meg(sequence.recording)
        n_times = meg.shape[1]
        first = int(sequence.onsets[0])

        #one shift for the whole stream: every word is read from its nominal
        #onset, so the readout lands up to `jitter` early or late, jointly
        start = first
        if self.jitter_samples > 0:
            start += int(torch.randint(-self.jitter_samples, self.jitter_samples + 1, (1,)).item())
        start = min(max(start, 0), n_times - 1)

        #where the MEG stops: an isolated word gets exactly its window from the
        #start, as the holdout serves it; a sentence stops at its last onset
        #plus a tail -- fixed for validation, drawn per read for training
        if sequence.tail is not None:
            tail = sequence.tail
        else:
            tail = int(torch.randint(self.tail_range[0], self.tail_range[1] + 1, (1,)).item())
        if sequence.isolated:
            end = min(start + tail, n_times)
        else:
            end = min(int(sequence.onsets[-1]) + tail, n_times)

        #⚠️ the sequences are built to fit; a longer stream means the config's
        #span, tail and stream length disagree, and is never cropped silently
        length = end - start
        if length > self.stream_samples:
            raise ValueError(
                f"Stream of {length} samples exceeds the padded {self.stream_samples}; "
                f"raise tmax or lower max_onset_span_seconds."
            )

        stream = np.zeros((meg.shape[0], self.stream_samples), dtype=np.float16)
        stream[:, :length] = meg[:, start:end]

        #offsets are from the NOMINAL start, so jitter moves the signal under
        #the readout rather than the readout with the signal
        offsets = sequence.onsets.astype(np.int64) - first
        n_tokens = len(offsets)

        return (
            torch.from_numpy(stream),
            length,
            torch.from_numpy(offsets),
            torch.from_numpy(sequence.labels.astype(np.int64)),
            torch.full((n_tokens,), sequence.isolated, dtype=torch.bool),
        )


def collate_streams(items):
    """Stack streams; flatten their words, each tagged with its stream.

    ARGS:
        items (list): (stream, length, offsets, labels, isolated) per sequence

    RETURNS:
        dict:
            streams : [S, n_components, stream_samples] float16
            stream_lengths : [S] int64
            offsets : [N] int64, onset in samples from its stream's start
            labels : [N] int64
            isolated : [N] bool
            sequence : [N] int64, which stream each word is in
    """
    streams, lengths, offsets, labels, isolated = zip(*items)

    #the stream index follows the concatenation order exactly
    sequence = torch.cat(
        [torch.full((len(item),), index, dtype=torch.int64) for index, item in enumerate(labels)]
    )

    return {
        "streams": torch.stack(streams, dim=0),
        "stream_lengths": torch.tensor(lengths, dtype=torch.int64),
        "offsets": torch.cat(offsets, dim=0),
        "labels": torch.cat(labels, dim=0),
        "isolated": torch.cat(isolated, dim=0),
        "sequence": sequence,
    }
