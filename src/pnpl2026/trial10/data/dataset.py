"""Read sequences out of the reference cache.

Recordings are stored time-major, [n_times, 306], so a word's window is one
contiguous slice. It is transposed to channels-first here and zero-padded to
the full window; `lengths` carries how much of it is real. Under the reference
protocol every window is full; under the holdout protocol it stops at the
sequence's end.

The scaler was applied at build time; the clamp and the baseline are the
model's, so a holdout window goes through them identically.

A batch is FLAT, as in trials 7-8: every token of every sequence, with its
sequence index and position, so padding never crosses to the GPU as windows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MANIFEST_NAME = "manifest.json"


def load_manifest(cache_dir):
    """Read the reference cache's manifest.

    ARGS:
        cache_dir (Path or str)

    RETURNS:
        manifest (dict)
    """
    path = Path(cache_dir) / MANIFEST_NAME
    if not path.exists():
        raise FileNotFoundError(
            f"No {MANIFEST_NAME} in {cache_dir}. Build the reference cache first:\n"
            f"    python -m pnpl2026.trial10.cli.build_reference --config ..."
        )
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


class ReferenceRecordings:
    """One pnpl partition of the cache: recordings and their word tables.

    ATTRS:
        cache_dir (Path)
        sample_rate (float): 50 Hz
        records (list): the manifest's entries for this partition
        tables (list): word tables, as dicts of arrays
        n_times (list): samples per recording
    """

    def __init__(self, cache_dir, partition):
        """
        ARGS:
            cache_dir (Path or str)
            partition (str): "train", "validation" or "test", pnpl's own
        """
        self.cache_dir = Path(cache_dir)
        manifest = load_manifest(cache_dir)
        self.sample_rate = float(manifest["sample_rate"])

        #the partition was fixed at build time from pnpl's manifest; honour it
        self.records = [r for r in manifest["recordings"] if r["partition"] == partition]
        if not self.records:
            raise ValueError(
                f"No {partition} recordings in {cache_dir}. Download them, then rebuild:\n"
                f"    python -m pnpl2026.cli.prepare --partition {partition} --subjects deep"
            )

        self.tables = []
        for record in self.records:
            with np.load(self.cache_dir / f"words_{record['key']}.npz") as stored:
                self.tables.append({name: stored[name] for name in stored.files})
        self.n_times = [int(record["n_times"]) for record in self.records]
        self._handles = {}

    def meg(self, recording):
        """The memory-mapped [n_times, 306] array of one recording.

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


class SequenceWindows(Dataset):
    """Sequences, each word's window sliced from its recording on read.

    ATTRS:
        recordings (ReferenceRecordings)
        sequences (list): of Sequence
        window_samples (int): the full window, and the padded length
        jitter_samples (int): one-sided onset jitter, 0 outside training
    """

    def __init__(self, recordings, sequences, window_samples, jitter_samples=0):
        """
        ARGS:
            recordings (ReferenceRecordings)
            sequences (list): of Sequence
            window_samples (int): 150 at 50 Hz
            jitter_samples (int): 0 is the reference, and always for scoring
        """
        self.recordings = recordings
        self.sequences = sequences
        self.window_samples = window_samples
        self.jitter_samples = jitter_samples

    def __len__(self):
        """Sequences, not words: one item is one sequence."""
        return len(self.sequences)

    def __getitem__(self, index):
        """Slice every word's window of one sequence.

        ARGS:
            index (int): position in `sequences`

        RETURNS:
            dict of tensors, one row per word:
                windows [n_tokens, 306, window_samples] float16, zero past
                each window's length; lengths, targets, primary, moses
                [n_tokens] int64; isolated [n_tokens] bool
        """
        sequence = self.sequences[index]
        meg = self.recordings.meg(sequence.recording)
        n_times = meg.shape[0]
        n_tokens = len(sequence.onsets)

        #where the sequence's MEG stops: nowhere under the reference protocol,
        #its last onset plus the tail under the holdout one
        end = n_times
        if sequence.tail is not None and not sequence.isolated:
            end = min(int(sequence.onsets[-1]) + sequence.tail, n_times)

        windows = np.zeros((n_tokens, meg.shape[1], self.window_samples), dtype=np.float16)
        lengths = np.zeros(n_tokens, dtype=np.int64)
        for position, onset in enumerate(sequence.onsets):
            #jitter shifts WHICH samples are cut; never past either end
            start = int(onset)
            if self.jitter_samples > 0:
                start += int(torch.randint(-self.jitter_samples, self.jitter_samples + 1, (1,)).item())
            start = min(max(start, 0), n_times - 1)

            #an isolated word gets exactly its window; any other word up to the
            #full window, stopped by its sequence's end
            if sequence.isolated:
                stop = min(start + sequence.tail, n_times)
            else:
                stop = min(start + self.window_samples, end)

            #[time, sensors] -> [sensors, time], into the padded buffer
            windows[position, :, : stop - start] = np.asarray(meg[start:stop]).T
            lengths[position] = stop - start

        return {
            "windows": torch.from_numpy(windows),
            "lengths": torch.from_numpy(lengths),
            "targets": torch.from_numpy(sequence.targets.astype(np.int64)),
            "primary": torch.from_numpy(sequence.primary.astype(np.int64)),
            "moses": torch.from_numpy(sequence.moses.astype(np.int64)),
            "isolated": torch.full((n_tokens,), sequence.isolated, dtype=torch.bool),
        }


def collate_tokens(items):
    """Flatten a list of sequences into one batch of tokens.

    ARGS:
        items (list): dicts from `SequenceWindows.__getitem__`

    RETURNS:
        dict: every field concatenated over tokens, plus
            sequence [n_tokens] -- which sequence a token came from
            position [n_tokens] -- its place in that sequence
    """
    batch = {name: torch.cat([item[name] for item in items], dim=0) for name in items[0]}

    #the grid coordinates the sentence transformer needs, in concatenation order
    batch["sequence"] = torch.cat(
        [torch.full((len(item["targets"]),), index, dtype=torch.int64) for index, item in enumerate(items)]
    )
    batch["position"] = torch.cat([torch.arange(len(item["targets"]), dtype=torch.int64) for item in items])
    return batch
