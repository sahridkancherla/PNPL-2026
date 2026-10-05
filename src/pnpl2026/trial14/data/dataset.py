"""Read sequences out of the reference cache.

Recordings are stored time-major, [n_times, 306], so a word's window is one
contiguous slice. It is transposed to channels-first here and zero-padded to
the full window; `lengths` carries how much of it is real. Under the reference
protocol every window is full; under the fragment protocol it stops at the
fragment's end, as the holdout's epochs do.

Trial 13 adds SHAPING, for training only: each sentence may be served as a
holdout-like fragment (a random run of consecutive words, the MEG cut a tail
after its last onset) or as one word alone with 1 s. Every word still has its
t5 target, so no supervision is lost -- only context and window length.

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


class Shaping:
    """How training items are reshaped to look like the holdout.

    ATTRS:
        fragment_p (float): chance an item becomes a fragment
        isolated_p (float): chance an item becomes one word alone
        length_weights (torch.Tensor): [max_words] weights of lengths 1..max
        tail_samples (tuple): (low, high) MEG kept after a fragment's last onset
        isolated_samples (int): an isolated word's window
    """

    def __init__(self, fragment_p, isolated_p, length_weights, tail_samples, isolated_samples):
        """
        ARGS:
            fragment_p (float)
            isolated_p (float)
            length_weights (list): relative weights of fragment lengths 1..max
            tail_samples (tuple): (low, high), in samples
            isolated_samples (int)
        """
        self.fragment_p = float(fragment_p)
        self.isolated_p = float(isolated_p)
        self.length_weights = torch.tensor(length_weights, dtype=torch.float64)
        self.tail_samples = tuple(int(value) for value in tail_samples)
        self.isolated_samples = int(isolated_samples)

    @property
    def active(self):
        """True when any item can be reshaped."""
        return self.fragment_p + self.isolated_p > 0

    def choose(self, n_tokens):
        """Draw how one item of `n_tokens` words is served.

        Uses torch's generator, which Lightning seeds per worker.

        ARGS:
            n_tokens (int): words in the sentence

        RETURNS:
            (start, stop, isolated, tail): the kept words are [start, stop);
            tail is None for a whole sentence (full windows)
        """
        draw = float(torch.rand(1))

        #one word alone, its window exactly the isolated length
        if draw < self.isolated_p:
            pick = int(torch.randint(n_tokens, (1,)))
            return pick, pick + 1, True, self.isolated_samples

        #a run of consecutive words, length weighted as the holdout's, at most n
        if draw < self.isolated_p + self.fragment_p:
            weights = self.length_weights[: min(n_tokens, len(self.length_weights))]
            size = 1 + int(torch.multinomial(weights, 1))
            start = int(torch.randint(n_tokens - size + 1, (1,)))
            low, high = self.tail_samples
            return start, start + size, False, int(torch.randint(low, high + 1, (1,)))

        #the whole sentence, as trial 10 trained
        return 0, n_tokens, False, None


class SequenceWindows(Dataset):
    """Sequences, each word's window sliced from its recording on read.

    ATTRS:
        recordings (ReferenceRecordings)
        sequences (list): of Sequence
        window_samples (int): the full window, and the padded length
        jitter_samples (int): one-sided onset jitter, 0 outside training
        shaping (Shaping or None): training only; None serves every sequence
            exactly as built
    """

    def __init__(self, recordings, sequences, window_samples, jitter_samples=0, shaping=None):
        """
        ARGS:
            recordings (ReferenceRecordings)
            sequences (list): of Sequence
            window_samples (int): 150 at 50 Hz
            jitter_samples (int): 0 is the reference, and always for scoring
            shaping (Shaping or None): never for scoring
        """
        self.recordings = recordings
        self.sequences = sequences
        self.window_samples = window_samples
        self.jitter_samples = jitter_samples
        self.shaping = shaping if shaping is not None and shaping.active else None

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

        #which words are served, and how: the sequence as built, or reshaped
        keep, isolated, tail = slice(None), sequence.isolated, sequence.tail
        if self.shaping is not None:
            start, stop, isolated, tail = self.shaping.choose(len(sequence.onsets))
            keep = slice(start, stop)

        #⚠️ every per-word array is cut by the SAME slice, or targets drift
        onsets = sequence.onsets[keep]
        targets, primary, moses = sequence.targets[keep], sequence.primary[keep], sequence.moses[keep]
        n_tokens = len(onsets)

        #where the MEG stops: nowhere for a whole sentence, the last onset plus
        #the tail for a fragment; an isolated word's own window is set below
        end = n_times
        if tail is not None and not isolated:
            end = min(int(onsets[-1]) + tail, n_times)

        windows = np.zeros((n_tokens, meg.shape[1], self.window_samples), dtype=np.float16)
        lengths = np.zeros(n_tokens, dtype=np.int64)
        for position, onset in enumerate(onsets):
            #jitter shifts WHICH samples are cut; never past either end
            start = int(onset)
            if self.jitter_samples > 0:
                start += int(torch.randint(-self.jitter_samples, self.jitter_samples + 1, (1,)).item())
            start = min(max(start, 0), n_times - 1)

            #an isolated word gets exactly its window; any other word up to the
            #full window, stopped by its sequence's end
            if isolated:
                stop = min(start + tail, n_times)
            else:
                stop = min(start + self.window_samples, end)

            #⚠️ never an empty window: an all-padding window pools to NaN. Only
            #jitter past a fragment's end can cause it
            stop = max(stop, min(start + 1, n_times))

            #[time, sensors] -> [sensors, time], into the padded buffer
            windows[position, :, : stop - start] = np.asarray(meg[start:stop]).T
            lengths[position] = stop - start

        return {
            "windows": torch.from_numpy(windows),
            "lengths": torch.from_numpy(lengths),
            "targets": torch.from_numpy(targets.astype(np.int64)),
            "primary": torch.from_numpy(primary.astype(np.int64)),
            "moses": torch.from_numpy(moses.astype(np.int64)),
            "isolated": torch.full((n_tokens,), isolated, dtype=torch.bool),
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
