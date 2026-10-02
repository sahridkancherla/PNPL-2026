"""Read holdout-shaped sequences out of the continuous cache.

Every word's window is sliced from its recording at read time: from the word's
onset (plus jitter, in training) to the earlier of `window_seconds` later and
the sequence's end. Windows are zero-padded to one length; each carries its
valid sample count, which the model uses to mask the padding.

⚠️ The cut-off is what makes this the holdout's problem and not an easier one.
In a holdout sentence the MEG stops 1.1-1.7 s after the last scored word, so a
word near the end has barely more than its 1 s. Training on full 3 s windows
would score validation on signal the holdout never provides.

A batch is FLAT: every token of every sequence, with its sequence index and
position. The model scatters them into a padded [sequences, words] grid only
for the sentence transformer, so padding slots never cross to the GPU as
windows.
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


class SequenceWindows(Dataset):
    """Holdout-shaped sequences, windows sliced from the recordings on read.

    ATTRS:
        recordings (ContinuousRecordings)
        sequences (list): of sequences.Sequence
        window_samples (int): the longest window, and the padded length
        tail_range (tuple): (low, high) samples, for tails drawn per read
        jitter_samples (int): one-sided onset jitter, 0 outside training
    """

    def __init__(self, recordings, sequences, window_samples, tail_range, jitter_samples=0):
        """
        ARGS:
            recordings (ContinuousRecordings): where the MEG is
            sequences (list): of Sequence, from `build_sequences`
            window_samples (int): the longest window after onset
            tail_range (tuple): (low, high) samples kept after the last onset
            jitter_samples (int): one-sided jitter; pass 0 for validation
        """
        self.recordings = recordings
        self.sequences = sequences
        self.window_samples = window_samples
        self.tail_range = tail_range
        self.jitter_samples = jitter_samples

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        """Slice every word's window of one sequence.

        ARGS:
            index (int): position in `sequences`

        RETURNS:
            windows : [n_tokens, n_components, window_samples] float16, zero
                beyond each window's end
            lengths : [n_tokens] int64 valid samples per window
            labels : [n_tokens] int64 class ids
            isolated : [n_tokens] bool
        """
        sequence = self.sequences[index]
        meg = self.recordings.meg(sequence.recording)
        n_times = meg.shape[1]
        n_tokens = len(sequence.onsets)

        #where the sequence's MEG stops: the isolated window, or the last onset
        #plus a tail -- fixed for validation, drawn per read for training
        if sequence.tail is not None:
            tail = sequence.tail
        else:
            tail = int(torch.randint(self.tail_range[0], self.tail_range[1] + 1, (1,)).item())
        end = n_times if sequence.isolated else min(int(sequence.onsets[-1]) + tail, n_times)

        windows = np.zeros((n_tokens, meg.shape[0], self.window_samples), dtype=np.float16)
        lengths = np.zeros(n_tokens, dtype=np.int64)
        for position, onset in enumerate(sequence.onsets):
            #jitter shifts WHICH samples are cut; the cached recording is real
            #signal on both sides, so a shifted window is still real data
            start = int(onset)
            if self.jitter_samples > 0:
                start += int(torch.randint(-self.jitter_samples, self.jitter_samples + 1, (1,)).item())
            start = min(max(start, 0), n_times - 1)

            #an isolated word gets exactly its window; a sentence word gets up
            #to window_samples, stopped by the sequence's end
            if sequence.isolated:
                stop = min(start + sequence.tail, n_times)
            else:
                stop = min(start + self.window_samples, end)

            windows[position, :, : stop - start] = meg[:, start:stop]
            lengths[position] = stop - start

        return (
            torch.from_numpy(windows),
            torch.from_numpy(lengths),
            torch.from_numpy(sequence.labels.astype(np.int64)),
            torch.full((n_tokens,), sequence.isolated, dtype=torch.bool),
        )


def collate_tokens(items):
    """Flatten a list of sequences into one batch of tokens.

    ARGS:
        items (list): (windows, lengths, labels, isolated) per sequence

    RETURNS:
        dict:
            windows : [n_tokens, n_components, window_samples] float16
            lengths : [n_tokens] int64
            labels : [n_tokens] int64
            isolated : [n_tokens] bool
            sequence : [n_tokens] int64, which sequence each token is from
            position : [n_tokens] int64, its place in that sequence
    """
    windows, lengths, labels, isolated = zip(*items)

    #sequence index and position are what the model needs to rebuild the
    #[sequences, words] grid; both follow the concatenation order exactly
    sequence = torch.cat(
        [torch.full((len(item),), index, dtype=torch.int64) for index, item in enumerate(labels)]
    )
    position = torch.cat([torch.arange(len(item), dtype=torch.int64) for item in labels])

    return {
        "windows": torch.cat(windows, dim=0),
        "lengths": torch.cat(lengths, dim=0),
        "labels": torch.cat(labels, dim=0),
        "isolated": torch.cat(isolated, dim=0),
        "sequence": sequence,
        "position": position,
    }
