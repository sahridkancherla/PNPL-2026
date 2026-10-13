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

Trial 17 adds the HOLDOUT RUN: scored words only, neighbours directly adjacent
or with one unscored word between, the MEG cut a tail after the last onset --
the holdout's own runs (`Shaping.holdout_run`, `sequences.holdout_chains`).

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

from ...boundary import boundary_features
from ...recentre import RecentredRecording
from .sequences import holdout_chains

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
    """pnpl partitions of the cache, and optionally the other subjects': recordings and word tables.

    ATTRS:
        cache_dir (Path): subject 0's cache
        sample_rate (float): 50 Hz
        records (list): the manifests' entries, each with its `cache_dir` and
            `subject_index` (0 for subject 0, 1.. for the others)
        tables (list): word tables, as dicts of arrays
        n_times (list): samples per recording
    """

    def __init__(self, cache_dir, partition, others_dir=None):
        """
        ARGS:
            cache_dir (Path or str): subject 0's cache
            partition (str or list): "train", "validation" or "test", pnpl's
                own; a list takes several
            others_dir (Path or str or None): the other subjects' cache
                (cli.build_others), every recording of it added
        """
        self.cache_dir = Path(cache_dir)
        manifest = load_manifest(cache_dir)
        self.sample_rate = float(manifest["sample_rate"])
        partitions = [partition] if isinstance(partition, str) else list(partition)

        #the partition was fixed at build time from pnpl's manifest; honour it
        self.records = [
            dict(record, cache_dir=str(self.cache_dir), subject_index=0)
            for record in manifest["recordings"]
            if record["partition"] in partitions
        ]
        if not self.records:
            raise ValueError(
                f"No {partitions} recordings in {cache_dir}. Download them, then rebuild:\n"
                f"    python -m pnpl2026.cli.prepare --partition {partitions[0]} --subjects deep"
            )

        #all-data: every other subject's recording, its subject a layer of its own
        if others_dir is not None:
            others = load_manifest(others_dir)
            if float(others["sample_rate"]) != self.sample_rate:
                raise ValueError(f"{others_dir} is {others['sample_rate']} Hz; subject 0's cache is {self.sample_rate}.")
            subjects = list(others["subjects"])
            self.records += [
                dict(record, cache_dir=str(others_dir), subject_index=1 + subjects.index(record["subject"]))
                for record in others["recordings"]
            ]

        self.tables = []
        for record in self.records:
            with np.load(Path(record["cache_dir"]) / f"words_{record['key']}.npz") as stored:
                self.tables.append({name: stored[name] for name in stored.files})
        self.n_times = [int(record["n_times"]) for record in self.records]
        self._handles = {}

    def subject_index(self, recording):
        """The subject layer one recording's windows go through.

        ARGS:
            recording (int): index into `records`

        RETURNS:
            int: 0 for subject 0
        """
        return int(self.records[recording]["subject_index"])

    def meg(self, recording):
        """The memory-mapped [n_times, 306] array of one recording.

        ARGS:
            recording (int): index into `records`

        RETURNS:
            np.memmap, float16
        """
        if recording not in self._handles:
            record = self.records[recording]
            self._handles[recording] = np.load(Path(record["cache_dir"]) / f"meg_{record['key']}.npy", mmap_mode="r")

        #trial 23's P8b: a recording re-centred on the training recordings' covariance
        transforms = getattr(self, "transforms", None)
        if transforms is not None and recording in transforms:
            return RecentredRecording(self._handles[recording], transforms[recording])
        return self._handles[recording]

    def __getstate__(self):
        """Drop open memory maps when pickled into a worker; it reopens its own."""
        state = dict(self.__dict__)
        state["_handles"] = {}
        return state


class Shaping:
    """How training items are reshaped to look like the holdout.

    Four kinds of item, drawn per sentence per epoch:

        isolated       one word alone, exactly the isolated window (G2's)
        fragment       a run of consecutive words of ANY kind (G1's)
        holdout run    trial 17: scored words only, neighbours at most
                       `run_max_skipped` unscored words and
                       `run_max_gap_samples` apart -- the holdout's own runs
        whole          the sentence as built, full windows

    ATTRS:
        fragment_p (float): chance an item becomes a fragment
        isolated_p (float): chance an item becomes one word alone
        run_p (float): chance an item becomes a holdout run
        length_weights (torch.Tensor): [max_words] weights of lengths 1..max
        tail_samples (tuple): (low, high) MEG kept after a run's last onset
        isolated_samples (int): an isolated word's window
        run_max_skipped (int): unscored words allowed between run neighbours
        run_max_gap_samples (int): longest onset gap within a run
        run_min_words (int): shortest run served; a sentence without one
            stays whole (the holdout's runs are 4-9 words, 152 of 156)
    """

    def __init__(
        self,
        fragment_p,
        isolated_p,
        length_weights,
        tail_samples,
        isolated_samples,
        run_p=0.0,
        run_max_skipped=1,
        run_max_gap_samples=40,
        run_min_words=4,
    ):
        """
        ARGS:
            fragment_p (float)
            isolated_p (float)
            length_weights (list): relative weights of run lengths 1..max
            tail_samples (tuple): (low, high), in samples
            isolated_samples (int)
            run_p (float): holdout runs' share
            run_max_skipped (int): 1, the holdout's
            run_max_gap_samples (int): 0.8 s at the cache's rate
            run_min_words (int): 4, the holdout's shortest common length
        """
        self.fragment_p = float(fragment_p)
        self.isolated_p = float(isolated_p)
        self.run_p = float(run_p)
        self.length_weights = torch.tensor(length_weights, dtype=torch.float64)
        self.tail_samples = tuple(int(value) for value in tail_samples)
        self.isolated_samples = int(isolated_samples)
        self.run_max_skipped = int(run_max_skipped)
        self.run_max_gap_samples = int(run_max_gap_samples)
        self.run_min_words = int(run_min_words)

    @property
    def active(self):
        """True when any item can be reshaped."""
        return self.fragment_p + self.isolated_p + self.run_p > 0

    def draw_tail(self):
        """One tail length, uniform on the holdout's range.

        RETURNS:
            int, samples
        """
        low, high = self.tail_samples
        return int(torch.randint(low, high + 1, (1,)))

    def holdout_run(self, primary, onsets):
        """Positions of one holdout-shaped run of scored words, or None.

        The length is drawn from the holdout's histogram between
        `run_min_words` and the longest chain the sentence has; then one window
        of that many consecutive chain members, uniform over every window of
        every long-enough chain. A sentence whose chains are all shorter than
        `run_min_words` has no holdout run.

        ARGS:
            primary (np.ndarray): [n_tokens] scored columns, -1 unscored
            onsets (np.ndarray): [n_tokens] onset samples

        RETURNS:
            np.ndarray of token positions, or None without a long-enough chain
        """
        chains = holdout_chains(primary, onsets, self.run_max_skipped, self.run_max_gap_samples)
        longest = max((len(chain) for chain in chains), default=0)
        if longest < self.run_min_words:
            return None

        #a length the sentence can supply, weighted as the holdout's lengths are
        weights = self.length_weights[self.run_min_words - 1 : min(longest, len(self.length_weights))]
        size = self.run_min_words + int(torch.multinomial(weights, 1))

        #every placement of `size` consecutive chain members, one drawn uniformly
        placements = [(chain, start) for chain in chains for start in range(len(chain) - size + 1)]
        chain, start = placements[int(torch.randint(len(placements), (1,)))]
        return chain[start : start + size]

    def choose(self, primary, onsets):
        """Draw how one sentence is served.

        Uses torch's generator, which Lightning seeds per worker.

        ARGS:
            primary (np.ndarray): [n_tokens] scored columns, -1 unscored
            onsets (np.ndarray): [n_tokens] onset samples, ascending

        RETURNS:
            (keep, isolated, tail): keep [n_kept] token positions, ascending;
            tail is None for a whole sentence (full windows)
        """
        n_tokens = len(primary)
        draw = float(torch.rand(1))

        #one word alone, its window exactly the isolated length
        if draw < self.isolated_p:
            pick = int(torch.randint(n_tokens, (1,)))
            return np.array([pick]), True, self.isolated_samples

        #a run of consecutive words of any kind, length weighted as the holdout's
        if draw < self.isolated_p + self.fragment_p:
            weights = self.length_weights[: min(n_tokens, len(self.length_weights))]
            size = 1 + int(torch.multinomial(weights, 1))
            start = int(torch.randint(n_tokens - size + 1, (1,)))
            return np.arange(start, start + size), False, self.draw_tail()

        #a holdout run; a sentence without a long-enough chain stays whole
        if draw < self.isolated_p + self.fragment_p + self.run_p:
            keep = self.holdout_run(primary, onsets)
            if keep is not None:
                return keep, False, self.draw_tail()

        #the whole sentence, as trial 10 trained
        return np.arange(n_tokens), False, None


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
                [n_tokens] int64; isolated [n_tokens] bool; subject
                [n_tokens] int64, the subject layer (0 is subject 0)
        """
        sequence = self.sequences[index]
        meg = self.recordings.meg(sequence.recording)
        n_times = meg.shape[0]

        #which words are served, and how: the sequence as built, or reshaped
        keep, isolated, tail = slice(None), sequence.isolated, sequence.tail
        if self.shaping is not None:
            keep, isolated, tail = self.shaping.choose(sequence.primary, sequence.onsets)

        #⚠️ every per-word array is cut by the SAME positions, or targets drift
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
            "subject": torch.full((n_tokens,), self.recordings.subject_index(sequence.recording), dtype=torch.int64),
            #trial 23's P6: each token's place and gaps, from the served onsets (before jitter)
            "boundary": torch.from_numpy(boundary_features(np.asarray(onsets, dtype=np.float64) / self.recordings.sample_rate)),
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
