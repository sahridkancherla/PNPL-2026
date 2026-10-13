"""Trial 19's training items: one continuous span of a recording, and the words in it.

Trial 10-18's items were one 3 s window per word. Here an item is the MEG
from its first word's onset to a tail after its last, read once, plus each
word's onset within it; the network (`models/continuous.py`) encodes the
span once and reads every word out at its onset. The holdout arrives in
exactly this form: a sentence epoch with word onsets, or a 1 s lone word.

    span start     the first kept word's onset (the holdout has no lead-in),
                   minus a random lead of up to `lead_max_samples` if asked
    span end       the last kept onset + the tail: the sequence's own for a
                   fragment or a lone word; drawn from the holdout's range
                   (`whole_tail` "holdout") or the full window ("full") for
                   a whole sentence
    markers        which onsets the network is told about: all at test; in
                   training, with `marker_drop_p`, only the scored words'
                   (`marker_drop_mode` "scored"), or each marker dropped on
                   its own ("random", trial 21): the second never lets the
                   network learn that a marked word is a list word
    phonemes       the label of every sample of the span, from
                   `cli.build_phoneme_labels`, -1 where unknown
    time masking   `time_mask_p`: blank a few stretches of the span, training

Shaping (fragments, lone words, holdout runs) is trial 13-17's, unchanged:
it chooses WHICH words an item keeps; this file cuts the span around them.
The cache is trial 10's: per-recording robust-scaled, 50 Hz, time-major, so
a span is one contiguous read. The clamp and the baseline are the model's.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..recentre import RecentredRecording
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

    def __init__(self, cache_dir, partition, others_dir=None, phoneme_dir=None):
        """
        ARGS:
            cache_dir (Path or str): subject 0's cache
            partition (str or list): "train", "validation" or "test", pnpl's
                own; a list takes several
            others_dir (Path or str or None): the other subjects' cache
                (cli.build_others), every recording of it added
            phoneme_dir (Path or str or None): per-sample phoneme labels
                (cli.build_phoneme_labels); a recording without a file there
                serves -1 everywhere
        """
        self.cache_dir = Path(cache_dir)
        self.phoneme_dir = Path(phoneme_dir) if phoneme_dir else None
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

    def phonemes(self, recording):
        """The memory-mapped [n_times] int8 phoneme labels of one recording, or None.

        ARGS:
            recording (int): index into `records`

        RETURNS:
            np.memmap or None - None without a phoneme table for it
        """
        if self.phoneme_dir is None:
            return None
        key = ("phon", recording)
        if key not in self._handles:
            path = self.phoneme_dir / f"phon_{self.records[recording]['key']}.npy"
            self._handles[key] = np.load(path, mmap_mode="r") if path.exists() else None
        return self._handles[key]

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


class ContinuousItems(Dataset):
    """Spans of recordings with their words' onsets, for the continuous encoder.

    ATTRS:
        recordings (ReferenceRecordings)
        sequences (list): of Sequence
        window_samples (int): the full per-word window, the "full" tail
        tail_samples (tuple): (low, high) the holdout's tail range
        whole_tail (str): "holdout" or "full", for whole sentences
        lead_max_samples (int): random lead before the first onset, training
        jitter_samples (int): one-sided onset jitter, training only
        marker_drop_p (float): chance an item keeps only scored words' markers
            ("scored"), or the chance each marker is dropped ("random")
        marker_drop_mode (str): "scored" or "random"
        time_mask_p (float): chance an item gets time masking
        time_mask_samples (tuple): (low, high) length of each masked stretch
        shaping (Shaping or None): training only
        training (bool): draws happen only when True
    """

    def __init__(
        self,
        recordings,
        sequences,
        window_samples,
        tail_samples,
        whole_tail="holdout",
        lead_max_samples=0,
        jitter_samples=0,
        marker_drop_p=0.0,
        time_mask_p=0.0,
        time_mask_samples=(5, 15),
        shaping=None,
        training=False,
        marker_drop_mode="scored",
    ):
        """
        ARGS:
            recordings (ReferenceRecordings)
            sequences (list): of Sequence
            window_samples (int): 150 at 50 Hz
            tail_samples (tuple): the holdout's (low, high) after the last onset
            whole_tail (str): "holdout" draws from `tail_samples`; "full"
                keeps `window_samples` after the last onset
            lead_max_samples (int): 0 is the holdout's shape
            jitter_samples (int): 0 is the reference, and always for scoring
            marker_drop_p (float)
            time_mask_p (float)
            time_mask_samples (tuple)
            shaping (Shaping or None)
            training (bool): False serves every sequence exactly as built
            marker_drop_mode (str): "scored" or "random"
        """
        if marker_drop_mode not in ("scored", "random"):
            raise ValueError(f"marker_drop_mode must be 'scored' or 'random', got {marker_drop_mode!r}")
        if whole_tail not in ("holdout", "full"):
            raise ValueError(f"whole_tail must be 'holdout' or 'full', got {whole_tail!r}")
        self.recordings = recordings
        self.sequences = sequences
        self.window_samples = int(window_samples)
        self.tail_samples = (int(tail_samples[0]), int(tail_samples[1]))
        self.whole_tail = whole_tail
        self.lead_max_samples = int(lead_max_samples)
        self.jitter_samples = int(jitter_samples)
        self.marker_drop_p = float(marker_drop_p)
        self.marker_drop_mode = marker_drop_mode
        self.time_mask_p = float(time_mask_p)
        self.time_mask_samples = (int(time_mask_samples[0]), int(time_mask_samples[1]))
        self.shaping = shaping
        self.training = bool(training)

    def __len__(self):
        return len(self.sequences)

    def draw(self, low, high):
        """One integer in [low, high], from torch's generator (worker-safe).

        ARGS:
            low (int), high (int)

        RETURNS:
            int
        """
        return int(torch.randint(low, high + 1, (1,)).item())

    def __getitem__(self, index):
        """One span and its words.

        ARGS:
            index (int): position in `sequences`

        RETURNS:
            dict of tensors:
                signal [306, T] float16, the span; onset [n_tokens] int64,
                sample within the span; marker [n_tokens] bool; targets,
                primary, moses, subject [n_tokens] int64; isolated [n_tokens]
                bool; phonemes [T] int64, -1 where unknown
        """
        sequence = self.sequences[index]
        meg = self.recordings.meg(sequence.recording)
        n_times = meg.shape[0]

        #which words are served: the sequence as built, or reshaped
        keep, isolated, tail = slice(None), sequence.isolated, sequence.tail
        if self.shaping is not None and self.training:
            keep, isolated, tail = self.shaping.choose(sequence.primary, sequence.onsets)

        #⚠️ every per-word array is cut by the SAME positions, or targets drift
        onsets = sequence.onsets[keep].astype(np.int64)
        targets, primary, moses = sequence.targets[keep], sequence.primary[keep], sequence.moses[keep]
        n_tokens = len(onsets)

        #the tail after the last onset: the sequence's own, else a whole sentence's
        if tail is None:
            if self.whole_tail == "holdout" and self.training:
                tail = self.draw(*self.tail_samples)
            else:
                tail = self.window_samples

        #the span: a lead before the first onset only when asked, never for a lone word
        lead = 0
        if self.training and self.lead_max_samples > 0 and not isolated:
            lead = self.draw(0, self.lead_max_samples)
        start = max(int(onsets[0]) - lead, 0)
        end = min(int(onsets[-1]) + int(tail), n_times)

        #⚠️ never an empty span, and every onset inside it
        end = max(end, int(onsets[-1]) + 1)
        signal = np.asarray(meg[start:end]).T.astype(np.float16)
        length = end - start

        #onsets within the span; jitter moves the readout, never past the span
        onset = onsets - start
        if self.training and self.jitter_samples > 0:
            shifts = torch.randint(-self.jitter_samples, self.jitter_samples + 1, (n_tokens,)).numpy()
            onset = np.clip(onset + shifts, 0, length - 1)

        #markers: all; or only the scored words' when the draw says so; or
        #each dropped on its own, so the marked set carries no identity at all
        marker = np.ones(n_tokens, dtype=bool)
        if self.training and self.marker_drop_p > 0:
            if self.marker_drop_mode == "random":
                marker = (torch.rand(n_tokens) >= self.marker_drop_p).numpy()
            elif torch.rand(1).item() < self.marker_drop_p:
                scored = primary >= 0
                if scored.any():
                    marker = scored.copy()

        #time masking: a few stretches blanked, the labels untouched
        if self.training and self.time_mask_p > 0 and torch.rand(1).item() < self.time_mask_p:
            for _ in range(self.draw(1, 2)):
                span = self.draw(*self.time_mask_samples)
                begin = self.draw(0, max(length - span, 0))
                signal[:, begin : begin + span] = 0

        #per-sample phoneme labels of the span, -1 without a table
        phonemes = np.full(length, -1, dtype=np.int64)
        table = self.recordings.phonemes(sequence.recording)
        if table is not None:
            phonemes[:] = np.asarray(table[start:end], dtype=np.int64)

        return {
            "signal": torch.from_numpy(signal),
            "onset": torch.from_numpy(onset.astype(np.int64)),
            "marker": torch.from_numpy(marker),
            "targets": torch.from_numpy(targets.astype(np.int64)),
            "primary": torch.from_numpy(primary.astype(np.int64)),
            "moses": torch.from_numpy(moses.astype(np.int64)),
            "isolated": torch.full((n_tokens,), bool(isolated), dtype=torch.bool),
            "subject": torch.full((n_tokens,), self.recordings.subject_index(sequence.recording), dtype=torch.int64),
            "phonemes": torch.from_numpy(phonemes),
        }


#the per-word fields of an item, concatenated over the batch in this order
TOKEN_FIELDS = ("onset", "marker", "targets", "primary", "moses", "isolated", "subject")


def collate_items(items):
    """Pad the items' spans to one length; flatten their words into one list.

    ARGS:
        items (list): dicts from `ContinuousItems.__getitem__`

    RETURNS:
        dict:
            signal [B, 306, T_max] float16, zero past each length
            lengths [B] int64
            phonemes [B, T_max] int64, -1 on padding
            onset, marker, targets, primary, moses, isolated, subject
                [n_tokens] -- every item's words, item after item
            sequence [n_tokens] -- which item a word came from
            position [n_tokens] -- its place in that item
    """
    n_items = len(items)
    lengths = torch.tensor([item["signal"].shape[1] for item in items], dtype=torch.int64)
    n_channels = items[0]["signal"].shape[0]
    longest = int(lengths.max())

    #zero past each item's real samples: the model masks by `lengths`
    signal = torch.zeros(n_items, n_channels, longest, dtype=torch.float16)
    phonemes = torch.full((n_items, longest), -1, dtype=torch.int64)
    for index, item in enumerate(items):
        length = item["signal"].shape[1]
        signal[index, :, :length] = item["signal"]
        phonemes[index, :length] = item["phonemes"]

    batch = {name: torch.cat([item[name] for item in items], dim=0) for name in TOKEN_FIELDS}
    batch["signal"] = signal
    batch["lengths"] = lengths
    batch["phonemes"] = phonemes

    #the subject layer is chosen per ITEM: every word of an item shares it
    batch["item_subject"] = torch.tensor([int(item["subject"][0]) for item in items], dtype=torch.int64)

    #the grid coordinates, in concatenation order
    batch["sequence"] = torch.cat(
        [torch.full((len(item["targets"]),), index, dtype=torch.int64) for index, item in enumerate(items)]
    )
    batch["position"] = torch.cat([torch.arange(len(item["targets"]), dtype=torch.int64) for item in items])
    return batch
