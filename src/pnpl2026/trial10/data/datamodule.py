"""LightningDataModule over the reference cache, for trial 10.

Trains on pnpl's train partition and validates on its validation partition,
both under the reference protocol -- the checkpoint is selected as the
competition paper's was. The test partition is never touched here; only
`cli.evaluate` reads it, once, at the end.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ReferenceRecordings, SequenceWindows, collate_tokens
from .sequences import holdout_sequences, reference_sequences

#holdout-protocol sequences are drawn once, from a seed of their own, so every
#run and seed is scored on identical windows
HOLDOUT_SEED = 20260930


def build_partition(cfg, table, partition, protocol, jitter_samples=0):
    """One partition's dataset under one protocol.

    Shared by training and `cli.evaluate`, so both cut sequences identically.

    ARGS:
        cfg (Trial10Config)
        table (TargetTable): for each word's t5 row
        partition (str): "train", "validation" or "test"
        protocol (str): "reference" or "holdout"
        jitter_samples (int): training only

    RETURNS:
        SequenceWindows
    """
    recordings = ReferenceRecordings(cfg.resolved_cache_dir(), partition)

    #⚠️ the cache must be at the rate the config describes, or every window
    #and every baseline is silently the wrong length
    if recordings.sample_rate != float(cfg.sfreq):
        raise ValueError(f"Cache is {recordings.sample_rate} Hz, config sfreq is {cfg.sfreq}.")

    rows = [table.rows(words["word"]) for words in recordings.tables]
    window_samples = cfg.samples(cfg.window_seconds)

    if protocol == "reference":
        sequences = reference_sequences(
            recordings.tables, recordings.n_times, rows, window_samples, cfg.max_tokens, cfg.token_set
        )
    elif protocol == "holdout":
        sequences = holdout_sequences(
            recordings.tables,
            recordings.n_times,
            rows,
            float(cfg.sfreq),
            cfg.holdout_max_tokens,
            cfg.isolated_fraction,
            cfg.isolated_seconds,
            cfg.tail_seconds,
            HOLDOUT_SEED,
        )
    else:
        raise ValueError(f"Unknown protocol {protocol!r}")

    return SequenceWindows(recordings, sequences, window_samples, jitter_samples)


def describe(name, dataset):
    """One line on what a dataset holds.

    ARGS:
        name (str): label for the line
        dataset (SequenceWindows)

    RETURNS:
        str
    """
    sizes = np.array([len(s.onsets) for s in dataset.sequences])
    scored = sum(int((s.primary >= 0).sum()) for s in dataset.sequences)
    targets = sum(int((s.targets >= 0).sum()) for s in dataset.sequences)
    return (
        f"[data] {name}: {sizes.sum():,} tokens in {len(sizes):,} sequences "
        f"(median {np.median(sizes):.0f}, max {sizes.max()}); {targets:,} with a t5 target, "
        f"{scored:,} scored"
    )


class Trial10DataModule(L.LightningDataModule):
    """Train and validation sequences, both under the training protocol.

    ATTRS:
        cfg: the run config
        table (TargetTable)
        train_set (SequenceWindows)
        val_set (SequenceWindows)
    """

    def __init__(self, cfg, table):
        """
        ARGS:
            cfg (Trial10Config)
            table (TargetTable)
        """
        super().__init__()
        self.cfg = cfg
        self.table = table
        self.train_set = None
        self.val_set = None

    def setup(self, stage=None):
        """Index both partitions.

        ARGS:
            stage (str or None): Lightning's stage hint, unused
        """
        #Lightning calls setup once per stage; indexing twice is wasted I/O
        if self.train_set is not None:
            return

        jitter = self.cfg.samples(self.cfg.jitter_seconds)
        self.train_set = build_partition(self.cfg, self.table, "train", "reference", jitter)
        self.val_set = build_partition(self.cfg, self.table, "validation", "reference")
        print(describe("train", self.train_set))
        print(describe("validation", self.val_set))

    def _loader(self, dataset, shuffle):
        """A DataLoader with the settings both partitions share.

        ARGS:
            dataset (SequenceWindows)
            shuffle (bool): True for training only

        RETURNS:
            a torch DataLoader
        """
        extra = {}
        if self.cfg.num_workers > 0:
            extra = {"persistent_workers": True, "prefetch_factor": 4}
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_tokens,
            #a last batch of two sentences is a much easier SigLIP problem
            drop_last=shuffle,
            **extra,
        )

    def train_dataloader(self):
        """Training sequences, reshuffled every epoch."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        """Validation sequences, in their fixed order."""
        #⚠️ never shuffled: evaluation order is fixed
        return self._loader(self.val_set, shuffle=False)
