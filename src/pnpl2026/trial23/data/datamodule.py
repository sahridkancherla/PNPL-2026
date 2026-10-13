"""LightningDataModule over trial 10's reference cache, for trial 17 (trial 16's).

Trains on pnpl's train partition -- whole sentences, some reshaped like the
holdout when the config asks -- and validates on its validation partition
under `selection_protocol`, the fragment protocol by default, so best.ckpt and
early stopping follow the protocol that looks like the leaderboard. The test
partition is never touched here; only `cli.evaluate` reads it.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import PROTOCOLS
from .dataset import ContinuousItems, ReferenceRecordings, Shaping, collate_items
from .sequences import (
    CHAIN38_MAX_WORDS,
    CHAIN38_MIN_WORDS,
    chain_sequences,
    fragment_sequences,
    holdout_chains,
    isolated_sequences,
    reference_sequences,
    sparse_sequences,
)

#fragment- and sparse-protocol sequences are drawn once, from a seed of their
#own, so every run and seed is scored on identical windows
HOLDOUT_SEED = 20260930


def build_partition(cfg, table, partition, protocol, jitter_samples=0, shaping=None, others_dir=None, training=False):
    """One partition's dataset under one protocol.

    Shared by training and `cli.evaluate`, so both cut sequences identically.

    ARGS:
        cfg (Trial23Config)
        table (TargetTable or NoTargets): for each word's t5 row
        partition (str or list): "train", "validation" or "test"; a list
            takes several (all-data training)
        protocol (str): "reference", "fragment", "sparse", "isolated", "chains" or "chains38"
        jitter_samples (int): training only
        shaping (Shaping or None): training only
        others_dir (Path or None): the other subjects' cache, training only
        training (bool): draws (tails, leads, markers, masks, shaping) happen
            only for the training set

    RETURNS:
        ContinuousItems
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {protocol!r}; known: {PROTOCOLS}")
    recordings = ReferenceRecordings(cfg.resolved_cache_dir(), partition, others_dir, cfg.resolved_phoneme_dir())

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
    elif protocol == "fragment":
        sequences = fragment_sequences(
            recordings.tables,
            recordings.n_times,
            rows,
            float(cfg.sfreq),
            cfg.holdout_max_tokens,
            cfg.fragment_min_words,
            cfg.fragment_max_gap_seconds,
            cfg.isolated_fraction,
            cfg.isolated_seconds,
            cfg.tail_seconds,
            HOLDOUT_SEED,
        )
    elif protocol == "isolated":
        sequences = isolated_sequences(recordings.tables, recordings.n_times, rows, cfg.samples(cfg.isolated_seconds))
    elif protocol in ("chains", "chains38"):
        #trial 22: the holdout's own token form, for embedding dumps only;
        #trial 23's chains38 keeps runs of 3-8 words, as the research note asks
        short = protocol == "chains38"
        sequences = chain_sequences(
            recordings.tables,
            recordings.n_times,
            rows,
            float(cfg.sfreq),
            CHAIN38_MIN_WORDS if short else cfg.fragment_min_words,
            CHAIN38_MAX_WORDS if short else cfg.holdout_max_tokens,
            cfg.fragment_max_gap_seconds,
            cfg.tail_seconds,
            HOLDOUT_SEED,
        )
    else:
        sequences = sparse_sequences(
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

    #the holdout's tail range and the span settings, in this cache's samples
    tail_low, tail_high = (cfg.samples(value) for value in cfg.tail_seconds)
    mask_low, mask_high = (cfg.samples(value) for value in cfg.time_mask_seconds)
    return ContinuousItems(
        recordings,
        sequences,
        window_samples,
        (tail_low, tail_high),
        whole_tail=cfg.whole_tail,
        lead_max_samples=cfg.samples(cfg.lead_max_seconds),
        jitter_samples=jitter_samples,
        marker_drop_p=cfg.marker_drop_p,
        marker_drop_mode=cfg.marker_drop_mode,
        time_mask_p=cfg.time_mask_p,
        time_mask_samples=(mask_low, max(mask_high, mask_low)),
        shaping=shaping,
        training=training,
    )


def training_shaping(cfg):
    """The holdout-shaping a run's config asks for, or None.

    ARGS:
        cfg (Trial23Config)

    RETURNS:
        Shaping or None
    """
    if cfg.fragment_p + cfg.isolated_p + cfg.run_p <= 0:
        return None
    tail = tuple(cfg.samples(value) for value in cfg.tail_seconds)
    return Shaping(
        cfg.fragment_p,
        cfg.isolated_p,
        cfg.fragment_lengths,
        tail,
        cfg.samples(cfg.isolated_seconds),
        run_p=cfg.run_p,
        run_max_skipped=cfg.run_max_skipped,
        run_max_gap_samples=cfg.samples(cfg.fragment_max_gap_seconds),
        run_min_words=cfg.fragment_min_words,
    )


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
    alone = sum(1 for s in dataset.sequences if s.isolated)
    return (
        f"[data] {name}: {sizes.sum():,} tokens in {len(sizes):,} sequences "
        f"(median {np.median(sizes):.0f}, max {sizes.max()}, {alone:,} alone); {targets:,} with a t5 "
        f"target, {scored:,} scored"
    )


class Trial23DataModule(L.LightningDataModule):
    """Training sentences, shaped or not, and validation under the selection protocol.

    ATTRS:
        cfg: the run config
        table (TargetTable)
        train_set (SequenceWindows)
        val_set (SequenceWindows)
    """

    def __init__(self, cfg, table):
        """
        ARGS:
            cfg (Trial23Config)
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
        shaping = training_shaping(self.cfg)
        #all-data: subject 0's chosen partitions, and the others' cache if named
        self.train_set = build_partition(
            self.cfg,
            self.table,
            list(self.cfg.train_partitions),
            self.cfg.training_protocol,
            jitter,
            shaping,
            self.cfg.resolved_others_cache_dir(),
            training=True,
        )
        self.val_set = build_partition(self.cfg, self.table, "validation", self.cfg.selection_protocol)
        print(describe("train", self.train_set))
        if shaping is not None:
            print(
                f"[data] train shaping: {100 * shaping.fragment_p:.0f}% fragments, "
                f"{100 * shaping.isolated_p:.0f}% isolated, {100 * shaping.run_p:.0f}% holdout runs, "
                f"the rest whole sentences"
            )

        #how many sentences can supply a holdout run at all: the rest stay whole
        if shaping is not None and shaping.run_p > 0:
            possible = sum(
                1
                for sequence in self.train_set.sequences
                if max((len(chain) for chain in holdout_chains(sequence.primary, sequence.onsets,
                        shaping.run_max_skipped, shaping.run_max_gap_samples)), default=0) >= shaping.run_min_words
            )
            share = possible / max(len(self.train_set.sequences), 1)
            print(f"[data] holdout runs of {shaping.run_min_words}+ words possible in {100 * share:.1f}% of training "
                  f"sentences: about {100 * shaping.run_p * share:.0f}% of items will be runs")
        print(describe(f"validation ({self.cfg.selection_protocol} protocol)", self.val_set))

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
            collate_fn=collate_items,
            #a last batch of two sentences is a much easier SigLIP problem
            drop_last=shuffle,
            **extra,
        )

    def train_dataloader(self):
        """Training sequences, reshuffled (and reshaped) every epoch."""
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        """Validation sequences, in their fixed order."""
        #⚠️ never shuffled: evaluation order is fixed
        return self._loader(self.val_set, shuffle=False)
