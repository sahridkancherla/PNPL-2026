"""LightningDataModule over the continuous cache, for trial 11: sentence streams.

The train/val split is NOT made here. It is trial 1's, recorded per recording
in the continuous cache's manifest; this module only honours it.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ContinuousRecordings, SequenceStreams, collate_streams
from .sequences import build_sequences

#validation sequences are drawn once, from a seed of their own, so every run
#-- whatever its training seed -- is scored on the identical windows
VALIDATION_SEED = 20260930


def class_weights_from(labels, n_classes, power=0.5):
    """Inverse-frequency class weights, tapered by `power`, mean-normalised.

    ARGS:
        labels (np.ndarray): [n_tokens] class ids of the TRAINING split only
        n_classes (int): 50
        power (float): 0.0 unweighted, 0.5 sqrt-inverse, 1.0 full inverse

    RETURNS:
        weights : [n_classes] float32 tensor, mean 1 over present classes
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)

    #an absent class gets zero weight, not an infinite one
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float64)
    weights[present] = (1.0 / counts[present]) ** power

    #mean 1 over present classes, so `power` moves only the spread
    weights[present] /= weights[present].mean()
    return torch.from_numpy(weights).float()


class Trial11DataModule(L.LightningDataModule):
    """Holdout-shaped train and validation sequences over the continuous cache.

    ATTRS:
        cfg: the run config
        train_set (SequenceStreams): jittered, fresh tails every read
        val_set (SequenceStreams): no jitter, fixed tails, fixed order
        class_weights (torch.Tensor): [n_classes], from the training tokens
    """

    def __init__(self, cfg):
        """
        ARGS:
            cfg (Trial11Config): the merged run config
        """
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.class_weights = None

    def setup(self, stage=None):
        """Index both splits, build their sequences, derive class weights.

        ARGS:
            stage (str or None): Lightning's stage hint, unused
        """
        #Lightning calls setup once per stage; indexing twice is wasted I/O
        if self.train_set is not None:
            return

        cfg = self.cfg
        cache_dir = cfg.resolved_continuous_dir()
        sample_rate = float(cfg.sfreq)
        stream_samples = int(round(cfg.window_seconds * sample_rate))
        tail_range = tuple(int(round(value * sample_rate)) for value in cfg.tail_seconds)

        splits = {}
        for split in ("train", "val"):
            recordings = ContinuousRecordings(cache_dir, split)

            #⚠️ the cache must be at the sample rate the model is built for,
            #or every window and every frame count is silently the wrong length
            if recordings.sample_rate != sample_rate:
                raise ValueError(
                    f"Cache is {recordings.sample_rate} Hz, config sfreq is {sample_rate}."
                )

            sequences = build_sequences(
                recordings.tables,
                recordings.n_times,
                sample_rate,
                cfg.max_tokens,
                cfg.isolated_fraction,
                cfg.isolated_seconds,
                cfg.tail_seconds,
                #validation tails are fixed once; training draws fresh ones
                fixed_tails=(split == "val"),
                seed=cfg.seed if split == "train" else VALIDATION_SEED,
                max_onset_span_seconds=cfg.max_onset_span_seconds,
            )
            splits[split] = (recordings, sequences)

        train_recordings, train_sequences = splits["train"]
        val_recordings, val_sequences = splits["val"]

        #⚠️ the chance-line control: permute every TRAINING token's label
        #across the whole split, keeping class counts. Validation untouched
        if cfg.shuffle_labels:
            flat = np.concatenate([sequence.labels for sequence in train_sequences])
            flat = np.random.default_rng(cfg.seed).permutation(flat)
            start = 0
            for sequence in train_sequences:
                sequence.labels = flat[start : start + len(sequence.labels)]
                start += len(sequence.labels)

        self.train_set = SequenceStreams(
            train_recordings,
            train_sequences,
            stream_samples,
            tail_range,
            jitter_samples=int(round(cfg.jitter_seconds * sample_rate)),
        )

        #⚠️ validation takes NO jitter: it must be served like the holdout
        self.val_set = SequenceStreams(val_recordings, val_sequences, stream_samples, tail_range)

        train_labels = np.concatenate([sequence.labels for sequence in train_sequences])
        self.class_weights = class_weights_from(train_labels, cfg.n_classes, cfg.class_weight_power)
        self._report(train_sequences, val_sequences)

    def _report(self, train_sequences, val_sequences):
        """Print what the sequences look like, next to the holdout's shape.

        ARGS:
            train_sequences (list): of Sequence
            val_sequences (list): of Sequence
        """
        if self.cfg.shuffle_labels:
            print("[data] WARNING: TRAINING LABELS SHUFFLED -- this run is the chance line")

        for name, sequences in (("train", train_sequences), ("val", val_sequences)):
            sizes = np.array([len(s.labels) for s in sequences if not s.isolated])
            n_isolated = sum(s.isolated for s in sequences)
            n_tokens = int(sizes.sum()) + n_isolated
            print(
                f"[data] {name}: {n_tokens:,} words in {len(sizes):,} sequences "
                f"(median {np.median(sizes):.0f}, max {sizes.max()}) + {n_isolated:,} "
                f"isolated ({n_isolated / max(n_tokens, 1):.1%}) -- holdout: median 5, "
                f"max 9, 9.6% isolated"
            )

        #the metric averages over classes PRESENT, so a missing class moves it
        val_labels = np.concatenate([s.labels for s in val_sequences])
        counts = np.bincount(val_labels, minlength=self.cfg.n_classes)
        present = counts[counts > 0]
        print(
            f"[data] validation classes present {len(present)}/{self.cfg.n_classes}; "
            f"rarest has {present.min()} words, per-class recall SE "
            f"~{0.5 / max(present.min(), 1) ** 0.5:.3f}"
        )

    def _loader(self, dataset, shuffle):
        """A DataLoader with the settings both splits share.

        ARGS:
            dataset (SequenceStreams): the split to wrap
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
            collate_fn=collate_streams,
            #a last batch of two sequences is a much easier SigLIP problem
            drop_last=shuffle,
            **extra,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        #⚠️ never shuffled: evaluation reports depend on this order
        return self._loader(self.val_set, shuffle=False)
