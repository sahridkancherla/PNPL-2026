"""LightningDataModule over the trial-1 cache.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.

The train/val split is NOT made here. It was made by whole session when the
cache was built, and is recorded per shard in the manifest; this module only
honours it. Re-deriving a split at train time would let a config edit put a
session on both sides of the line, which reads as a validation score well
above the leaderboard and gives no warning at all.
"""

from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader

from .cache_dataset import CachedWindows


def class_weights_from(labels, n_classes, power=0.5):
    """Inverse-frequency class weights, tapered by `power`, mean-normalised.

    The vocabulary spans 121x between `the` and `people`. Full inverse
    frequency (power=1.0) reproduces that 121x in the loss, so a single rare
    token in a batch dominates the gradient and training becomes a lottery on
    batch composition. The square root (power=0.5) keeps the direction of the
    correction while compressing its range to ~11x.

    ⚠️ power=0.0 is unweighted cross-entropy, which under a BALANCED metric
    systematically favours frequent words. Neither end is obviously right --
    this is a knob to sweep, not a solved question.

    ARGS:
        labels (np.ndarray): [n_windows] class ids of the TRAINING split only
        n_classes (int): 50
        power (float): 0.0 unweighted, 0.5 sqrt-inverse, 1.0 full inverse

    RETURNS:
        weights : [n_classes] float32 tensor, mean 1 over present classes
    """
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)

    #an absent class must not become an infinite weight; it gets zero instead,
    #which is what cross_entropy needs for a class it will never see
    present = counts > 0
    weights = np.zeros(n_classes, dtype=np.float64)
    weights[present] = (1.0 / counts[present]) ** power

    #normalise to mean 1 over present classes, so `power` changes the SPREAD
    #of the weights without also changing the effective learning rate
    weights[present] /= weights[present].mean()

    return torch.from_numpy(weights).float()


class Trial1DataModule(L.LightningDataModule):
    """Train and validation loaders over the cached, whitened windows.

    ATTRS:
        cfg: the run config
        train_set (CachedWindows): jittered, shuffled
        val_set (CachedWindows): un-jittered, fixed order
        class_weights (torch.Tensor): [n_classes], from the training split
    """

    def __init__(self, cfg):
        """
        ARGS:
            cfg (Trial1Config): the merged run config
        """
        super().__init__()
        self.cfg = cfg
        self.train_set = None
        self.val_set = None
        self.class_weights = None

    def setup(self, stage=None):
        """Index both splits and derive the class weights.

        ARGS:
            stage (str or None): Lightning's stage hint, unused -- both splits
                are cheap to index, so both are always built
        """
        #Lightning calls setup once per stage; indexing twice is wasted I/O
        if self.train_set is not None:
            return

        cache_dir = self.cfg.resolved_cache_dir()

        self.train_set = CachedWindows(
            cache_dir,
            split="train",
            tmin=self.cfg.tmin,
            tmax=self.cfg.tmax,
            jitter_seconds=self.cfg.jitter_seconds,
        )

        #⚠️ validation takes NO jitter. The score has to come from the same
        #fixed window the holdout will be served, or the two are not comparable
        self.val_set = CachedWindows(
            cache_dir,
            split="val",
            tmin=self.cfg.tmin,
            tmax=self.cfg.tmax,
            jitter_seconds=0.0,
        )

        self.class_weights = class_weights_from(
            self.train_set.labels,
            self.cfg.n_classes,
            self.cfg.class_weight_power,
        )
        self._report()

    def _report(self):
        """Print what the cache and split actually contain.

        Without this the run is opaque: a cache built with the wrong bounds, or
        a split that lost half the classes, still produces a healthy loss curve.
        """
        train, val = self.train_set, self.val_set
        total = len(train) + len(val)

        print(f"[data] cache {self.cfg.resolved_cache_dir()}")
        print(
            f"[data] crop: {train.n_model_samples} samples at offset "
            f"{train.core_offset}, jitter +/-{train.max_jitter} samples"
        )
        if self.cfg.jitter_seconds > 0 and train.max_jitter == 0:
            print(
                "[data] WARNING: jitter requested but the cached margin allows "
                "none. Rebuild the cache with wider bounds, or expect no "
                "augmentation at all."
            )

        print(
            f"[data] train {len(train)} | val {len(val)} "
            f"({len(val) / max(total, 1):.1%})"
        )
        print(
            f"[data] train sessions {sorted(set(train.sessions))} | "
            f"val sessions {sorted(set(val.sessions))}"
        )

        #the metric averages only over classes PRESENT in the labels, so a
        #validation split missing classes is not comparable to the leaderboard
        n_val_classes = len(set(val.labels.tolist()))
        print(
            f"[data] classes present: train "
            f"{len(set(train.labels.tolist()))}/{self.cfg.n_classes}, "
            f"val {n_val_classes}/{self.cfg.n_classes}"
        )
        if n_val_classes < self.cfg.n_classes:
            print(
                f"[data] WARNING: {self.cfg.n_classes - n_val_classes} class(es) "
                "absent from validation -- BAcc averages over present classes "
                "only, so this number is not the leaderboard's."
            )

        #the rarest class sets the noise floor on every BAcc comparison
        counts = np.bincount(val.labels, minlength=self.cfg.n_classes)
        present = counts[counts > 0]
        if len(present):
            rarest = int(present.min())
            print(
                f"[data] rarest validation class has {rarest} example(s); "
                f"per-class recall SE ~{0.5 / max(rarest, 1) ** 0.5:.3f}. "
                "Treat BAcc differences below this as noise."
            )

    def _loader(self, dataset, shuffle):
        """Build a DataLoader with the settings both splits share.

        ARGS:
            dataset (CachedWindows): the split to wrap
            shuffle (bool): True for training only

        RETURNS:
            a torch DataLoader
        """
        #each read touches a memory-mapped shard, so the loader is I/O bound;
        #persistent workers avoid re-opening every handle once per epoch
        extra = {}
        if self.cfg.num_workers > 0:
            extra = {"persistent_workers": True, "prefetch_factor": 4}
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            **extra,
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self):
        #⚠️ never shuffled: evaluation reports depend on this order
        return self._loader(self.val_set, shuffle=False)
