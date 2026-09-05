"""LightningDataModule over LibriBrain100 for the deep track."""

from __future__ import annotations

import lightning as L
import torch
from torch.utils.data import DataLoader, Subset

from pnpl.datasets import LibriBrain100
from pnpl.tasks import WordClassification

from ..config import DATA_DIR, Config
from .datasets import CompetitionWordDataset
from .splits import session_split


class LibriBrainDataModule(L.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.train_set: Subset | None = None
        self.val_set: Subset | None = None
        self.class_weights: torch.Tensor | None = None

    def prepare_data(self) -> None:
        """Download only -- run on the login node via cli/prepare.py."""

    def setup(self, stage: str | None = None) -> None:
        if self.train_set is not None:
            return

        base = LibriBrain100(
            data_path=DATA_DIR,
            task=WordClassification(tmin=self.cfg.tmin, tmax=self.cfg.tmax),
            partition="train",
        )
        full = CompetitionWordDataset(base)
        train_idx, val_idx = session_split(full.sessions(), self.cfg.val_sessions)
        self.train_set = Subset(full, train_idx)
        self.val_set = Subset(full, val_idx)

        # Inverse-frequency weights over the training split only. The 50 words
        # are far from balanced and the metric is balanced accuracy.
        labels = full.labels()
        counts = torch.zeros(self.cfg.n_classes, dtype=torch.float)
        for i in train_idx:
            counts[labels[i]] += 1
        self.class_weights = torch.where(
            counts > 0, counts.sum() / (counts * self.cfg.n_classes), torch.zeros_like(counts)
        )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)
