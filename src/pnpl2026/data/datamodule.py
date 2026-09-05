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

        self._report(base, full, train_idx, val_idx, labels, counts)

    def _report(self, base, full, train_idx, val_idx, labels, counts) -> None:
        """Print what the filter and split actually did.

        Without this the run is opaque: a broken vocabulary filter or a
        degenerate split still produces a healthy-looking loss curve.
        """
        sessions = full.sessions()
        held = sorted({sessions[i] for i in val_idx})
        val_classes = {labels[i] for i in val_idx}
        n_train_classes = int((counts > 0).sum())

        print(
            f"[data] {len(full)} scored samples kept of {len(base.samples)} "
            f"({100 * len(full) / max(len(base.samples), 1):.1f}%)"
        )
        print(f"[data] sessions: {sorted(set(sessions))} | held out for val: {held}")
        print(f"[data] train {len(train_idx)} | val {len(val_idx)}")
        print(
            f"[data] classes present: train {n_train_classes}/{self.cfg.n_classes}, "
            f"val {len(val_classes)}/{self.cfg.n_classes}"
        )
        if len(val_classes) < self.cfg.n_classes:
            missing = self.cfg.n_classes - len(val_classes)
            print(
                f"[data] WARNING: {missing} class(es) absent from validation -- "
                "balanced accuracy averages only over classes that appear, so it "
                "is not comparable to the leaderboard. Consider other val_sessions."
            )

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=False)
