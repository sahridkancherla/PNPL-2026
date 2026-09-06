"""LightningModule: loss, optimiser, and the scored metric.

Validation reports BAcc@k (see .metrics): macro-averaged Recall@k over the
50-word vocabulary. BAcc@`metric_k` (10 by default) is the leaderboard metric
and what early stopping / checkpointing track; BAcc@1 is logged alongside it
only as a tiebreaker.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F
from .config import Config
from .metrics import balanced_topk_accuracy
from .models import build_model


class LitWordClassifier(L.LightningModule):
    def __init__(self, cfg: Config, class_weights: torch.Tensor | None = None):
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg
        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            **cfg.model_kwargs,
        )
        # Non-persistent: the weights follow the module across devices but stay
        # out of the checkpoint. They are derived from the training split, not
        # learned, and inference constructs this module without them -- saving
        # them would make load_from_checkpoint fail on an unexpected key.
        # All-ones is equivalent to unweighted cross-entropy.
        if class_weights is None:
            class_weights = torch.ones(cfg.n_classes)
        self.register_buffer("class_weights", class_weights, persistent=False)
        # Full logits, not argmax: top-k needs the ranking.
        self._val_logits: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y, weight=self.class_weights)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y, weight=self.class_weights)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        self._val_logits.append(logits.detach().cpu())
        self._val_targets.append(y.detach().cpu())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_logits:
            return
        logits = torch.cat(self._val_logits)
        targets = torch.cat(self._val_targets)
        n = self.cfg.n_classes
        k = self.cfg.metric_k

        self.log("val/bacc1", balanced_topk_accuracy(logits, targets, 1, n))
        self.log(
            f"val/bacc{k}",
            balanced_topk_accuracy(logits, targets, k, n),
            prog_bar=True,
        )
        self._val_logits.clear()
        self._val_targets.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
