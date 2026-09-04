"""LightningModule: loss, optimiser, and the scored metric.

The competition scores *balanced accuracy* on argmax over the 50-word primary
vocabulary. Plain accuracy is misleading on this label distribution -- report
balanced accuracy as the headline validation metric and nothing else.
"""

from __future__ import annotations

import lightning as L

from .config import Config
from .models import build_model


class LitWordClassifier(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg
        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            **cfg.model_kwargs,
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """TODO: class-weighted cross-entropy."""
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        """TODO: accumulate preds/targets, log balanced accuracy."""
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError
