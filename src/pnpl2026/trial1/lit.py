"""LightningModule for trial 1: loss, optimiser, and the scored metric.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED: never run, and `lightning` is not
installed on the machine this was written on.

BAcc@k comes from `pnpl2026.metrics`, the one implementation in the repo. A
second copy of the scorer is the same class of bug as a second label mapping:
it agrees for months and then quietly does not.

⚠️ Train metrics are logged next to validation ones deliberately. Without the
gap between them there is no way to tell an underfitting model from an
overfitting one, and so no way to choose between more capacity and more
regularisation.
"""

from __future__ import annotations

import lightning as L
import torch
import torch.nn.functional as F

from ..metrics import balanced_topk_accuracy
from .models import build_model


class LitTrialOne(L.LightningModule):
    """Wraps the trial-1 network with its loss, metric and optimiser.

    ATTRS:
        cfg (Trial1Config): the run config
        model (nn.Module): the network being trained
        class_weights (torch.Tensor): [n_classes], buffer, not checkpointed
    """

    def __init__(self, cfg, class_weights=None):
        """
        ARGS:
            cfg (Trial1Config): the merged run config
            class_weights (torch.Tensor or None): [n_classes] from the training
                split. None means unweighted cross-entropy
        """
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg

        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            **cfg.model_kwargs,
        )

        #non-persistent: the weights follow the module across devices but stay
        #out of the checkpoint. They come from the training split, not from
        #learning, and inference builds this module without them -- saving them
        #would make load_from_checkpoint fail on an unexpected key
        if class_weights is None:
            class_weights = torch.ones(cfg.n_classes)
        self.register_buffer("class_weights", class_weights, persistent=False)

        #full logits, not argmax: top-k needs the whole ranking
        self._epoch_logits = {"train": [], "val": []}
        self._epoch_targets = {"train": [], "val": []}

    def forward(self, windows):
        return self.model(windows)

    def _step(self, batch, stage):
        """Shared body of the training and validation steps.

        ARGS:
            batch (tuple): (windows, labels) from the DataLoader
            stage (str): "train" or "val", the metric prefix

        RETURNS:
            loss : scalar tensor
        """
        windows, labels = batch
        logits = self(windows)
        loss = F.cross_entropy(logits, labels, weight=self.class_weights)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True)

        #detached and moved off the accelerator, or the epoch's worth of logits
        #pins GPU memory that the next forward pass needs
        self._epoch_logits[stage].append(logits.detach().cpu())
        self._epoch_targets[stage].append(labels.detach().cpu())
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def _log_balanced_accuracy(self, stage):
        """Score one epoch's accumulated predictions and clear the buffers.

        ARGS:
            stage (str): "train" or "val"
        """
        #sanity-check callbacks can run an epoch with nothing accumulated
        if not self._epoch_logits[stage]:
            return

        logits = torch.cat(self._epoch_logits[stage])
        targets = torch.cat(self._epoch_targets[stage])
        n_classes = self.cfg.n_classes
        k = self.cfg.metric_k

        #BAcc@1 is a tiebreaker only; its floor is 0.02, not 0.20, so it looks
        #broken next to the scored metric unless read against the right chance
        self.log(
            f"{stage}/bacc1",
            balanced_topk_accuracy(logits, targets, 1, n_classes),
        )
        self.log(
            f"{stage}/bacc{k}",
            balanced_topk_accuracy(logits, targets, k, n_classes),
            prog_bar=(stage == "val"),
        )

        #cleared every epoch, or memory grows with the whole run
        self._epoch_logits[stage].clear()
        self._epoch_targets[stage].clear()

    def on_train_epoch_end(self):
        self._log_balanced_accuracy("train")

    def on_validation_epoch_end(self):
        self._log_balanced_accuracy("val")

    def configure_optimizers(self):
        """AdamW, matching the shared baseline so the comparison is fair.

        RETURNS:
            a torch optimiser
        """
        #no schedule yet: a schedule is a change worth measuring on its own,
        #not one to introduce silently alongside a new architecture
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
