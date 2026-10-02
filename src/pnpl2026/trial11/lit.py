"""LightningModule for trial 11: per-word losses over holdout-shaped sentence streams.

BAcc@k comes from `pnpl2026.metrics`, the one implementation in the repo.

Three things here that trial 6 did not have:

    the batch is S sentence STREAMS plus a flat list of their words; every
    metric and loss is per word. Nothing is masked in trial 11

    `val/bacc10_isolated` and `val/bacc10_sentence` split the score the way
    the holdout splits its words -- 9.6% alone, the rest in sentences. A
    shared stream can only help the second

    a gradient probe: how hard each loss pulls on the vector both heads read.
    The design asks for the two to be of similar size before either weight is
    trusted
"""

from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F

from ..metrics import balanced_topk_accuracy
from .embeddings import load_target_embeddings
from .losses import semantic_logits, siglip_loss
from .models import build_model

#SigLIP's initialisation: t = 10, b = -10, per the design and the paper
INITIAL_LOG_SCALE = math.log(10.0)
INITIAL_LOGIT_BIAS = -10.0


class WeightEMA:
    """An exponential moving average of the model's learned parameters.

    Parameters only, not buffers -- the buffers here are fixed filters, and
    averaging a constant only accumulates float32 rounding (trial 6).

    ATTRS:
        decay (float): how much of the old average to keep per step
        shadow (dict): name -> averaged tensor
    """

    def __init__(self, module, decay=0.999):
        """
        ARGS:
            module (nn.Module): the model whose parameters are averaged
            decay (float): 0.999 averages over roughly the last 1000 steps
        """
        self.decay = decay

        #detached clones: never a gradient, never an alias of the live weights
        self.shadow = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}

    @torch.no_grad()
    def update(self, module):
        """Fold the current parameters into the average.

        ARGS:
            module (nn.Module): the model, after an optimiser step
        """
        for name, parameter in module.named_parameters():
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1 - self.decay)

    def state_dict(self):
        """The averaged parameters, for a checkpoint."""
        return self.shadow

    def to(self, device):
        """Move the average to `device`, returning self.

        ARGS:
            device (torch.device): where the averaged module now lives
        """
        self.shadow = {name: value.to(device) for name, value in self.shadow.items()}
        return self

    def load_state_dict(self, state):
        """Restore the average from a checkpoint, in place.

        ARGS:
            state (dict): name -> tensor, as written by `state_dict`
        """
        for name, value in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(value)


class LitTrialEleven(L.LightningModule):
    """Wraps the trial-7 network with its two losses, metrics and optimiser.

    ATTRS:
        cfg (Trial11Config): the run config
        model (StreamZipformerNet)
        target_embeddings (torch.Tensor): [n_classes, embed_dim] buffer
        class_weights (torch.Tensor): [n_classes] buffer, not checkpointed
        ema (WeightEMA or None)
    """

    def __init__(self, cfg, class_weights=None):
        """
        ARGS:
            cfg (Trial11Config): the merged run config
            class_weights (torch.Tensor or None): [n_classes] from the training
                tokens. None means unweighted cross-entropy
        """
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg

        #⚠️ loaded before the model: the table's width sets the semantic head's
        #output size, and a mismatch is a silent shape bug
        self.uses_semantic = cfg.semantic_weight > 0
        if self.uses_semantic:
            table = load_target_embeddings(
                cfg.resolved_word_embeddings(),
                n_classes=cfg.n_classes,
                centre=cfg.centre_embeddings,
            )
        else:
            table = torch.zeros(cfg.n_classes, 1)

        #persistent: a differently-built table would rescore every prediction
        self.register_buffer("target_embeddings", table, persistent=True)

        #⚠️ from the config, never hard-coded: the padded STREAM length, which
        #the trunk checks its reach against and every factor must tile
        n_times = int(round(cfg.window_seconds * cfg.sfreq))

        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            embed_dim=table.shape[1],
            sample_rate=float(cfg.sfreq),
            n_times=n_times,
            word_mask_p=cfg.word_mask_p,
            component_dropout=cfg.component_dropout,
            gaussian_noise=cfg.gaussian_noise,
            **cfg.model_kwargs,
        )

        #the SigLIP temperature and bias are learned, not tuned
        self.log_scale = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
        self.logit_bias = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        if class_weights is None:
            class_weights = torch.ones(cfg.n_classes)
        self.register_buffer("class_weights", class_weights, persistent=False)

        self.ema = None
        self._ema_backup = None

        #⚠️ NOT named `_buffers`, which is nn.Module's own registry
        self._epoch_store = {
            stage: {"blend": [], "closed": [], "semantic": [], "targets": [], "isolated": []}
            for stage in ("train", "val")
        }

    # -- plumbing ---------------------------------------------------------

    def forward(self, batch):
        """
        ARGS:
            batch (dict): from `collate_streams`

        RETURNS:
            dict from StreamZipformerNet.forward
        """
        return self.model(
            batch["streams"],
            batch["stream_lengths"],
            batch["sequence"],
            batch["offsets"],
        )

    def on_fit_start(self):
        """Build the EMA -- or move a restored one -- onto the module's device."""
        if self.cfg.ema_decay > 0 and self.ema is None:
            self.ema = WeightEMA(self.model, self.cfg.ema_decay)

        #⚠️ a resume restores the EMA before Lightning moves the module to the
        #GPU, so the average arrives on the CPU. Trial 5 crashed on exactly this
        if self.ema is not None:
            self.ema.to(self.device)

    def on_save_checkpoint(self, checkpoint):
        """Store the averaged parameters alongside the live ones."""
        if self.ema is not None:
            checkpoint["weight_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Restore the averaged parameters, if this checkpoint carries them."""
        state = checkpoint.get("weight_ema")
        if state is None:
            return
        if self.ema is None:
            self.ema = WeightEMA(self.model, max(self.cfg.ema_decay, 1e-9))
        self.ema.load_state_dict(state)

    def apply_ema_weights(self):
        """Overwrite the live parameters with the averaged ones, for inference.

        ⚠️ Not optional there: validation, and so best.ckpt, is chosen on the
        average, so predicting with the live weights submits another model.

        RETURNS:
            bool: True if averaged weights were found and applied
        """
        if self.ema is None:
            return False

        #strict=False because the average holds parameters only, not buffers
        self.model.load_state_dict(self.ema.shadow, strict=False)
        return True

    # -- losses -----------------------------------------------------------

    def _step(self, batch, stage):
        """Shared body of the training and validation steps.

        ARGS:
            batch (dict): from `collate_streams`
            stage (str): "train" or "val", the metric prefix

        RETURNS:
            loss : scalar tensor
        """
        out = self(batch)
        labels = batch["labels"]
        logit_scale = self.log_scale.exp()

        #⚠️ a word masked for regularisation is not scored: it would teach
        #pure prediction from context. Under D8 every word is masked, and
        #predicting from context is exactly the question
        scored = torch.ones_like(out["masked"]) if self.cfg.mask_target else ~out["masked"]

        total = out["logits"].new_zeros(())
        losses = {}
        if self.cfg.closed_set_weight > 0:
            losses["closed"] = self.cfg.closed_set_weight * F.cross_entropy(
                out["logits"][scored],
                labels[scored],
                weight=self.class_weights,
                label_smoothing=self.cfg.label_smoothing,
            )
            self.log(f"{stage}/loss_closed", losses["closed"], on_epoch=True, batch_size=int(scored.sum()))

        if self.uses_semantic:
            losses["semantic"] = self.cfg.semantic_weight * siglip_loss(
                out["embedding"][scored],
                labels[scored],
                self.target_embeddings,
                logit_scale,
                self.logit_bias,
                dedup=self.cfg.siglip_dedup,
                shuffle=(stage == "train"),
            )
            self.log(
                f"{stage}/loss_semantic", losses["semantic"], on_epoch=True, batch_size=int(scored.sum())
            )

        for value in losses.values():
            total = total + value

        if stage == "train":
            self._probe_gradients(losses, out["context"])
        self.log(f"{stage}/loss", total, prog_bar=True, on_epoch=True, batch_size=int(scored.sum()))

        #semantic scores against the 50 targets, for the metric and the blend
        semantic = (
            semantic_logits(out["embedding"], self.target_embeddings, logit_scale)
            if self.uses_semantic
            else torch.zeros_like(out["logits"])
        )

        #training metrics over scored words only; validation masks nothing
        #except under D8, where every word is predicted from context
        store = self._epoch_store[stage]
        store["blend"].append(self._blend(out["logits"], semantic)[scored].detach().cpu())
        store["closed"].append(out["logits"][scored].detach().cpu())
        store["semantic"].append(semantic[scored].detach().cpu())
        store["targets"].append(labels[scored].detach().cpu())
        store["isolated"].append(batch["isolated"][scored].detach().cpu())
        return total

    def _probe_gradients(self, losses, context):
        """Log how hard each loss pulls on the vector both heads read.

        The design's check that the two weights are balanced: once SigLIP is
        normalised by N, its gradient into the shared encoder should be of the
        same order as cross-entropy's. Measured on `context`, so it costs a
        backward pass through the heads only.

        ARGS:
            losses (dict): "closed" and "semantic" -> weighted scalar losses
            context (torch.Tensor): [N, d_model], upstream of both heads
        """
        every = self.cfg.grad_probe_every
        if every <= 0 or self.global_step % every != 0 or len(losses) < 2:
            return

        #retain_graph: the real backward pass still has to run after this.
        #⚠️ logged as an EPOCH mean: Lightning writes step values only every
        #log_every_n_steps, offset by one, so a probe every 50 steps never lands
        norms = {}
        for name, value in losses.items():
            (gradient,) = torch.autograd.grad(value, context, retain_graph=True)
            norms[name] = gradient.norm()
            self.log(f"train/grad_{name}", norms[name], on_step=False, on_epoch=True, batch_size=1)
        self.log(
            "train/grad_ratio_semantic_to_closed",
            norms["semantic"] / norms["closed"].clamp_min(1e-12),
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )

    def _blend(self, closed_logits, semantic):
        """Combine the two heads into the logits the metric reads.

        Both sides are log-softmaxed first: raw logits and scaled cosines are
        not on the same scale, and the louder head would win regardless.

        ARGS:
            closed_logits (torch.Tensor): [n_words, n_classes]
            semantic (torch.Tensor): [n_words, n_classes]

        RETURNS:
            [n_words, n_classes] - a log-probability blend
        """
        weight = self.cfg.inference_semantic_weight
        if weight <= 0:
            return closed_logits
        if weight >= 1:
            return semantic

        #dim=1 is the class axis in both
        return (1 - weight) * closed_logits.log_softmax(dim=1) + weight * semantic.log_softmax(dim=1)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Fold the post-step weights into the EMA."""
        if self.ema is not None:
            self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    # -- evaluating on the averaged weights -------------------------------

    def on_validation_start(self):
        """Swap the averaged weights in, keeping the live ones aside."""
        #the sanity-check pass runs before the EMA exists; that is not an error
        if self.ema is None:
            return
        self._ema_backup = {
            name: value.detach().clone() for name, value in self.model.state_dict().items()
        }
        self.model.load_state_dict(self.ema.shadow, strict=False)

    def on_validation_end(self):
        """Put the live weights back, or training would continue from the mean."""
        if self._ema_backup is None:
            return
        self.model.load_state_dict(self._ema_backup)
        self._ema_backup = None

    # -- metrics ----------------------------------------------------------

    def _log_balanced_accuracy(self, stage):
        """Score one epoch's words, overall and split as the holdout splits them.

        ARGS:
            stage (str): "train" or "val"
        """
        store = self._epoch_store[stage]
        if not store["targets"]:
            return

        targets = torch.cat(store["targets"])
        isolated = torch.cat(store["isolated"])
        blend = torch.cat(store["blend"])
        k, n_classes = self.cfg.metric_k, self.cfg.n_classes

        #BAcc@1 is a tiebreaker only; its floor is 0.02, not 0.20
        self.log(f"{stage}/bacc1", balanced_topk_accuracy(blend, targets, 1, n_classes))
        self.log(
            f"{stage}/bacc{k}",
            balanced_topk_accuracy(blend, targets, k, n_classes),
            prog_bar=(stage == "val"),
        )

        #each head alone, because the blend weight is a guess to be replaced
        self.log(
            f"{stage}/bacc{k}_closed",
            balanced_topk_accuracy(torch.cat(store["closed"]), targets, k, n_classes),
        )
        if self.uses_semantic:
            self.log(
                f"{stage}/bacc{k}_semantic",
                balanced_topk_accuracy(torch.cat(store["semantic"]), targets, k, n_classes),
            )

        #the holdout's two kinds of word, scored apart
        for name, chosen in (("isolated", isolated), ("sentence", ~isolated)):
            if chosen.any():
                self.log(
                    f"{stage}/bacc{k}_{name}",
                    balanced_topk_accuracy(blend[chosen], targets[chosen], k, n_classes),
                )

        #cleared every epoch, or memory grows with the whole run
        for values in store.values():
            values.clear()

    def on_train_epoch_end(self):
        self._log_balanced_accuracy("train")

    def on_validation_epoch_end(self):
        self._log_balanced_accuracy("val")

    # -- optimiser --------------------------------------------------------

    def configure_optimizers(self):
        """AdamW with linear warmup into a cosine decay.

        RETURNS:
            dict: optimiser plus a per-step scheduler
        """
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        #estimate_stepping_batches accounts for max_epochs, so the cosine
        #actually finishes at the end of the run
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = max(int(total_steps * self.cfg.warmup_fraction), 1)
        floor = self.cfg.min_lr / self.cfg.lr

        def scale(step):
            """Multiplier on the peak learning rate at `step`."""
            if step < warmup_steps:
                return (step + 1) / warmup_steps

            #cosine from 1 down to `floor` over what remains
            progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, scale)
        print(
            f"[optim] AdamW lr {self.cfg.lr:g}, {warmup_steps} warmup steps of "
            f"{total_steps}, cosine to {self.cfg.min_lr:g}"
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
