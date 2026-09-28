"""LightningModule for trial 5: three losses, a schedule, an EMA, the metric.

BAcc@k comes from `pnpl2026.metrics`, the one implementation in the repo. A
second copy of the scorer is the same class of bug as a second label mapping: it
agrees for months and then quietly does not.

⚠️ Train metrics are logged next to validation ones deliberately. Without the
gap between them there is no way to tell an underfitting model from an
overfitting one, and so no way to choose between more capacity and more
regularisation. Trial 3 stalled underfitting and only the gap showed it.

**Three things here that trials 1-3 did not have**, each of which the literature
measures as worth more than an architecture change:

    a warmup and a cosine schedule at a HIGHER peak learning rate. Trial 3's
    flat loss at a constant 1e-4 with no warmup is the textbook symptom

    a weight EMA, evaluated instead of the live weights

    the scored logits are a CALIBRATED BLEND of two heads, and each head's own
    score is logged beside the blend so the blend weight can be chosen on
    evidence rather than assumed
"""

from __future__ import annotations

import math

import lightning as L
import torch
import torch.nn.functional as F

from ..metrics import balanced_topk_accuracy
from .embeddings import load_target_embeddings
from .losses import d_siglip_loss, semantic_logits
from .models import build_model

#SigLIP initialises its temperature high and its bias negative, so early
#training sees mostly-negative logits and does not saturate every pair at once
INITIAL_LOG_SCALE = math.log(10.0)
INITIAL_LOGIT_BIAS = -10.0


class WeightEMA:
    """An exponential moving average of the model's floating-point parameters.

    Evaluating an average of recent weights rather than the latest ones is
    reliably worth about a point and costs one extra copy of the model. It also
    makes the epoch-to-epoch BAcc curve far less noisy, which matters when the
    noise band is already ~1.2 points wide and every decision is a comparison
    inside it.

    ATTRS:
        decay (float): how much of the old average to keep per step
        shadow (dict): name -> averaged tensor
    """

    def __init__(self, module, decay=0.999):
        """
        ARGS:
            module (nn.Module): the model whose weights are averaged
            decay (float): 0.999 averages over roughly the last 1000 steps
        """
        self.decay = decay

        #detached clones: the average must never carry a gradient, and it must
        #not alias the live parameters or it would track them exactly
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.state_dict().items()
            if parameter.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, module):
        """Fold the current weights into the average.

        ARGS:
            module (nn.Module): the model, after an optimiser step
        """
        for name, value in module.state_dict().items():
            #a non-float entry -- a step counter, say -- is copied, not averaged
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)

    def state_dict(self):
        """The averaged weights, for a checkpoint."""
        return self.shadow

    def load_state_dict(self, state):
        """Restore averaged weights from a checkpoint.

        ARGS:
            state (dict): name -> tensor, as written by `state_dict`
        """
        #⚠️ copied in place rather than rebound, so the tensors stay on whatever
        #device the module was moved to
        for name, value in state.items():
            if name in self.shadow:
                self.shadow[name].copy_(value)


class LitTrialFive(L.LightningModule):
    """Wraps the trial-5 network with its three losses, metric and optimiser.

    ATTRS:
        cfg (Trial5Config): the run config
        model (nn.Module): the network being trained
        target_embeddings (torch.Tensor): [n_classes, embed_dim] buffer, fixed
        class_weights (torch.Tensor): [n_classes] buffer, not checkpointed
        ema (WeightEMA or None): None when cfg.ema_decay is 0
    """

    def __init__(self, cfg, class_weights=None, n_sessions=0):
        """
        ARGS:
            cfg (Trial5Config): the merged run config
            class_weights (torch.Tensor or None): [n_classes] from the training
                split. None means unweighted cross-entropy
            n_sessions (int): training sessions, for the adversary. 0 omits it
        """
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
        self.cfg = cfg

        #⚠️ loaded before the model, because the table's width decides the
        #semantic head's output size and a mismatch is a silent shape bug
        self.uses_semantic = cfg.semantic_weight > 0
        if self.uses_semantic:
            table = load_target_embeddings(
                cfg.resolved_word_embeddings(),
                n_classes=cfg.n_classes,
                centre=cfg.centre_embeddings,
            )
            embed_dim = table.shape[1]
        else:
            #a single-column placeholder: the head is built but never trained
            table = torch.zeros(cfg.n_classes, 1)
            embed_dim = 1

        #persistent: the targets are part of what this checkpoint means, and a
        #differently-built table would silently rescore every prediction
        self.register_buffer("target_embeddings", table, persistent=True)

        #⚠️ derived from the config, never hard-coded: the trunk checks its own
        #receptive field against this frame count, and a stale 250 here would
        #disable the one guard that catches an over-reaching dilation ladder
        n_times = int(round((cfg.tmax - cfg.tmin) * cfg.sfreq))

        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            n_sessions=n_sessions if cfg.session_adv_weight > 0 else 0,
            embed_dim=embed_dim,
            sample_rate=float(cfg.sfreq),
            n_times=n_times,
            #⚠️ forwarded rather than applied here. The augmentation must run
            #AFTER the robust instance norm, and the norm is inside the model --
            #see TimeMask in models/frontend.py for what going first costs
            time_mask_spans=cfg.time_mask_spans,
            time_mask_width=cfg.time_mask_width,
            component_dropout=cfg.component_dropout,
            **cfg.model_kwargs,
        )

        #the SigLIP temperature and bias are learned, not tuned
        self.log_scale = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
        self.logit_bias = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        #non-persistent: the weights come from the training split, not from
        #learning, and inference builds this module without them
        if class_weights is None:
            class_weights = torch.ones(cfg.n_classes)
        self.register_buffer("class_weights", class_weights, persistent=False)

        #built in on_fit_start, once the module is on its final device
        self.ema = None
        self._ema_backup = None

        #⚠️ NOT named `_buffers`: that is nn.Module's own registry of every
        #registered buffer, and assigning to it silently wipes
        #`target_embeddings` and `class_weights`. This already happened once.
        #full logits, not argmax: top-k needs the whole ranking
        self._epoch_store = {
            stage: {"blend": [], "closed": [], "semantic": [], "targets": []}
            for stage in ("train", "val")
        }

    # -- plumbing ---------------------------------------------------------

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [batch, n_channels, n_times]

        RETURNS:
            dict from TrialFiveNet.forward
        """
        return self.model(windows)

    def on_fit_start(self):
        """Build the EMA once the module is on its final device."""
        #built here rather than in __init__ so the shadow tensors are created
        #on the accelerator and `update` never copies across devices
        if self.cfg.ema_decay > 0 and self.ema is None:
            self.ema = WeightEMA(self.model, self.cfg.ema_decay)

    def on_save_checkpoint(self, checkpoint):
        """Store the averaged weights alongside the live ones."""
        if self.ema is not None:
            checkpoint["weight_ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Restore the averaged weights, if this checkpoint carries them."""
        #a resume that silently dropped the EMA would restart the average from
        #the live weights and throw away everything it had smoothed
        state = checkpoint.get("weight_ema")
        if state is None:
            return
        if self.ema is None:
            self.ema = WeightEMA(self.model, max(self.cfg.ema_decay, 1e-9))
        self.ema.load_state_dict(state)

    def apply_ema_weights(self):
        """Overwrite the live weights with the averaged ones, permanently.

        ⚠️ For INFERENCE only, and it is not optional there. Validation is scored
        on the EMA weights (see `on_validation_start`), so `best.ckpt` was
        selected by what the average scored -- while the checkpoint's live
        weights are whatever the last optimiser step happened to produce.
        Predicting with the live weights therefore submits a different model
        from the one the run was chosen on, and the two differ by roughly the
        point the EMA was worth.

        RETURNS:
            bool: True if averaged weights were found and applied
        """
        if self.ema is None:
            return False

        #strict=False because the shadow holds only floating-point entries
        self.model.load_state_dict(self.ema.shadow, strict=False)
        return True

    # -- augmentation -----------------------------------------------------

    # -- losses -----------------------------------------------------------

    def _adversary_strength(self):
        """How hard to push the encoder away from session identity, right now.

        ⚠️ Ramped from zero over the first few epochs. An adversary at full
        strength from step one prevents the encoder from learning anything, and
        the result looks like a broken architecture rather than a scheduling
        mistake.

        RETURNS:
            float: the gradient-reversal coefficient
        """
        warmup = max(self.cfg.session_adv_warmup_epochs, 0)
        if warmup == 0:
            return self.cfg.session_adv_weight

        #current_epoch is 0-based, so epoch `warmup` is the first at full weight
        fraction = min(1.0, self.current_epoch / warmup)
        return self.cfg.session_adv_weight * fraction

    def _step(self, batch, stage):
        """Shared body of the training and validation steps.

        ARGS:
            batch (tuple): (windows, labels, sessions) from the DataLoader
            stage (str): "train" or "val", the metric prefix

        RETURNS:
            loss : scalar tensor
        """
        windows, labels, sessions = batch

        #⚠️ no augmentation call here. It lives inside the model, after the
        #instance norm, and is gated on module.training -- so validation and
        #`predict.py` get the clean path without either having to know
        out = self(windows)
        closed_logits = out["logits"]

        #exp() once, and reused by both the loss and the scoring path
        logit_scale = self.log_scale.exp()

        total = windows.new_zeros(())

        if self.cfg.closed_set_weight > 0:
            closed_loss = F.cross_entropy(
                closed_logits,
                labels,
                weight=self.class_weights,
                label_smoothing=self.cfg.label_smoothing,
            )
            total = total + self.cfg.closed_set_weight * closed_loss
            self.log(f"{stage}/loss_closed", closed_loss, on_epoch=True)

        if self.uses_semantic:
            semantic_loss = d_siglip_loss(
                out["embedding"],
                labels,
                self.target_embeddings,
                logit_scale,
                self.logit_bias,
            )
            total = total + self.cfg.semantic_weight * semantic_loss
            self.log(f"{stage}/loss_semantic", semantic_loss, on_epoch=True)
            semantic = semantic_logits(
                out["embedding"], self.target_embeddings, logit_scale
            )
        else:
            #zeros, so the blend below is a no-op at inference_semantic_weight 0
            semantic = torch.zeros_like(closed_logits)

        #the adversary reads the pooled vector and is only meaningful where a
        #session label exists; validation sessions are deliberately unlabelled
        if stage == "train" and self.cfg.session_adv_weight > 0:
            labelled = sessions >= 0
            if labelled.any():
                strength = self._adversary_strength()
                session_out = self.model.session_logits(out["pooled"], strength)
                if session_out is not None:
                    session_loss = F.cross_entropy(
                        session_out[labelled], sessions[labelled]
                    )
                    #NOT multiplied by the weight again -- the strength is
                    #already inside the reversal, and doubling it would scale
                    #the adversary's own descent as well as the encoder's ascent
                    total = total + session_loss
                    self.log("train/loss_session", session_loss, on_epoch=True)
                    self.log("train/adv_strength", strength, on_epoch=True)

        self.log(f"{stage}/loss", total, prog_bar=True, on_epoch=True)

        #detached and moved off the accelerator, or an epoch's worth of logits
        #pins memory the next forward pass needs
        blend = self._blend(closed_logits, semantic)
        store = self._epoch_store[stage]
        store["blend"].append(blend.detach().cpu())
        store["closed"].append(closed_logits.detach().cpu())
        store["semantic"].append(semantic.detach().cpu())
        store["targets"].append(labels.detach().cpu())
        return total

    def _blend(self, closed_logits, semantic):
        """Combine the two heads into the logits the metric actually reads.

        ⚠️ Both sides are log-softmaxed first. The closed-set head's logits and
        a scaled cosine similarity are not on the same scale, and averaging raw
        logits would let whichever head happens to be louder decide every
        prediction regardless of which is better.

        ARGS:
            closed_logits (torch.Tensor): [batch, n_classes]
            semantic (torch.Tensor): [batch, n_classes]

        RETURNS:
            [batch, n_classes] - a log-probability blend
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
        #⚠️ Lightning runs a sanity-check validation before training starts, when
        #the EMA does not exist yet; that is the `is None` case, not an error
        if self.ema is None:
            return
        self._ema_backup = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
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
        """Score one epoch's accumulated predictions and clear the buffers.

        ARGS:
            stage (str): "train" or "val"
        """
        store = self._epoch_store[stage]

        #sanity-check callbacks can run an epoch with nothing accumulated
        if not store["targets"]:
            return

        targets = torch.cat(store["targets"])
        n_classes = self.cfg.n_classes
        k = self.cfg.metric_k

        #BAcc@1 is a tiebreaker only; its floor is 0.02, not 0.20, so it looks
        #broken next to the scored metric unless read against the right chance
        blend = torch.cat(store["blend"])
        self.log(f"{stage}/bacc1", balanced_topk_accuracy(blend, targets, 1, n_classes))
        self.log(
            f"{stage}/bacc{k}",
            balanced_topk_accuracy(blend, targets, k, n_classes),
            prog_bar=(stage == "val"),
        )

        #⚠️ each head scored separately as well, because the blend weight is a
        #config guess. Without these two numbers there is no way to tell a good
        #blend from a good head carrying a useless one
        self.log(
            f"{stage}/bacc{k}_closed",
            balanced_topk_accuracy(torch.cat(store["closed"]), targets, k, n_classes),
        )
        if self.uses_semantic:
            self.log(
                f"{stage}/bacc{k}_semantic",
                balanced_topk_accuracy(
                    torch.cat(store["semantic"]), targets, k, n_classes
                ),
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

        ⚠️ Deliberately NOT ScaledAdam, which Zipformer measures as better. One
        variable at a time: the architecture is already new, and trial 3's
        post-mortem was that window, rate and model all moved at once and the
        result was unattributable. ScaledAdam is its own run.

        RETURNS:
            dict: optimiser plus a per-step scheduler
        """
        optimiser = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        #estimate_stepping_batches accounts for accumulation and for max_epochs,
        #so the cosine actually finishes at the end of the run
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = max(int(total_steps * self.cfg.warmup_fraction), 1)
        floor = self.cfg.min_lr / self.cfg.lr

        def scale(step):
            """Multiplier on the peak learning rate at `step`."""
            #linear warmup: a pre-norm attention stack given its peak rate on
            #step one diverges or, worse, settles into the flat curve trial 3 got
            if step < warmup_steps:
                return (step + 1) / warmup_steps

            #cosine from 1 down to `floor` over what remains
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, scale)
        print(
            f"[optim] AdamW lr {self.cfg.lr:g}, {warmup_steps} warmup steps of "
            f"{total_steps}, cosine to {self.cfg.min_lr:g}"
        )

        return {
            "optimizer": optimiser,
            #"step", not "epoch": the warmup is a few hundred steps, which is a
            #fraction of one epoch here
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
