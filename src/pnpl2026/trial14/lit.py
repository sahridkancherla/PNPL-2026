"""LightningModule for trial 14: retrieval plus a trained 50-word head, blended.

BAcc@k comes from `pnpl2026.metrics`, the one implementation in the repo.

    losses    D-SigLIP of each word's output against its own t5 embedding
              (every word a target), weighted by `semantic_weight`; and the
              50-word head's logit-adjusted cross-entropy on scored words,
              weighted by `closed_weight`
    scores    retrieval: cosine against the 50 competition words' t5
              embeddings; the head: its plain logits; the BLEND of the two as
              log-probabilities, `inference_closed_weight` of the way to the
              head. val/bacc10 is the blend; `_retrieval` and `_closed` alone

The head is a dropout and one linear map from the sentence transformer's
output (`context`, d_model wide) to 50 logits. It lives here, outside the
network, so the network's weights are trial 13's key for key: a run starts
from a G2 checkpoint (`init_checkpoint`) and adds only the head.

⚠️ Two tables, kept differently:

    `target_table`   every word's row, ~100 MB. Training only, so NOT
                     checkpointed -- it would triple every checkpoint
    `primary_table`, `moses_table`   the 50 + 50 candidates. Checkpointed:
                     prediction needs them and nothing else

`load_trained` rebuilds a module from a checkpoint alone -- trial 14's, or
trial 10's, 12's and 13's. Those have no head: `has_closed_head` is False and
they are scored by retrieval alone, whatever a config says.
"""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from ..metrics import balanced_topk_accuracy
from .config import Trial14Config
from .data.dataset import load_manifest
from .losses import blend_scores, logit_adjusted_cross_entropy, semantic_logits, siglip_loss
from .models import T5_WIDTH, build_model

#SigLIP's initialisation: t = 10, b = -10, per the paper
INITIAL_LOG_SCALE = math.log(10.0)
INITIAL_LOGIT_BIAS = -10.0

#the scored vocabularies' size; both are 50 words
VOCABULARY_SIZE = 50

#the state-dict keys trial 14 adds; an older checkpoint lacks exactly these
HEAD_KEYS = ("closed_head.1.weight", "closed_head.1.bias", "log_prior")


def spatial_inputs(cfg):
    """What the spatial layer is built from, read from the reference cache.

    ARGS:
        cfg (Trial14Config)

    RETURNS:
        dict: "positions" [306, 2] or None, "pca" dict or None
    """
    cache_dir = cfg.resolved_cache_dir()
    manifest = load_manifest(cache_dir)

    #positions exist only when the h5 files carried channel names
    positions = None
    if manifest.get("has_positions"):
        positions = np.load(cache_dir / "positions.npy")

    pca = None
    if (cache_dir / "pca.npz").exists():
        with np.load(cache_dir / "pca.npz") as stored:
            pca = {name: stored[name] for name in ("mean", "components", "scale")}
    return {"positions": positions, "pca": pca}


class WeightEMA:
    """An exponential moving average of the model's learned parameters.

    Parameters only, not buffers. Off in every trial 14 config (ema_decay 0).

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
            device (torch.device)
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


class LitTrialFourteen(L.LightningModule):
    """The reference network, a 50-word head, both losses, and the blend.

    ATTRS:
        cfg (Trial14Config)
        model (ReferenceNet)
        closed_head (nn.Sequential): dropout, then d_model -> 50
        target_table (torch.Tensor): [n_words, 1024] buffer, not checkpointed
        primary_table (torch.Tensor): [50, 1024] buffer, the scored candidates
        moses_table (torch.Tensor): [50, 1024] buffer, the secondary's
        log_prior (torch.Tensor): [50] buffer, log class frequencies in training
        log_scale, logit_bias (nn.Parameter): SigLIP's t and b
        has_closed_head (bool): False for an older checkpoint without one
        ema (WeightEMA or None)
    """

    def __init__(self, cfg, table=None, spatial=None, embed_dim=T5_WIDTH, prior=None):
        """
        ARGS:
            cfg (Trial14Config): the merged run config
            table (TargetTable or None): None only when a checkpoint is about
                to overwrite the candidate tables -- see `load_trained`
            spatial (dict or None): from `spatial_inputs`; None reads the cache
            embed_dim (int): the placeholders' width when `table` is None
            prior (np.ndarray or None): [50] class frequencies from the
                training tokens; None is uniform, until a checkpoint says otherwise
        """
        super().__init__()

        #⚠️ ignore the non-config arguments: Lightning compares each with the
        #config by ==, and a NumPy prior makes that comparison raise
        self.save_hyperparameters(cfg.to_dict(), ignore=["table", "spatial", "embed_dim", "prior"])
        self.cfg = cfg

        #the full table trains; only its two 50-row slices are ever scored
        if table is not None:
            full = table.table
            primary, moses = full[table.primary_rows], full[table.moses_rows]
        else:
            full = torch.zeros(0, embed_dim)
            primary = torch.zeros(VOCABULARY_SIZE, embed_dim)
            moses = torch.zeros(VOCABULARY_SIZE, embed_dim)
        self.register_buffer("target_table", full, persistent=False)
        self.register_buffer("primary_table", primary.clone(), persistent=True)
        self.register_buffer("moses_table", moses.clone(), persistent=True)

        if spatial is None:
            spatial = spatial_inputs(cfg)

        #clamp, baseline and rate from the config, so the model and the data
        #can never disagree on them
        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            embed_dim=primary.shape[1],
            sample_rate=float(cfg.sfreq),
            clamp=cfg.clamp,
            baseline_seconds=cfg.baseline_seconds,
            positions=spatial.get("positions"),
            pca=spatial.get("pca"),
            **cfg.model_kwargs,
        )

        #the SigLIP temperature and bias are learned, not tuned
        self.log_scale = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
        self.logit_bias = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        #the 50-word head, on the sentence transformer's output
        self.closed_head = nn.Sequential(nn.Dropout(cfg.closed_dropout), nn.Linear(self.model.d_model, VOCABULARY_SIZE))
        if prior is None:
            prior = np.full(VOCABULARY_SIZE, 1.0 / VOCABULARY_SIZE)
        self.register_buffer("log_prior", torch.log(torch.as_tensor(prior, dtype=torch.float32)), persistent=True)
        self.has_closed_head = True

        #what trains: a frozen run trains the head alone; no head loss, no head
        if cfg.freeze_backbone:
            for parameter in [*self.model.parameters(), self.log_scale, self.logit_bias]:
                parameter.requires_grad_(False)
        if cfg.closed_weight <= 0:
            for parameter in self.closed_head.parameters():
                parameter.requires_grad_(False)

        self.ema = None
        self._ema_backup = None

        #⚠️ NOT named `_buffers`, which is nn.Module's own registry
        names = ("retrieval", "closed", "targets", "isolated", "moses", "moses_targets")
        self._epoch_store = {stage: {name: [] for name in names} for stage in ("train", "val")}

    # -- plumbing ---------------------------------------------------------

    @property
    def closed_weight_at_inference(self):
        """The blend weight actually used: 0 when there is no trained head."""
        return self.cfg.inference_closed_weight if self.has_closed_head else 0.0

    def train(self, mode=True):
        """Training mode, except that a frozen backbone stays in eval mode.

        Frozen means frozen: dropout off and BatchNorm statistics fixed, so
        the head learns from exactly the features the checkpoint produces.

        ARGS:
            mode (bool)

        RETURNS:
            self
        """
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.model.train(False)
        return self

    def on_train_batch_start(self, batch, batch_idx):
        """Re-assert a frozen backbone's eval mode before every training step.

        ⚠️ Not redundant with `train`: Lightning records and restores each
        submodule's mode around validation, and the smoke test caught a frozen
        run's BatchNorm statistics drifting despite the override above.

        ARGS:
            batch (dict): unused
            batch_idx (int): unused
        """
        if self.cfg.freeze_backbone and self.model.training:
            self.model.train(False)

    def forward(self, batch):
        """
        ARGS:
            batch (dict): from `collate_tokens`

        RETURNS:
            dict from ReferenceNet.forward, plus "closed" : [N, 50] the head's
            plain logits
        """
        #a frozen backbone needs no graph: the head's gradient stops at context
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.freeze_backbone):
            out = self.model(batch["windows"], batch["lengths"], batch["sequence"], batch["position"])
        out["closed"] = self.closed_head(out["context"])
        return out

    def on_fit_start(self):
        """Build the EMA -- or move a restored one -- onto the module's device."""
        if self.cfg.ema_decay > 0 and self.ema is None:
            self.ema = WeightEMA(self.model, self.cfg.ema_decay)

        #⚠️ a resume restores the EMA before the module reaches the GPU
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

        RETURNS:
            bool: True if averaged weights were found and applied
        """
        if self.ema is None:
            return False

        #strict=False because the average holds parameters only, not buffers
        self.model.load_state_dict(self.ema.shadow, strict=False)
        return True

    def load_initial_weights(self, path):
        """Start from an earlier run's weights: everything but the new head.

        ARGS:
            path (Path): a trial 10, 12, 13 or 14 checkpoint

        RETURNS:
            str: the source run's name, for the log
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint["state_dict"]

        #⚠️ the prior is this run's own, counted from its training tokens: an
        #older checkpoint has none, and a newer one's must not override it
        state = {name: value for name, value in state.items() if name != "log_prior"}
        missing, unexpected = self.load_state_dict(state, strict=False)

        #anything missing beyond the head, or anything unexpected, is a different network
        if unexpected or set(missing) - set(HEAD_KEYS):
            raise ValueError(f"{path} does not fit this network: missing {missing}, unexpected {unexpected}")
        return checkpoint.get("hyper_parameters", {}).get("run_name", str(path))

    # -- the step ---------------------------------------------------------

    def candidate_logits(self, embedding):
        """Retrieval scores against both scored vocabularies.

        ARGS:
            embedding (torch.Tensor): [N, 1024] the network's output

        RETURNS:
            primary : [N, 50] - competition column order
            moses : [N, 50] - Moses column order
        """
        #float32 cosines whatever the autocast; rankings are what is scored
        embedding = embedding.float()
        scale = self.log_scale.exp().float()
        return (
            semantic_logits(embedding, self.primary_table, scale),
            semantic_logits(embedding, self.moses_table, scale),
        )

    def scores(self, out, closed_weight=None):
        """Every score set a word gets: retrieval, head, blend, and Moses.

        ARGS:
            out (dict): from `forward`
            closed_weight (float or None): the blend; None uses the config's

        RETURNS:
            dict of [N, 50] float32: "retrieval", "closed" (None without a
            head), "blend", "moses"
        """
        retrieval, moses = self.candidate_logits(out["embedding"])
        closed = out["closed"].float() if self.has_closed_head else None
        weight = self.closed_weight_at_inference if closed_weight is None else closed_weight
        return {"retrieval": retrieval, "closed": closed, "blend": blend_scores(retrieval, closed, weight), "moses": moses}

    def _step(self, batch, stage):
        """Shared body of the training and validation steps.

        ARGS:
            batch (dict): from `collate_tokens`
            stage (str): "train" or "val", the metric prefix

        RETURNS:
            loss : scalar tensor
        """
        out = self(batch)
        losses = {}

        #D-SigLIP: every word with a t5 row is a target, scored or not
        has_target = batch["targets"] >= 0
        if self.cfg.semantic_weight > 0 and not self.cfg.freeze_backbone and has_target.any():
            #⚠️ outside autocast: cosines times 10 in bf16 keep two significant digits
            with torch.autocast(device_type=self.device.type, enabled=False):
                losses["semantic"] = self.cfg.semantic_weight * siglip_loss(
                    out["embedding"][has_target].float(),
                    batch["targets"][has_target],
                    self.target_table,
                    self.log_scale.exp(),
                    self.logit_bias,
                    dedup=self.cfg.siglip_dedup,
                    shuffle=(stage == "train"),
                )

        #the head: logit-adjusted cross-entropy on scored words only
        scored = batch["primary"] >= 0
        if self.cfg.closed_weight > 0 and scored.any():
            with torch.autocast(device_type=self.device.type, enabled=False):
                losses["closed"] = self.cfg.closed_weight * logit_adjusted_cross_entropy(
                    out["closed"][scored].float(),
                    batch["primary"][scored],
                    self.log_prior,
                    self.cfg.logit_adjust_tau,
                    self.cfg.label_smoothing,
                )

        #nothing to learn from this batch: a zero that keeps the graph connected
        if not losses:
            trainable = [value for value in (out["embedding"], out["closed"]) if value.requires_grad]
            return trainable[0].sum() * 0.0 if trainable else torch.zeros((), device=self.device, requires_grad=True)

        total = sum(losses.values())
        size = max(int(has_target.sum()), int(scored.sum()), 1)
        self.log(f"{stage}/loss", total, prog_bar=True, on_epoch=True, batch_size=size)
        for name, value in losses.items():
            self.log(f"{stage}/loss_{name}", value, on_epoch=True, batch_size=size)

        #the epoch's scores, scored words only; detached, on the CPU
        with torch.no_grad():
            sets = self.scores(out)
            store = self._epoch_store[stage]
            store["retrieval"].append(sets["retrieval"][scored].cpu())
            if sets["closed"] is not None:
                store["closed"].append(sets["closed"][scored].cpu())
            store["targets"].append(batch["primary"][scored].cpu())
            store["isolated"].append(batch["isolated"][scored].cpu())
            in_moses = batch["moses"] >= 0
            store["moses"].append(sets["moses"][in_moses].cpu())
            store["moses_targets"].append(batch["moses"][in_moses].cpu())
        return total

    def training_step(self, batch, batch_idx):
        """One optimiser step's loss; the metrics accumulate per epoch."""
        return self._step(batch, "train")

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Fold the post-step weights into the EMA."""
        if self.ema is not None:
            self.ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        """The same step on validation sequences, no dedup shuffle."""
        return self._step(batch, "val")

    # -- evaluating on the averaged weights -------------------------------

    def on_validation_start(self):
        """Swap the averaged weights in, keeping the live ones aside."""
        if self.ema is None:
            return
        self._ema_backup = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        self.model.load_state_dict(self.ema.shadow, strict=False)

    def on_validation_end(self):
        """Put the live weights back, or training would continue from the mean."""
        if self._ema_backup is None:
            return
        self.model.load_state_dict(self._ema_backup)
        self._ema_backup = None

    # -- metrics ----------------------------------------------------------

    def _log_balanced_accuracy(self, stage):
        """Score one epoch's words: the blend, each score set alone, and Moses.

        ARGS:
            stage (str): "train" or "val"
        """
        store = self._epoch_store[stage]
        if not store["targets"]:
            return
        k = self.cfg.metric_k
        targets = torch.cat(store["targets"])
        retrieval = torch.cat(store["retrieval"])
        closed = torch.cat(store["closed"]) if store["closed"] else None

        if len(targets):
            #the headline is the blend at the run's own weight
            blend = blend_scores(retrieval, closed, self.closed_weight_at_inference)
            self.log(f"{stage}/bacc{k}", balanced_topk_accuracy(blend, targets, k, VOCABULARY_SIZE), prog_bar=(stage == "val"))
            #BAcc@1 is a tiebreaker only; its floor is 0.02, not 0.20
            self.log(f"{stage}/bacc1", balanced_topk_accuracy(blend, targets, 1, VOCABULARY_SIZE))
            self.log(f"{stage}/bacc{k}_retrieval", balanced_topk_accuracy(retrieval, targets, k, VOCABULARY_SIZE))
            if closed is not None and self.cfg.closed_weight > 0:
                self.log(f"{stage}/bacc{k}_closed", balanced_topk_accuracy(closed, targets, k, VOCABULARY_SIZE))

        #the paper's secondary, the Moses words, by retrieval
        moses = torch.cat(store["moses"])
        moses_targets = torch.cat(store["moses_targets"])
        if len(moses_targets):
            self.log(f"{stage}/moses_bacc{k}", balanced_topk_accuracy(moses, moses_targets, k, VOCABULARY_SIZE))

        #cleared every epoch, or memory grows with the whole run
        for values in store.values():
            values.clear()

    def on_train_epoch_end(self):
        """Training BAcc, logged beside validation's so the gap is visible."""
        self._log_balanced_accuracy("train")

    def on_validation_epoch_end(self):
        """Validation BAcc@10: what selects the checkpoint and stops the run."""
        self._log_balanced_accuracy("val")

    # -- optimiser --------------------------------------------------------

    def configure_optimizers(self):
        """AdamW, the new head at its own rate, linear warmup into a cosine decay.

        RETURNS:
            dict: optimiser plus a per-step scheduler
        """
        #only what trains; the head starts from random, so it moves faster
        head = [parameter for parameter in self.closed_head.parameters() if parameter.requires_grad]
        rest = [
            parameter
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and not name.startswith("closed_head.")
        ]
        groups = []
        if rest:
            groups.append({"params": rest, "lr": self.cfg.lr})
        if head:
            groups.append({"params": head, "lr": self.cfg.head_lr})
        optimiser = torch.optim.AdamW(groups, weight_decay=self.cfg.weight_decay)

        #estimated_stepping_batches accounts for max_epochs, so the cosine
        #finishes at the end of the run
        total_steps = max(int(self.trainer.estimated_stepping_batches), 1)
        warmup_steps = max(int(total_steps * self.cfg.warmup_fraction), 1)
        floor = self.cfg.min_lr / self.cfg.lr

        def scale(step):
            """Multiplier on every group's peak learning rate at `step`."""
            if step < warmup_steps:
                return (step + 1) / warmup_steps

            #cosine from 1 down to `floor` over what remains
            progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
            return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, scale)
        n_rest, n_head = sum(p.numel() for p in rest), sum(p.numel() for p in head)
        print(
            f"[optim] AdamW: {n_rest:,} parameters at lr {self.cfg.lr:g}, the head's {n_head:,} at "
            f"{self.cfg.head_lr:g}; wd {self.cfg.weight_decay:g}, {warmup_steps} warmup steps of {total_steps}"
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def load_trained(checkpoint_path, device="cpu", use_ema=True):
    """Rebuild a trained module from its checkpoint alone.

    Needs neither the reference cache nor the t5 table: the spatial layer's
    buffers and the candidate tables are all in the state dict. An older
    trial's checkpoint has no head; it loads with `has_closed_head` False.

    ARGS:
        checkpoint_path (Path or str): best.ckpt or last.ckpt
        device (str or torch.device): where the module goes
        use_ema (bool): swap in averaged weights when the checkpoint has them

    RETURNS:
        module (LitTrialFourteen): in eval mode, on `device`
        cfg (Trial14Config): the run's config, from the checkpoint
        info (dict): the checkpoint's epoch and global_step
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    #a trial 10, 12 or 13 config is a subset of trial 14's: missing fields default
    cfg = Trial14Config(**checkpoint["hyper_parameters"])
    state = checkpoint["state_dict"]

    #placeholders of the stored shapes; load_state_dict overwrites every one
    prefix = "model.encoder.spatial."
    spatial = {"positions": None, "pca": None}
    if f"{prefix}positions" in state:
        spatial["positions"] = state[f"{prefix}positions"].numpy()
    if f"{prefix}components" in state:
        spatial["pca"] = {name: state[f"{prefix}{name}"].numpy() for name in ("mean", "components", "scale")}

    #the target width from the stored candidates, not assumed to be t5-large's
    module = LitTrialFourteen(cfg, table=None, spatial=spatial, embed_dim=state["primary_table"].shape[1])

    #only the head may be missing; anything else is a different architecture
    missing, unexpected = module.load_state_dict(state, strict=False)
    if unexpected or set(missing) - set(HEAD_KEYS):
        raise ValueError(f"{checkpoint_path} does not fit: missing {missing}, unexpected {unexpected}")
    module.has_closed_head = "closed_head.1.weight" in state and cfg.closed_weight > 0
    module.on_load_checkpoint(checkpoint)
    if use_ema and module.apply_ema_weights():
        print("[load] averaged (EMA) weights applied")

    info = {"epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step")}
    return module.eval().to(device), cfg, info


@torch.no_grad()
def adapt_batch_norm(module, batches, device):
    """Re-estimate every BatchNorm's running statistics on new inputs (AdaBN).

    Only the BatchNorm layers run in training mode, so dropout and the
    merger's spatial dropout stay off; each layer's statistics become a plain
    average over `batches`, and its learned scale and shift are untouched.
    Labels are never read. Masked BatchNorm keeps padding out of the estimate.

    ARGS:
        module (LitTrialFourteen): on `device`
        batches (iterable): dicts from `collate_tokens`, the new inputs
        device (torch.device)

    RETURNS:
        int: how many BatchNorm layers were re-estimated
    """
    norms = [layer for layer in module.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        return 0

    #everything in eval mode, then the BatchNorms alone back in training mode
    module.eval()
    for norm in norms:
        norm.reset_running_stats()
        #None makes the running statistics a cumulative average over batches
        norm.momentum = None
        norm.train()

    #forward passes only: the statistics update as a side effect
    for batch in batches:
        module({name: value.to(device) for name, value in batch.items()})

    for norm in norms:
        norm.eval()
    return len(norms)
