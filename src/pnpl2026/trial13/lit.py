"""LightningModule for trial 13: D-SigLIP over every word, scored by retrieval.

BAcc@k comes from `pnpl2026.metrics`, the one implementation in the repo.

    loss      D-SigLIP of each word's output against its own t5 embedding,
              negatives the other words of the batch -- every word a token
    metrics   retrieval: cosine against the 50 competition words' embeddings
              (val/bacc10, the scored one, and val/bacc1), and against the 50
              Moses words (val/moses_bacc10, the paper's secondary)

Words outside a vocabulary still train and still give the others context;
they are simply not scored.

⚠️ Two tables, kept differently:

    `target_table`   every word's row, ~100 MB. Training only, so NOT
                     checkpointed -- it would triple every checkpoint
    `primary_table`, `moses_table`   the 50 + 50 candidates. Checkpointed:
                     prediction needs them and nothing else

`load_trained` rebuilds a module from a checkpoint alone, with no cache and no
t5 table on the machine -- trial 13's, or trial 10's and 12's, whose saved
configs and state dicts this module reads unchanged.

`adapt_batch_norm` re-estimates every BatchNorm's running statistics on new
inputs (AdaBN), labels never involved. Off unless a CLI is asked for it.
"""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from ..metrics import balanced_topk_accuracy
from .config import Trial13Config
from .data.dataset import load_manifest
from .losses import semantic_logits, siglip_loss
from .models import T5_WIDTH, build_model

#SigLIP's initialisation: t = 10, b = -10, per the paper
INITIAL_LOG_SCALE = math.log(10.0)
INITIAL_LOGIT_BIAS = -10.0

#the scored vocabularies' size; both are 50 words
VOCABULARY_SIZE = 50


def spatial_inputs(cfg):
    """What the spatial layer is built from, read from the reference cache.

    ARGS:
        cfg (Trial13Config)

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

    Parameters only, not buffers. Off in the reference (ema_decay 0).

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


class LitTrialThirteen(L.LightningModule):
    """Wraps the reference network with D-SigLIP, retrieval metrics and AdamW.

    ATTRS:
        cfg (Trial13Config)
        model (ReferenceNet)
        target_table (torch.Tensor): [n_words, 1024] buffer, not checkpointed
        primary_table (torch.Tensor): [50, 1024] buffer, the scored candidates
        moses_table (torch.Tensor): [50, 1024] buffer, the secondary's
        log_scale, logit_bias (nn.Parameter): SigLIP's t and b
        ema (WeightEMA or None)
    """

    def __init__(self, cfg, table=None, spatial=None, embed_dim=T5_WIDTH):
        """
        ARGS:
            cfg (Trial13Config): the merged run config
            table (TargetTable or None): None only when a checkpoint is about
                to overwrite the candidate tables -- see `load_trained`
            spatial (dict or None): from `spatial_inputs`; None reads the cache
            embed_dim (int): the placeholders' width when `table` is None
        """
        super().__init__()
        self.save_hyperparameters(cfg.to_dict())
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

        self.ema = None
        self._ema_backup = None

        #⚠️ NOT named `_buffers`, which is nn.Module's own registry
        self._epoch_store = {
            stage: {"primary": [], "primary_targets": [], "moses": [], "moses_targets": [], "isolated": []}
            for stage in ("train", "val")
        }

    # -- plumbing ---------------------------------------------------------

    def forward(self, batch):
        """
        ARGS:
            batch (dict): from `collate_tokens`

        RETURNS:
            dict from ReferenceNet.forward
        """
        return self.model(batch["windows"], batch["lengths"], batch["sequence"], batch["position"])

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

    def _step(self, batch, stage):
        """Shared body of the training and validation steps.

        ARGS:
            batch (dict): from `collate_tokens`
            stage (str): "train" or "val", the metric prefix

        RETURNS:
            loss : scalar tensor
        """
        out = self(batch)
        embedding = out["embedding"]

        #every word with a t5 row is a target, scored or not
        has_target = batch["targets"] >= 0
        n_targets = int(has_target.sum())

        #a batch with no target at all has nothing to contrast; a zero that
        #keeps the graph connected, rather than a crash in the dedup
        if n_targets == 0:
            return embedding.sum() * 0.0

        #⚠️ outside autocast: cosines times 10 in bf16 keep two significant digits
        with torch.autocast(device_type=self.device.type, enabled=False):
            loss = siglip_loss(
                embedding[has_target].float(),
                batch["targets"][has_target],
                self.target_table,
                self.log_scale.exp(),
                self.logit_bias,
                dedup=self.cfg.siglip_dedup,
                shuffle=(stage == "train"),
            )
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=n_targets)

        #retrieval scores for the epoch's metrics, on scored words only
        with torch.no_grad():
            primary_logits, moses_logits = self.candidate_logits(embedding.detach())
            store = self._epoch_store[stage]
            scored = batch["primary"] >= 0
            store["primary"].append(primary_logits[scored].cpu())
            store["primary_targets"].append(batch["primary"][scored].cpu())
            store["isolated"].append(batch["isolated"][scored].cpu())
            moses = batch["moses"] >= 0
            store["moses"].append(moses_logits[moses].cpu())
            store["moses_targets"].append(batch["moses"][moses].cpu())
        return loss

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
        """Score one epoch's words against both vocabularies.

        ARGS:
            stage (str): "train" or "val"
        """
        store = self._epoch_store[stage]
        if not store["primary_targets"]:
            return
        k = self.cfg.metric_k

        #BAcc@1 is a tiebreaker only; its floor is 0.02, not 0.20
        primary = torch.cat(store["primary"])
        targets = torch.cat(store["primary_targets"])
        if len(targets):
            self.log(f"{stage}/bacc1", balanced_topk_accuracy(primary, targets, 1, VOCABULARY_SIZE))
            self.log(
                f"{stage}/bacc{k}",
                balanced_topk_accuracy(primary, targets, k, VOCABULARY_SIZE),
                prog_bar=(stage == "val"),
            )

        #the paper's secondary, the Moses words
        moses = torch.cat(store["moses"])
        moses_targets = torch.cat(store["moses_targets"])
        if len(moses_targets):
            self.log(
                f"{stage}/moses_bacc{k}",
                balanced_topk_accuracy(moses, moses_targets, k, VOCABULARY_SIZE),
            )

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
        """AdamW with linear warmup into a cosine decay, as the reference.

        RETURNS:
            dict: optimiser plus a per-step scheduler
        """
        optimiser = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        #estimated_stepping_batches accounts for max_epochs, so the cosine
        #finishes at the end of the run
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
            f"[optim] AdamW lr {self.cfg.lr:g} wd {self.cfg.weight_decay:g}, {warmup_steps} warmup "
            f"steps of {total_steps}, cosine to {self.cfg.min_lr:g}"
        )
        return {
            "optimizer": optimiser,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def load_trained(checkpoint_path, device="cpu", use_ema=True):
    """Rebuild a trained module from its checkpoint alone.

    Needs neither the reference cache nor the t5 table: the spatial layer's
    buffers and the candidate tables are all in the state dict.

    ARGS:
        checkpoint_path (Path or str): best.ckpt or last.ckpt
        device (str or torch.device): where the module goes
        use_ema (bool): swap in averaged weights when the checkpoint has them

    RETURNS:
        module (LitTrialThirteen): in eval mode, on `device`
        cfg (Trial13Config): the run's config, from the checkpoint
        info (dict): the checkpoint's epoch and global_step
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    #a trial 10 or 12 config is a subset of trial 13's: missing fields default
    cfg = Trial13Config(**checkpoint["hyper_parameters"])
    state = checkpoint["state_dict"]

    #placeholders of the stored shapes; load_state_dict overwrites every one
    prefix = "model.encoder.spatial."
    spatial = {"positions": None, "pca": None}
    if f"{prefix}positions" in state:
        spatial["positions"] = state[f"{prefix}positions"].numpy()
    if f"{prefix}components" in state:
        spatial["pca"] = {name: state[f"{prefix}{name}"].numpy() for name in ("mean", "components", "scale")}

    #the target width from the stored candidates, not assumed to be t5-large's
    module = LitTrialThirteen(cfg, table=None, spatial=spatial, embed_dim=state["primary_table"].shape[1])

    #strict: a missing or unexpected key here is a different architecture
    module.load_state_dict(state, strict=True)
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
        module (LitTrialThirteen): on `device`
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
