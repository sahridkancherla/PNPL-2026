"""LightningModule for trial 21 (trial 20's): the continuous encoder, D-SigLIP, retrieval, optional heads.

Trial 21 adds `warm_start_encoder`: a trained per-word checkpoint's encoder
(trials 10-18's `WordEncoder`) loaded into a dascoli-stem `ContinuousNet`,
module by module, so the hybrid starts where G2's features are.

BAcc@k comes from `pnpl2026.metrics`, the one implementation in the repo.

    loss      D-SigLIP of each word's t5 head against its own t5 embedding,
              negatives the other words of the batch -- every word a token;
              plus, when on: Variant K's second head (its own table), the
              word tower's D-SigLIP in the shared space, and the phoneme
              head's cross-entropy per step
    metrics   retrieval: cosine against the 50 competition words' embeddings
              (val/bacc10, the scored one, and val/bacc1), and against the 50
              Moses words (val/moses_bacc10, the paper's secondary)

Words outside a vocabulary still train and still give the others context;
they are simply not scored.

⚠️ Tables, kept differently:

    `target_table`, `description_table`   every word's row. Training only, so
                     NOT checkpointed -- they would multiply every checkpoint
    `primary_table`, `moses_table`, `primary_descriptions`,
    `moses_descriptions`   the 50 + 50 candidates. Checkpointed: prediction
                     needs them and nothing else

`candidate_scores` is what everything downstream ranks by: with one head,
exactly `candidate_logits`; with several, each head's 50 scores z-scored per
row and averaged -- the ensemble script's rule, inside one model -- or the
tower alone (`tower_score_mode`).

`load_trained` rebuilds a module from a checkpoint alone, with no cache and no
tables on the machine. `adapt_batch_norm` is kept for the scoring tools; this
family has no BatchNorm, so it is a no-op.
"""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import balanced_topk_accuracy
from .config import DEFAULT_TOWER_WEIGHT, Trial23Config
from .data.dataset import load_manifest
from .losses import semantic_logits, siglip_loss
from .models import T5_WIDTH, WordTower, build_model, step_label_indices

#SigLIP's initialisation: t = 10, b = -10, per the paper
INITIAL_LOG_SCALE = math.log(10.0)
INITIAL_LOGIT_BIAS = -10.0

#the scored vocabularies' size; both are 50 words
VOCABULARY_SIZE = 50


def sensor_type_ids(channel_names):
    """Each sensor's kind from its Neuromag name: 0 magnetometer, 1 and 2 gradiometers.

    ARGS:
        channel_names (list or None): e.g. "MEG0113"; the last digit is the kind

    RETURNS:
        [n_channels] int64 numpy, or None when a name does not end in 1-3
    """
    if not channel_names:
        return None
    kinds = []
    for name in channel_names:
        last = str(name).strip()[-1:]
        if last not in ("1", "2", "3"):
            return None
        kinds.append(int(last) - 1)
    return np.array(kinds, dtype=np.int64)


def spatial_inputs(cfg):
    """What the spatial layer is built from, read from the reference cache.

    ARGS:
        cfg (Trial23Config)

    RETURNS:
        dict: "positions" [306, 2] or None, "sensor_types" [306] or None
    """
    cache_dir = cfg.resolved_cache_dir()
    manifest = load_manifest(cache_dir)

    #positions exist only when the h5 files carried channel names
    positions = None
    if manifest.get("has_positions"):
        positions = np.load(cache_dir / "positions.npy")
    return {"positions": positions, "sensor_types": sensor_type_ids(manifest.get("channel_names"))}


def standardise_rows(scores):
    """Each row's scores to zero mean and unit spread, across the candidates.

    ARGS:
        scores (torch.Tensor): [N, n_candidates]

    RETURNS:
        standardised : [N, n_candidates]
        spread : [N, 1] - each row's spread before standardising
    """
    #dim=1 is the candidate axis: each word is ranked on its own
    centred = scores - scores.mean(dim=1, keepdim=True)
    spread = centred.std(dim=1, keepdim=True)

    #a flat row has no ranking; leave it flat rather than divide by 0
    safe = torch.where(spread > 0, spread, torch.ones_like(spread))
    return centred / safe, spread


def combine_heads(*heads, weights=None):
    """Several heads' scores as one set.

    The ensemble script's rule: z-score each head per row, average. The result
    is rescaled to the heads' mean spread, so a softmax over it is about as
    sharp as any one head's; that changes no ranking.

    ARGS:
        *heads (torch.Tensor): each [N, n_candidates]
        weights (list or None): one share per head, summing to 1; None is
            the plain average, computed exactly as before trial 23

    RETURNS:
        [N, n_candidates]
    """
    standardised, spreads = zip(*(standardise_rows(head) for head in heads))
    spread = torch.stack(spreads, dim=0).mean(dim=0)

    #dim=0 is the head axis
    if weights is None:
        return torch.stack(standardised, dim=0).mean(dim=0) * spread
    shares = torch.tensor(weights, dtype=standardised[0].dtype, device=standardised[0].device).view(-1, 1, 1)
    return (torch.stack(standardised, dim=0) * shares).sum(dim=0) * spread


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

    def load_state_dict(self, state):
        """Restore the average from a checkpoint.

        ARGS:
            state (dict): name -> tensor
        """
        self.shadow = {name: value.clone() for name, value in state.items()}

    def to(self, device):
        """Move the average to `device`, returning self.

        ARGS:
            device (torch.device or str)
        """
        self.shadow = {name: value.to(device) for name, value in self.shadow.items()}
        return self


class LitTrialTwentyThree(L.LightningModule):
    """Wraps the continuous network with D-SigLIP, retrieval metrics and AdamW.

    ATTRS:
        cfg (Trial23Config)
        model (ContinuousNet)
        target_table (torch.Tensor): [n_words, 1024] buffer, not checkpointed
        primary_table (torch.Tensor): [50, 1024] buffer, the scored candidates
        moses_table (torch.Tensor): [50, 1024] buffer, the secondary's
        log_scale, logit_bias (nn.Parameter): SigLIP's t and b
        target_table_second, primary_table_second, moses_table_second,
        log_scale_second, logit_bias_second: the same for Variant K's second
            head; present only when `cfg.has_second_head`
        tower (WordTower or None), description_table [n_words, total] (not
            checkpointed), primary_descriptions, moses_descriptions [50, total],
            tower_widths, log_scale_tower, logit_bias_tower: the word tower's;
            present only when `cfg.has_tower`
        ema (WeightEMA or None)
    """

    def __init__(
        self,
        cfg,
        table=None,
        spatial=None,
        embed_dim=T5_WIDTH,
        second_table=None,
        second_embed_dim=T5_WIDTH,
        descriptions=None,
        description_widths=None,
    ):
        """
        ARGS:
            cfg (Trial23Config): the merged run config
            table (TargetTable or None): None only when a checkpoint is about
                to overwrite the candidate tables -- see `load_trained`
            spatial (dict or None): from `spatial_inputs`; None reads the cache
            embed_dim (int): the placeholders' width when `table` is None
            second_table (TargetTable or None): the second head's table; None
                with one head, or when a checkpoint will overwrite it
            second_embed_dim (int): its placeholders' width
            descriptions (dict or None): from `models.tower.load_descriptions`;
                None when a checkpoint will overwrite the candidates' rows
            description_widths (list or None): the blocks' widths when
                `descriptions` is None, from the checkpoint
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

        #Variant K: the second head's tables, kept exactly as the first's
        if cfg.has_second_head:
            if second_table is not None:
                if second_table.words != table.words:
                    raise ValueError("The two target tables list different words; batch targets index both.")
                full = second_table.table
                primary, moses = full[second_table.primary_rows], full[second_table.moses_rows]
            else:
                full = torch.zeros(0, second_embed_dim)
                primary = torch.zeros(VOCABULARY_SIZE, second_embed_dim)
                moses = torch.zeros(VOCABULARY_SIZE, second_embed_dim)
            self.register_buffer("target_table_second", full, persistent=False)
            self.register_buffer("primary_table_second", primary.clone(), persistent=True)
            self.register_buffer("moses_table_second", moses.clone(), persistent=True)

        #the word tower: every word's descriptions train; the candidates' rows are kept
        self.tower = None
        if cfg.has_tower:
            if descriptions is not None:
                widths = list(descriptions["widths"])
                full = descriptions["table"]
                primary, moses = full[table.primary_rows], full[table.moses_rows]
            else:
                widths = list(description_widths or [])
                if not widths:
                    raise ValueError("A tower needs its block widths: pass descriptions or description_widths.")
                full = torch.zeros(0, sum(widths))
                primary = torch.zeros(VOCABULARY_SIZE, sum(widths))
                moses = torch.zeros(VOCABULARY_SIZE, sum(widths))
            self.register_buffer("description_table", full, persistent=False)
            self.register_buffer("primary_descriptions", primary.clone(), persistent=True)
            self.register_buffer("moses_descriptions", moses.clone(), persistent=True)
            self.register_buffer("tower_widths", torch.tensor(widths, dtype=torch.int64), persistent=True)
            self.tower = WordTower(widths, cfg.tower_hidden, cfg.tower_dim)

        if spatial is None:
            spatial = spatial_inputs(cfg)

        #clamp, baseline and rate from the config, so the model and the data
        #can never disagree on them
        self.model = build_model(
            cfg.model,
            n_channels=cfg.n_channels,
            embed_dim=self.primary_table.shape[1],
            sample_rate=float(cfg.sfreq),
            clamp=cfg.clamp,
            baseline_seconds=cfg.baseline_seconds,
            positions=spatial.get("positions"),
            sensor_types=spatial.get("sensor_types"),
            second_head=cfg.has_second_head,
            second_embed_dim=self.primary_table_second.shape[1] if cfg.has_second_head else T5_WIDTH,
            tower_dim=cfg.tower_dim if cfg.has_tower else 0,
            phoneme_head=cfg.has_phoneme_head,
            n_subjects=cfg.n_subjects,
            **cfg.model_kwargs,
        )

        #the SigLIP temperature and bias are learned, not tuned
        self.log_scale = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
        self.logit_bias = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        #each extra head learns its own: its cosines need not match the first's
        if cfg.has_second_head:
            self.log_scale_second = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
            self.logit_bias_second = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))
        if cfg.has_tower:
            self.log_scale_tower = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
            self.logit_bias_tower = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        self.ema = None
        self._ema_backup = None

        #⚠️ NOT named `_buffers`, which is nn.Module's own registry
        #"primary_first" is the t5 head alone, filled only with several heads
        self._epoch_store = {
            stage: {"primary": [], "primary_first": [], "primary_targets": [], "moses": [], "moses_targets": [], "isolated": []}
            for stage in ("train", "val")
        }

    # -- plumbing ---------------------------------------------------------

    def forward(self, batch):
        """
        ARGS:
            batch (dict): from `collate_items`

        RETURNS:
            dict from ContinuousNet.forward
        """
        #prediction's batches name no subject: subject 0 throughout
        return self.model(
            batch["signal"],
            batch["lengths"],
            batch["onset"],
            batch["marker"],
            batch["sequence"],
            batch.get("item_subject"),
            batch.get("position"),
        )

    @property
    def n_heads(self):
        """How many heads `candidate_scores` combines."""
        return 1 + int(self.cfg.has_second_head) + int(self.cfg.has_tower)

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

    # -- the heads --------------------------------------------------------

    def candidate_logits(self, embedding):
        """The t5 head's retrieval scores against both scored vocabularies.

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

    def candidate_logits_second(self, embedding):
        """The second head's retrieval scores, as `candidate_logits`.

        ARGS:
            embedding (torch.Tensor): [N, embed_dim] the second head's output

        RETURNS:
            primary : [N, 50], moses : [N, 50]
        """
        embedding = embedding.float()
        scale = self.log_scale_second.exp().float()
        return (
            semantic_logits(embedding, self.primary_table_second, scale),
            semantic_logits(embedding, self.moses_table_second, scale),
        )

    def candidate_logits_tower(self, brain):
        """The tower head's scores: the candidates through the tower, then cosines.

        ARGS:
            brain (torch.Tensor): [N, tower_dim] the brain side in the shared space

        RETURNS:
            primary : [N, 50], moses : [N, 50]
        """
        brain = brain.float()
        scale = self.log_scale_tower.exp().float()
        return (
            semantic_logits(brain, self.tower(self.primary_descriptions), scale),
            semantic_logits(brain, self.tower(self.moses_descriptions), scale),
        )

    def candidate_scores(self, out):
        """What every ranking, metric and prediction reads.

        ARGS:
            out (dict): from `forward`

        RETURNS:
            primary : [N, 50], moses : [N, 50] - with one head, exactly
            `candidate_logits`; with several, their row-standardised scores
            averaged, rescaled to their mean spread; with `tower_score_mode`
            "tower", the tower head alone. `tower_weight` other than 0.5
            gives the tower that share of the average (trial 23)
        """
        primary, moses = self.candidate_logits(out["embedding"])
        primaries, moseses = [primary], [moses]
        if self.cfg.has_second_head:
            primary_second, moses_second = self.candidate_logits_second(out["embedding_second"])
            primaries.append(primary_second)
            moseses.append(moses_second)
        if self.cfg.has_tower:
            primary_tower, moses_tower = self.candidate_logits_tower(out["brain_tower"])
            if self.cfg.tower_score_mode == "tower":
                return primary_tower, moses_tower
            primaries.append(primary_tower)
            moseses.append(moses_tower)
        if len(primaries) == 1:
            return primary, moses
        weights = self.head_weights(len(primaries))
        return combine_heads(*primaries, weights=weights), combine_heads(*moseses, weights=weights)

    def head_weights(self, n_heads):
        """Each scoring head's share of the combined score, the tower's last.

        ARGS:
            n_heads (int): the scoring heads, the tower among them if any

        RETURNS:
            list or None - None (the plain average) without a tower or at the
            default weight; else the tower `tower_weight`, the rest equally
        """
        if not self.cfg.has_tower or self.cfg.tower_weight == DEFAULT_TOWER_WEIGHT:
            return None
        rest = (1.0 - self.cfg.tower_weight) / (n_heads - 1)
        return [rest] * (n_heads - 1) + [self.cfg.tower_weight]

    # -- the step ---------------------------------------------------------

    def phoneme_loss(self, out, batch):
        """Cross-entropy of the phoneme head against each step's lagged label.

        ARGS:
            out (dict): from `forward`, with "phoneme_logits" [B, T', C]
            batch (dict): with "phonemes" [B, T], -1 where unknown

        RETURNS:
            scalar tensor; zero (graph-connected) when no step has a label
        """
        logits = out["phoneme_logits"].float()
        n_steps = logits.shape[1]

        #each step predicts the sample at its centre, the brain's lag earlier
        indices = step_label_indices(n_steps, self.model.stride, self.cfg.samples(self.cfg.phoneme_lag_seconds))
        indices = indices.to(batch["phonemes"].device)

        #the last step's centre can sit in the padding past an odd length
        labels = batch["phonemes"][:, indices.clamp(min=0, max=batch["phonemes"].shape[1] - 1)]
        usable = (indices >= 0).unsqueeze(0) & out["step_valid"]
        labels = torch.where(usable, labels, torch.full_like(labels, -1))
        if int((labels >= 0).sum()) == 0:
            return logits.sum() * 0.0

        #dim=-1 is the class axis; flattened over items and steps
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-1)

    def _step(self, batch, stage):
        """Shared body of the training and validation steps.

        ARGS:
            batch (dict): from `collate_items`
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
        labels = batch["targets"][has_target]
        parts = {}

        #⚠️ outside autocast: cosines times 10 in bf16 keep two significant digits
        with torch.autocast(device_type=self.device.type, enabled=False):
            parts["t5"] = siglip_loss(
                embedding[has_target].float(),
                labels,
                self.target_table,
                self.log_scale.exp(),
                self.logit_bias,
                dedup=self.cfg.siglip_dedup,
                shuffle=(stage == "train"),
            )
            loss = parts["t5"] * (self.cfg.tower_anchor_weight if self.cfg.has_tower else 1.0)

            #Variant K: the same words, deduplicated afresh, in the second space
            if self.cfg.has_second_head:
                parts["second"] = siglip_loss(
                    out["embedding_second"][has_target].float(),
                    labels,
                    self.target_table_second,
                    self.log_scale_second.exp(),
                    self.logit_bias_second,
                    dedup=self.cfg.siglip_dedup,
                    shuffle=(stage == "train"),
                )
                loss = loss + self.cfg.second_loss_weight * parts["second"]

            #the tower: only the batch's distinct words go through it, as a
            #small target table the relabelled words index
            if self.cfg.has_tower:
                distinct, relabelled = torch.unique(labels, return_inverse=True)
                parts["tower"] = siglip_loss(
                    out["brain_tower"][has_target].float(),
                    relabelled,
                    self.tower(self.description_table[distinct]),
                    self.log_scale_tower.exp(),
                    self.logit_bias_tower,
                    dedup=self.cfg.siglip_dedup,
                    shuffle=(stage == "train"),
                )
                loss = loss + parts["tower"]

            #the phoneme head: a teacher on the stem, nothing at test time
            if self.cfg.has_phoneme_head:
                parts["phonemes"] = self.phoneme_loss(out, batch)
                loss = loss + self.cfg.phoneme_weight * parts["phonemes"]

        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=n_targets)
        if len(parts) > 1:
            for name, value in parts.items():
                self.log(f"{stage}/loss_{name}", value, on_epoch=True, batch_size=n_targets)

        #retrieval scores for the epoch's metrics, on scored words only
        with torch.no_grad():
            detached = {name: value.detach() for name, value in out.items()}
            primary_logits, moses_logits = self.candidate_scores(detached)
            store = self._epoch_store[stage]
            scored = batch["primary"] >= 0
            store["primary"].append(primary_logits[scored].cpu())
            if self.n_heads > 1:
                store["primary_first"].append(self.candidate_logits(detached["embedding"])[0][scored].cpu())
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
        """The same step on validation items, no dedup shuffle."""
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

        #the t5 head alone, to see what the other heads add
        if store["primary_first"] and len(targets):
            first = torch.cat(store["primary_first"])
            self.log(f"{stage}/bacc{k}_first", balanced_topk_accuracy(first, targets, k, VOCABULARY_SIZE))

        #the paper's secondary, the Moses words
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
        """AdamW with linear warmup into a cosine decay, as the reference.

        RETURNS:
            dict: optimiser plus a per-step scheduler
        """
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimiser = torch.optim.AdamW(trainable, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

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
        return {"optimizer": optimiser, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}


#the per-word encoder's modules that the dascoli stem carries under the same names
WARM_START_MODULES = ("spatial", "initial_linear", "subject", "convs", "final")

#the head and SigLIP parameters a warm start may also take
WARM_START_HEAD = ("model.project.weight", "model.project.bias", "log_scale", "logit_bias")


def warm_start_encoder(module, checkpoint_path, with_head=True):
    """Load a per-word checkpoint's encoder into a dascoli-stem continuous model.

    The per-word network keeps its encoder under `model.encoder.<module>`;
    the hybrid keeps the same modules under `model.<module>`. Every tensor
    whose name and shape match is copied -- the merger's heads (its Fourier
    features are a fixed function of the same sensor positions), the 1x1,
    the subject layer, the ten conv layers with their BatchNorm statistics,
    and the final 1x1 pair. The per-word attention pool is not taken: the
    hybrid pools steps around an onset, a different thing. The checkpoint's
    EMA weights are preferred when it carries them, as `load_trained` does.

    ARGS:
        module (LitTrialTwentyThree): freshly built, `model.stem_kind` "dascoli"
        checkpoint_path (Path or str): a trial 10-18 best.ckpt
        with_head (bool): also copy the t5 head and SigLIP's scale and bias

    RETURNS:
        dict: "loaded" (list of keys copied), "skipped" (name -> reason),
        "missing" (encoder keys the checkpoint did not provide)
    """
    if module.model.stem_kind != "dascoli":
        raise ValueError("warm_start_encoder needs model_kwargs.stem: dascoli.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = dict(checkpoint["state_dict"])

    #averaged parameters, when the base trained with an EMA: its shadow is keyed
    #without the "model." prefix
    for name, value in (checkpoint.get("weight_ema") or {}).items():
        state[f"model.{name}"] = value

    own = module.state_dict()
    wanted = {}
    skipped = {}
    for key, value in state.items():
        prefix = "model.encoder."
        if key.startswith(prefix):
            target = "model." + key[len(prefix):]
            if target.split(".")[1] not in WARM_START_MODULES:
                skipped[key] = "not an encoder module the stem carries"
                continue
        elif with_head and key in WARM_START_HEAD:
            target = key
        else:
            continue
        if target not in own:
            skipped[key] = "no such tensor in the hybrid"
        elif tuple(own[target].shape) != tuple(value.shape):
            skipped[key] = f"shape {tuple(value.shape)} against {tuple(own[target].shape)}"
        else:
            wanted[target] = value

    #strict=False: everything else keeps its fresh initialisation
    module.load_state_dict(wanted, strict=False)
    #trial 21's own additions to the merger have no counterpart in a per-word checkpoint
    missing = [
        key for key in own
        if key.split(".")[0] == "model" and key.split(".")[1] in WARM_START_MODULES and key not in wanted
        and not key.endswith(("type_bias", "sensor_types"))
    ]
    return {"loaded": sorted(wanted), "skipped": skipped, "missing": missing}


def load_trained(checkpoint_path, device="cpu", use_ema=True):
    """Rebuild a trained module from its checkpoint alone.

    Needs neither the reference cache nor any table: the spatial layer's
    buffers, the candidate tables and the candidates' descriptions are all in
    the state dict.

    ARGS:
        checkpoint_path (Path or str): best.ckpt or last.ckpt
        device (str or torch.device): where the module goes
        use_ema (bool): swap in averaged weights when the checkpoint has them

    RETURNS:
        module (LitTrialTwentyThree): in eval mode, on `device`
        cfg (Trial23Config): the run's config, from the checkpoint
        info (dict): the checkpoint's epoch and global_step
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = Trial23Config(**checkpoint["hyper_parameters"])
    state = checkpoint["state_dict"]

    #placeholders of the stored shapes; load_state_dict overwrites every one
    prefix = "model.spatial."
    spatial = {"positions": None, "sensor_types": None}
    if f"{prefix}positions" in state:
        spatial["positions"] = state[f"{prefix}positions"].numpy()
    if f"{prefix}sensor_types" in state:
        spatial["sensor_types"] = state[f"{prefix}sensor_types"].numpy()

    #the target widths and the tower's blocks from the stored candidates
    second_width = state["primary_table_second"].shape[1] if "primary_table_second" in state else T5_WIDTH
    widths = state["tower_widths"].tolist() if "tower_widths" in state else None
    module = LitTrialTwentyThree(
        cfg,
        table=None,
        spatial=spatial,
        embed_dim=state["primary_table"].shape[1],
        second_embed_dim=second_width,
        description_widths=widths,
    )

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

    The megstate stem has no BatchNorm, so for it this returns 0 without
    touching anything; the dascoli stem's conv stack has one per layer.

    ARGS:
        module (LitTrialTwentyThree): on `device`
        batches (iterable): dicts from `collate_items`
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
        norm.momentum = None
        norm.train()
    for batch in batches:
        module({name: value.to(device) for name, value in batch.items()})
    for norm in norms:
        norm.eval()
    return len(norms)
