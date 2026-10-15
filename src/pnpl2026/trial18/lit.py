"""LightningModule for trial 18: trial 17's, plus Variant M's refinement (off by default).

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
t5 table on the machine -- trial 17's, or trial 10's, 12's, 13's, 15's and 16's, whose
saved configs and state dicts this module reads unchanged.

Variant K (`cfg.second_embeddings` set): a second head is trained with its own
D-SigLIP, temperature and bias against a second all-word table, and the loss is
loss_first + second_loss_weight * loss_second. Its tables are kept as the first
head's are. `candidate_scores` is what everything downstream ranks by: with one
head, exactly `candidate_logits`; with two, each head's 50 scores z-scored per
row and averaged -- the ensemble script's rule, inside one model.

`adapt_batch_norm` re-estimates every BatchNorm's running statistics on new
inputs (AdaBN), labels never involved. Off unless a CLI is asked for it.

Variant M (`cfg.refine_base` set): the network carries HDND's refiner, and
`cli.train` loads the base checkpoint's weights and calls `freeze_base`, so
only the refiner learns while the base runs exactly as at inference. Loss:

    D-SigLIP of the REFINED prediction, as the base was trained
    + refine_kl_weight   * KL(base || refined) over the 50 scored words,
                           the anchor that keeps the ranking near the base's
    + refine_pair_weight * softplus(-t (delta - delta_shuffled) . target),
                           the refiner's residual from the word's own features
                           must point at its target more than the residual
                           from another word's features (HDND's pairing loss;
                           t the base's frozen SigLIP scale)

`calibrate_refiner` fills its standardisation statistics from training
batches before it trains; `strength_curve` scores validation at each
strength once it has.
"""

from __future__ import annotations

import math

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import balanced_topk_accuracy
from .config import Trial18Config
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
        cfg (Trial18Config)

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


def combine_heads(first, second):
    """Variant K's two heads as one set of scores.

    The ensemble script's rule: z-score each head per row, average. The result
    is rescaled to the heads' mean spread, so a softmax over it is about as
    sharp as either head's; that changes no ranking.

    ARGS:
        first (torch.Tensor): [N, n_candidates], the first head's scaled cosines
        second (torch.Tensor): [N, n_candidates], the second head's

    RETURNS:
        [N, n_candidates]
    """
    first_standard, first_spread = standardise_rows(first)
    second_standard, second_spread = standardise_rows(second)
    return 0.5 * (first_standard + second_standard) * 0.5 * (first_spread + second_spread)


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


class LitTrialEighteen(L.LightningModule):
    """Wraps the reference network with D-SigLIP, retrieval metrics and AdamW.

    ATTRS:
        cfg (Trial18Config)
        model (ReferenceNet)
        target_table (torch.Tensor): [n_words, 1024] buffer, not checkpointed
        primary_table (torch.Tensor): [50, 1024] buffer, the scored candidates
        moses_table (torch.Tensor): [50, 1024] buffer, the secondary's
        log_scale, logit_bias (nn.Parameter): SigLIP's t and b
        target_table_second, primary_table_second, moses_table_second,
        log_scale_second, logit_bias_second: the same for Variant K's second
            head; present only when `cfg.has_second_head`
        ema (WeightEMA or None)
    """

    def __init__(self, cfg, table=None, spatial=None, embed_dim=T5_WIDTH, second_table=None, second_embed_dim=T5_WIDTH):
        """
        ARGS:
            cfg (Trial18Config): the merged run config
            table (TargetTable or None): None only when a checkpoint is about
                to overwrite the candidate tables -- see `load_trained`
            spatial (dict or None): from `spatial_inputs`; None reads the cache
            embed_dim (int): the placeholders' width when `table` is None
            second_table (TargetTable or None): the second head's table; None
                with one head, or when a checkpoint will overwrite it
            second_embed_dim (int): its placeholders' width
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
            pca=spatial.get("pca"),
            second_head=cfg.has_second_head,
            #the second head as wide as its own table, which need not be t5's
            second_embed_dim=self.primary_table_second.shape[1] if cfg.has_second_head else None,
            refine=cfg.refine_settings(),
            n_subjects=cfg.n_subjects,
            **cfg.model_kwargs,
        )

        #the SigLIP temperature and bias are learned, not tuned
        self.log_scale = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
        self.logit_bias = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        #the second head learns its own: its cosines need not match the first's
        if cfg.has_second_head:
            self.log_scale_second = torch.nn.Parameter(torch.tensor(INITIAL_LOG_SCALE))
            self.logit_bias_second = torch.nn.Parameter(torch.tensor(INITIAL_LOGIT_BIAS))

        self.ema = None
        self._ema_backup = None

        #Variant M: set by `freeze_base`, after the base's weights are loaded
        self.base_frozen = False

        #⚠️ NOT named `_buffers`, which is nn.Module's own registry
        #"primary_first" is the first head alone, filled only with two heads
        self._epoch_store = {
            stage: {
                "primary": [], "primary_first": [], "primary_base": [], "primary_targets": [],
                "moses": [], "moses_targets": [], "isolated": [],
            }
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
        #all-data batches name each window's subject; prediction's do not (subject 0)
        return self.model(batch["windows"], batch["lengths"], batch["sequence"], batch["position"], batch.get("subject"))

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

    # -- Variant M: refining a frozen base --------------------------------

    def freeze_base(self):
        """Train the refiner alone: every other weight fixed, the base as at inference."""
        if self.model.refiner is None:
            raise ValueError("freeze_base needs a refiner: set cfg.refine_base.")

        #the SigLIP scale and bias stay the base's too
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith("model.refiner."))
        self.base_frozen = True
        self.train(self.training)

    def train(self, mode=True):
        """Training mode, except a frozen base's BatchNorm and dropout, kept as at inference.

        ARGS:
            mode (bool): True for training

        RETURNS:
            self
        """
        super().train(mode)

        #running statistics, not batch ones, and no dropout, in the frozen part
        if self.base_frozen and mode:
            self.model.eval()
            self.model.refiner.train()
        return self

    @torch.no_grad()
    def calibrate_refiner(self, loader, n_batches, device):
        """Fill the refiner's standardisation statistics from training batches.

        ARGS:
            loader (DataLoader): training batches, shaped as the base trained
            n_batches (int): how many to measure
            device (torch.device or str)

        RETURNS:
            int: words measured
        """
        was_training = self.training
        self.eval().to(device)

        #float64 running sums, so the features never have to be held at once
        sums = {}
        for index, batch in enumerate(loader):
            if index >= n_batches:
                break
            out = self({name: value.to(device) for name, value in batch.items()})
            for name, values in (("pyramid", out["pyramids"]), ("query", out["embedding_base"])):
                values = values.double()
                total, squares, count = sums.get(name, (0.0, 0.0, 0))
                sums[name] = (total + values.sum(dim=0), squares + (values * values).sum(dim=0), count + len(values))

        #per-feature mean and spread; dim 0 was the word axis
        statistics = {}
        for name, (total, squares, count) in sums.items():
            mean = total / count
            spread = (squares / count - mean * mean).clamp_min(0.0).sqrt()
            statistics[name] = (mean.float(), spread.float())
        self.model.refiner.set_statistics(*statistics["pyramid"], *statistics["query"])
        self.train(was_training)
        return int(sums["query"][2])

    @torch.no_grad()
    def strength_curve(self, loader, device, strengths):
        """Validation BAcc@k at each refinement strength, from one forward pass.

        ARGS:
            loader (DataLoader): validation batches
            device (torch.device or str)
            strengths (list): the strengths compared

        RETURNS:
            dict: strength -> BAcc@k in [0, 1]
        """
        self.eval().to(device)
        bases, deltas, targets = [], [], []
        for batch in loader:
            batch = {name: value.to(device) for name, value in batch.items()}
            out = self(batch)
            scored = batch["primary"] >= 0
            bases.append(out["embedding_base"][scored])
            deltas.append(out["delta"][scored])
            targets.append(batch["primary"][scored])
        base, delta, target = torch.cat(bases), torch.cat(deltas), torch.cat(targets)

        #z0 + strength * delta: the correction rescaled, the base unchanged
        curve = {}
        for strength in strengths:
            primary, _ = self.candidate_logits(base + float(strength) * delta)
            score = balanced_topk_accuracy(primary.cpu(), target.cpu(), self.cfg.metric_k, VOCABULARY_SIZE)
            curve[float(strength)] = float(score)
        return curve

    def refinement_losses(self, out, batch, has_target):
        """Variant M's anchor and pairing losses, over the words with a target.

        ARGS:
            out (dict): from `forward`, with the refiner's outputs
            batch (dict): from `collate_tokens`
            has_target (torch.Tensor): [N] bool

        RETURNS:
            kl : scalar - KL(base || refined) over the 50 scored words
            pair : scalar - the shuffled-feature pairing loss
        """
        refined = out["embedding"][has_target].float()
        base = out["embedding_base"][has_target].float()
        scale = self.log_scale.exp().float()

        #the anchor: kl_div(log q, log p) is KL(p || q), p the base's
        base_log = torch.log_softmax(semantic_logits(base, self.primary_table, scale), dim=1)
        refined_log = torch.log_softmax(semantic_logits(refined, self.primary_table, scale), dim=1)
        kl = F.kl_div(refined_log, base_log, log_target=True, reduction="batchmean")

        #the same prediction, another word's features: the centre cancels
        pyramids = out["pyramids"][has_target].float()
        order = torch.randperm(len(pyramids), device=pyramids.device)
        shuffled = self.model.refiner.residual(base, pyramids[order]) - out["centre"].float()

        #how much more the true residual points at the target; dim=1 is the width
        targets = F.normalize(self.target_table[batch["targets"][has_target]].float(), dim=1)
        margin = ((out["delta"][has_target].float() - shuffled) * targets).sum(dim=1)
        pair = F.softplus(-scale * margin).mean()
        return kl, pair

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

    def candidate_scores(self, out):
        """What every ranking, metric and prediction reads.

        ARGS:
            out (dict): from `forward`

        RETURNS:
            primary : [N, 50], moses : [N, 50] - with one head, exactly
            `candidate_logits`; with two, the heads' row-standardised scores
            averaged, rescaled to the heads' mean spread
        """
        primary, moses = self.candidate_logits(out["embedding"])
        if not self.cfg.has_second_head:
            return primary, moses
        primary_second, moses_second = self.candidate_logits_second(out["embedding_second"])
        return combine_heads(primary, primary_second), combine_heads(moses, moses_second)

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
        two_heads = self.cfg.has_second_head

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

            #Variant K: the same words, deduplicated afresh, in the second space
            if two_heads:
                loss_first = loss
                loss_second = siglip_loss(
                    out["embedding_second"][has_target].float(),
                    batch["targets"][has_target],
                    self.target_table_second,
                    self.log_scale_second.exp(),
                    self.logit_bias_second,
                    dedup=self.cfg.siglip_dedup,
                    shuffle=(stage == "train"),
                )
                loss = loss_first + self.cfg.second_loss_weight * loss_second

            #Variant M: stay near the base's ranking; use the word's own features
            refining = self.model.refiner is not None
            if refining:
                loss_ret = loss
                loss_kl, loss_pair = self.refinement_losses(out, batch, has_target)
                loss = loss_ret + self.cfg.refine_kl_weight * loss_kl + self.cfg.refine_pair_weight * loss_pair
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, batch_size=n_targets)
        if refining:
            self.log(f"{stage}/loss_ret", loss_ret, on_epoch=True, batch_size=n_targets)
            self.log(f"{stage}/loss_kl", loss_kl, on_epoch=True, batch_size=n_targets)
            self.log(f"{stage}/loss_pair", loss_pair, on_epoch=True, batch_size=n_targets)
        if two_heads:
            self.log(f"{stage}/loss_first", loss_first, on_epoch=True, batch_size=n_targets)
            self.log(f"{stage}/loss_second", loss_second, on_epoch=True, batch_size=n_targets)

        #retrieval scores for the epoch's metrics, on scored words only
        with torch.no_grad():
            detached = {name: value.detach() for name, value in out.items()}
            primary_logits, moses_logits = self.candidate_scores(detached)
            store = self._epoch_store[stage]
            scored = batch["primary"] >= 0
            store["primary"].append(primary_logits[scored].cpu())
            if two_heads:
                store["primary_first"].append(self.candidate_logits(detached["embedding"])[0][scored].cpu())

            #Variant M: the base's own prediction beside the refined one
            if "embedding_base" in detached:
                store["primary_base"].append(self.candidate_logits(detached["embedding_base"])[0][scored].cpu())
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

        #Variant K: the first head alone, to see what the second one adds
        if store["primary_first"] and len(targets):
            first = torch.cat(store["primary_first"])
            self.log(f"{stage}/bacc{k}_first", balanced_topk_accuracy(first, targets, k, VOCABULARY_SIZE))

        #Variant M: the frozen base alone, to see what the refiner adds
        if store["primary_base"] and len(targets):
            base = torch.cat(store["primary_base"])
            self.log(f"{stage}/bacc{k}_base", balanced_topk_accuracy(base, targets, k, VOCABULARY_SIZE))

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
        #Variant M: a frozen base hands the optimiser the refiner alone
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
        module (LitTrialEighteen): in eval mode, on `device`
        cfg (Trial18Config): the run's config, from the checkpoint
        info (dict): the checkpoint's epoch and global_step
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    #a trial 10 or 12 config is a subset of trial 13's: missing fields default
    cfg = Trial18Config(**checkpoint["hyper_parameters"])
    state = checkpoint["state_dict"]

    #placeholders of the stored shapes; load_state_dict overwrites every one
    prefix = "model.encoder.spatial."
    spatial = {"positions": None, "pca": None}
    if f"{prefix}positions" in state:
        spatial["positions"] = state[f"{prefix}positions"].numpy()
    if f"{prefix}components" in state:
        spatial["pca"] = {name: state[f"{prefix}{name}"].numpy() for name in ("mean", "components", "scale")}

    #the target widths from the stored candidates, not assumed to be t5-large's
    second_width = state["primary_table_second"].shape[1] if "primary_table_second" in state else T5_WIDTH
    module = LitTrialEighteen(
        cfg,
        table=None,
        spatial=spatial,
        embed_dim=state["primary_table"].shape[1],
        second_embed_dim=second_width,
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

    Only the BatchNorm layers run in training mode, so dropout and the
    merger's spatial dropout stay off; each layer's statistics become a plain
    average over `batches`, and its learned scale and shift are untouched.
    Labels are never read. Masked BatchNorm keeps padding out of the estimate.

    ARGS:
        module (LitTrialEighteen): on `device`
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
