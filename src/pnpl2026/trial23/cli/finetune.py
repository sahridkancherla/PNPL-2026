"""Fine-tune a per-word member into a boundary / short-sentence specialist: P6 of the 13 Oct note.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.finetune --parent runs/trial13_g2_s0/best.ckpt \\
        --run-name trial23_p6_g2_s0 [--epochs 3 --lr-scale 0.3 --short-share 0.5]

The holdout is short, separate sentences; subject 0's training set is mostly
continuous Sherlock narration. The specialist, from a trained parent:

    boundary features   each token's place and gaps (`..boundary`) into its
                        word vector before the sentence transformer, the new
                        layer zero-initialised -- the fine-tune starts as the
                        parent exactly
    short sentences     TIMIT and MOCHA-TIMIT training recordings drawn for
                        `--short-share` of every epoch's items (a weighted
                        sampler, with replacement); everything else, Sherlock
                        included, still in the mix so nothing is forgotten
    the parent's recipe its own config -- shaping, heads, EMA, loss -- with
                        the learning rate scaled by `--lr-scale`, for a fixed
                        `--epochs`; the last epoch is the model (best.ckpt)

No selection: the boundary blend weight is chosen afterwards on MOCHA
validation (`cli/boundary.py --apply`). Validation still runs every epoch,
for the log. `pbs/trial23_finetune.pbs` runs one line of
configs/trial23/specialists.txt per array task.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import replace

import lightning as L
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader, WeightedRandomSampler

from ...config import RUNS_DIR
from ..perword.data import Trial18DataModule
from ..perword.data.dataset import collate_tokens
from ..perword.embeddings import TargetTable
from ..perword.lit import LitTrialEighteen, load_trained

#the short-sentence corpora of subject 0's training partition
SHORT_TASKS = ("TIMIT", "MOCHATIMIT")


def parse_args():
    """Command-line arguments for the specialist fine-tune."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parent", required=True, help="the parent's best.ckpt (per-word family)")
    parser.add_argument("--run-name", required=True, help="runs/<run-name>/ receives best.ckpt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr-scale", type=float, default=0.3, help="the parent's learning rate times this")
    parser.add_argument("--short-share", type=float, default=0.5, help="share of each epoch's items from TIMIT and MOCHA")
    parser.add_argument("--cache-dir", default=None, help="override the run's cache_dir (a staged copy)")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="a smoke run's cap; -1 trains every epoch")
    parser.add_argument("--force", action="store_true", help="overwrite an existing run directory")
    return parser.parse_args()


class ShortSentenceDataModule(Trial18DataModule):
    """The parent's data, with TIMIT and MOCHA drawn for a fixed share of every epoch."""

    def __init__(self, cfg, table, short_share):
        """
        ARGS:
            cfg (Trial18Config), table (TargetTable)
            short_share (float): in (0, 1)
        """
        super().__init__(cfg, table)
        self.short_share = short_share

    def sequence_weights(self):
        """Each training item's draw weight: the short corpora share `short_share` between them.

        RETURNS:
            [n_sequences] float64, and the share of items that are short before weighting
        """
        tasks = [self.train_set.recordings.records[sequence.recording]["task"] for sequence in self.train_set.sequences]
        short = np.array([task in SHORT_TASKS for task in tasks])
        if not short.any() or short.all():
            raise SystemExit(f"No mix to weight: {int(short.sum())} of {len(short)} items are TIMIT/MOCHA.")
        weights = np.where(short, self.short_share / short.sum(), (1.0 - self.short_share) / (~short).sum())
        return weights, float(short.mean())

    def train_dataloader(self):
        """The training items drawn by weight, with replacement, as many per epoch as the set holds."""
        weights, natural = self.sequence_weights()
        sampler = WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(weights), replacement=True)
        print(f"[finetune] TIMIT/MOCHA: {100 * natural:.1f}% of items, drawn for {100 * self.short_share:.0f}% of each epoch")
        extra = {"persistent_workers": True, "prefetch_factor": 4} if self.cfg.num_workers > 0 else {}
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, sampler=sampler, num_workers=self.cfg.num_workers,
                          pin_memory=torch.cuda.is_available(), collate_fn=collate_tokens, drop_last=True, **extra)


def specialist_config(parent_cfg, args):
    """The parent's config with the specialist's changes.

    ARGS:
        parent_cfg (Trial18Config)
        args (argparse.Namespace)

    RETURNS:
        Trial18Config
    """
    changes = {
        "run_name": args.run_name,
        "lr": parent_cfg.lr * args.lr_scale,
        "max_epochs": args.epochs,
        "seed": args.seed,
        "model_kwargs": dict(parent_cfg.model_kwargs, boundary_features=True),
    }
    if args.cache_dir:
        changes["cache_dir"] = args.cache_dir
    if args.num_workers is not None:
        changes["num_workers"] = args.num_workers
    return replace(parent_cfg, **changes)


def main():
    """Load the parent, build the specialist on its weights, fine-tune, save the last epoch."""
    args = parse_args()
    parent, parent_cfg, info = load_trained(args.parent, device="cpu", use_ema=True)
    cfg = specialist_config(parent_cfg, args)
    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    run_dir = RUNS_DIR / cfg.run_name
    if list(run_dir.glob("*.ckpt")) and not args.force:
        raise SystemExit(f"{run_dir} already holds checkpoints; pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False), encoding="utf-8")

    #the training targets, as the parent trained on them
    table = TargetTable(cfg.resolved_word_embeddings(), centre=cfg.centre_embeddings)
    second_table = TargetTable(cfg.resolved_second_embeddings(), centre=cfg.centre_embeddings) if cfg.has_second_head else None
    model = LitTrialEighteen(cfg, table, second_table=second_table)

    #the parent's (averaged) weights in; only the new boundary layer is missing
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    stray = [name for name in missing if not name.startswith("model.boundary.")]
    if unexpected or stray:
        raise SystemExit(f"The parent does not fit the specialist: unexpected {unexpected[:5]}, missing {stray[:5]}")
    del parent
    print(f"[finetune] {args.parent} (epoch {info['epoch']}) -> {cfg.run_name}: lr {cfg.lr:g}, {cfg.max_epochs} epochs, "
          f"boundary features on, TIMIT/MOCHA at {100 * args.short_share:.0f}% of items")

    datamodule = ShortSentenceDataModule(cfg, table, args.short_share)
    datamodule.setup()
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        max_steps=args.max_steps,
        accelerator=cfg.accelerator,
        devices=1,
        precision=cfg.precision,
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="", version=""),
        callbacks=[ModelCheckpoint(dirpath=run_dir, filename="last", monitor=None, save_top_k=1),
                   TQDMProgressBar(refresh_rate=1 if sys.stdout.isatty() else 200)],
    )
    trainer.fit(model, datamodule=datamodule)

    #no selection: the last epoch is the specialist
    shutil.copy2(run_dir / "last.ckpt", run_dir / "best.ckpt")
    (run_dir / "finetune.json").write_text(json.dumps({"parent": args.parent, "epochs": args.epochs, "lr_scale": args.lr_scale,
                                                        "short_share": args.short_share, "seed": args.seed}, indent=2), encoding="utf-8")
    print(f"[finetune] wrote {run_dir / 'best.ckpt'}")


if __name__ == "__main__":
    main()
