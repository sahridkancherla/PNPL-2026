"""Score a trial-4 checkpoint on the validation split, class by class.

    PYTHONPATH=src python -m pnpl2026.trial4.cli.evaluate \
        --ckpt runs/trial4_laptop/best.ckpt \
        --config configs/base.yaml configs/trial4.yaml

Reports each head separately as well as the blend, because the blend weight in
the config is a guess and this is where it gets replaced by a measurement.
"""

from __future__ import annotations

import argparse

import torch

from ..config import load_trial4_config
from ..data import Trial4DataModule
from ..evaluate import report
from ..lit import LitTrialFour
from ..losses import semantic_logits


def parse_args():
    """Command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="use the live weights rather than the averaged ones",
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    """Run the validation split through a checkpoint and report per class."""
    args = parse_args()
    cfg = load_trial4_config(*args.config)

    datamodule = Trial4DataModule(cfg)
    datamodule.setup()

    model = LitTrialFour.load_from_checkpoint(
        args.ckpt,
        cfg=cfg,
        map_location="cpu",
        n_sessions=datamodule.n_sessions,
        strict=False,
    )

    #the run was SELECTED on the averaged weights, so they are the default here
    #too -- see LitTrialFour.apply_ema_weights
    if not args.no_ema and model.apply_ema_weights():
        print("[eval] using the averaged (EMA) weights")
    model.eval()

    closed, semantic, targets = [], [], []
    logit_scale = model.log_scale.exp()

    #the val loader is never shuffled, so predictions and labels stay aligned
    for windows, labels, _sessions in datamodule.val_dataloader():
        out = model(windows)
        closed.append(out["logits"])
        semantic.append(
            semantic_logits(out["embedding"], model.target_embeddings, logit_scale)
            if model.uses_semantic
            else torch.zeros_like(out["logits"])
        )
        targets.append(labels)

    closed = torch.cat(closed)
    semantic = torch.cat(semantic)
    targets = torch.cat(targets)

    #each head on its own first, so the blend can be judged against them
    report(closed, targets, cfg.metric_k, cfg.n_classes, "closed-set head")
    if model.uses_semantic:
        report(semantic, targets, cfg.metric_k, cfg.n_classes, "semantic head")
    report(
        model._blend(closed, semantic),
        targets,
        cfg.metric_k,
        cfg.n_classes,
        f"blend at inference_semantic_weight={cfg.inference_semantic_weight}",
    )


if __name__ == "__main__":
    main()
