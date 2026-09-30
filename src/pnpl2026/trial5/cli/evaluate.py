"""Score a trial-5 checkpoint on the validation split, class by class.

    PYTHONPATH=src python -m pnpl2026.trial5.cli.evaluate \
        --ckpt runs/trial5_laptop/best.ckpt \
        --config configs/base.yaml configs/trial5.yaml

Reports each head separately as well as the blend, because the blend weight in
the config is a guess and this is where it gets replaced by a measurement.
"""

from __future__ import annotations

import argparse

import torch

from ..config import load_trial5_config
from ..data import Trial5DataModule
from ..evaluate import per_session_report, report
from ..lit import LitTrialFive
from ...metrics import balanced_topk_accuracy
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
    cfg = load_trial5_config(*args.config)

    datamodule = Trial5DataModule(cfg)
    datamodule.setup()

    model = LitTrialFive.load_from_checkpoint(
        args.ckpt,
        cfg=cfg,
        map_location="cpu",
        n_sessions=datamodule.n_sessions,
        strict=False,
    )

    #the run was SELECTED on the averaged weights, so they are the default here
    #too -- see LitTrialFive.apply_ema_weights
    if not args.no_ema and model.apply_ema_weights():
        print("[eval] using the averaged (EMA) weights")
    model.eval()

    closed, semantic, targets, sessions = [], [], [], []
    logit_scale = model.log_scale.exp()

    #⚠️ the session ids are kept, not discarded. The competition holdout is ONE
    #unseen recording, so how the score varies BETWEEN recordings matters more
    #than the pooled average over eighteen of them
    for windows, labels, batch_sessions in datamodule.val_dataloader():
        out = model(windows)
        closed.append(out["logits"])
        semantic.append(
            semantic_logits(out["embedding"], model.target_embeddings, logit_scale)
            if model.uses_semantic
            else torch.zeros_like(out["logits"])
        )
        targets.append(labels)
        sessions.extend(batch_sessions.tolist())

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

    #⚠️ the blend weight in the config is a GUESS. Sweeping it costs one pass
    #over tensors already in memory, and the run is scored on whichever wins
    if model.uses_semantic:
        print("\n=== inference_semantic_weight sweep ===")
        print(f"{'weight':>8}{'BAcc@'+str(cfg.metric_k):>12}")
        original = cfg.inference_semantic_weight
        best = (None, -1.0)
        for weight in [round(0.1 * i, 1) for i in range(11)]:
            cfg.inference_semantic_weight = weight
            score = balanced_topk_accuracy(
                model._blend(closed, semantic), targets, cfg.metric_k, cfg.n_classes
            )
            marker = "  <-- config" if abs(weight - original) < 1e-9 else ""
            print(f"{weight:>8.1f}{score:>12.4f}{marker}")
            if score > best[1]:
                best = (weight, score)
        cfg.inference_semantic_weight = original
        print(f"\n  best {best[1]:.4f} at inference_semantic_weight={best[0]} "
              f"(config has {original})")

    #the per-recording table LAST, on the blend the config actually ships
    per_session_report(
        model._blend(closed, semantic),
        targets,
        sessions,
        cfg.metric_k,
        cfg.n_classes,
        "blend",
    )


if __name__ == "__main__":
    main()
