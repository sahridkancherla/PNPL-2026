"""Score a trained trial-1 checkpoint on the validation split.

⚠️ WRITTEN BY AI -- 2026-09-24. UNVERIFIED.

    PYTHONPATH=src python -m pnpl2026.trial1.cli.evaluate \
        --config configs/base.yaml configs/trial1.yaml \
        --ckpt runs/trial1/best.ckpt

Reports BAcc@10 against its 0.20 floor, BAcc@1 as a tiebreaker, and the
per-class recall table -- which is where a model riding the class prior
becomes visible and a single scalar does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..config import load_trial1_config
from ..data import Trial1DataModule
from ..evaluate import build_report, collect_predictions, print_report, write_report
from ..lit import LitTrialOne


def parse_args():
    """Command-line arguments for an evaluation pass."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--ckpt", required=True, help="Path to a .ckpt file.")
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the JSON report. Defaults to beside the checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main():
    """Load the checkpoint, run the validation split, print and save the report."""
    args = parse_args()
    cfg = load_trial1_config(*args.config)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"No checkpoint at {ckpt_path}")

    #the config must be the one the run trained under, or the model is rebuilt
    #with a different width and the state dict will not load
    model = LitTrialOne.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(args.device)

    datamodule = Trial1DataModule(cfg)
    datamodule.setup()

    #⚠️ val_dataloader is never shuffled, which is what makes the per-class
    #breakdown line up with the labels it is compared against
    logits, targets = collect_predictions(
        model, datamodule.val_dataloader(), device=args.device
    )

    report = build_report(logits, targets, cfg)
    print_report(report, cfg)

    out = Path(args.out) if args.out else ckpt_path.parent / "eval_report.json"
    write_report(report, out)


if __name__ == "__main__":
    main()
