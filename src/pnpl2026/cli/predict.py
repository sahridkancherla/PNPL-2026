"""Holdout inference -> submission CSV (deep track).

    PYTHONPATH=src uv run python -m pnpl2026.cli.predict --ckpt runs/<name>/best.ckpt

Write this once and do not restructure it casually. LibriBrainCompetitionHoldout
owns the canonical row ordering -- the host builds the label CSV by iterating the
*same* loader, so indices align by construction. Never sort, shuffle, or reorder
the outputs of iter_windows(): probabilities are concatenated in exactly the
order yielded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

from ..config import DATA_DIR, RUNS_DIR, SUBMISSIONS_DIR, load_config
from ..lit import LitWordClassifier


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--config",
        default=None,
        help="Defaults to config.yaml beside the checkpoint.",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def available_runs() -> str:
    """Runs holding a checkpoint, for error messages."""
    if not RUNS_DIR.exists():
        return f"  (none -- {RUNS_DIR} does not exist)"
    found = sorted(d.name for d in RUNS_DIR.iterdir() if (d / "best.ckpt").exists())
    return "\n".join(f"  runs/{n}/best.ckpt" for n in found) or "  (no run has best.ckpt)"


def main() -> None:
    args = parse_args()
    ckpt = Path(args.ckpt)
    # Check the checkpoint before the config: a wrong run name otherwise
    # surfaces as a confusing complaint about a missing config.yaml.
    if not ckpt.exists():
        raise SystemExit(
            f"Checkpoint not found: {ckpt}\nAvailable:\n{available_runs()}"
        )

    cfg_path = Path(args.config) if args.config else ckpt.parent / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(
            f"Config not found: {cfg_path}\n"
            "train.py writes config.yaml beside the checkpoint; pass --config "
            "explicitly if this run predates that."
        )
    cfg = load_config(cfg_path)

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else SUBMISSIONS_DIR / f"submission_{cfg.run_name}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LitWordClassifier.load_from_checkpoint(ckpt, cfg=cfg, map_location=device)
    model.eval().to(device)

    holdout = LibriBrainCompetitionHoldout(track="deep", data_path=str(DATA_DIR))

    primary: list[np.ndarray] = []
    with torch.no_grad():
        for meg, _meta in holdout.iter_windows(batch_size=args.batch_size):
            x = torch.from_numpy(meg).float().to(device)
            probs = torch.softmax(model(x), dim=1)
            primary.append(probs.cpu().numpy())

    primary_probs = np.concatenate(primary)
    if len(primary_probs) != len(holdout.indices):
        raise RuntimeError(
            f"Row count mismatch: {len(primary_probs)} predictions vs "
            f"{len(holdout.indices)} holdout examples."
        )

    write_submission(out, indices=holdout.indices, primary_probs=primary_probs)
    print(f"wrote {out}  ({len(primary_probs)} rows)")


if __name__ == "__main__":
    main()
