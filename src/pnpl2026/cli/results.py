"""Summarize every run under runs/ in one table.

    PYTHONPATH=src uv run python -m pnpl2026.cli.results

CSVLogger writes multiple sparse rows per epoch -- train and val metrics land
on separate rows, most columns blank on any given row (see metrics.csv itself)
-- so "tail the log" only shows the last row, which is frequently a train-only
row with every val/* column empty. This groups by epoch and forward-fills,
matching how the tutorial notebook already reads the same file for plotting.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from ..config import RUNS_DIR


def load_run(run_dir) -> dict | None:
    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        return None

    df = pd.read_csv(metrics_csv)
    per_epoch = df.groupby("epoch").last().ffill()
    if per_epoch.empty:
        return None

    config = {}
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        import yaml

        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    git_sha = "unknown"
    prov_path = run_dir / "provenance.json"
    if prov_path.exists():
        git_sha = json.loads(prov_path.read_text(encoding="utf-8")).get("git_sha", "unknown")

    metric_col = f"val/bacc{config.get('metric_k', 10)}"
    best_row = (
        per_epoch[metric_col].idxmax() if metric_col in per_epoch.columns else None
    )
    ckpt = run_dir / "best.ckpt"

    return {
        "run": run_dir.name,
        "epochs": int(per_epoch.index.max()) + 1,
        "best_epoch": int(best_row) if best_row is not None else None,
        "best_bacc10": (
            round(float(per_epoch.loc[best_row, metric_col]), 4)
            if best_row is not None
            else None
        ),
        "last_bacc10": (
            round(float(per_epoch[metric_col].iloc[-1]), 4)
            if metric_col in per_epoch.columns
            else None
        ),
        "model": config.get("model", "?"),
        "lr": config.get("lr", "?"),
        "git_sha": git_sha[:8],
        "checkpoint": str(ckpt) if ckpt.exists() else None,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--sort-by", default="best_bacc10", help="Column to sort by (default: best_bacc10)"
    )
    args = p.parse_args()

    rows = [
        r for d in sorted(RUNS_DIR.iterdir()) if d.is_dir() and (r := load_run(d))
    ]
    if not rows:
        print(f"No runs with metrics.csv found under {RUNS_DIR}")
        return

    table = pd.DataFrame(rows).sort_values(args.sort_by, ascending=False, na_position="last")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
