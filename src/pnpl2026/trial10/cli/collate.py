"""Collate every trial-10 run into one results file, beside the paper's numbers.

    PYTHONPATH=src python -m pnpl2026.trial10.cli.collate

Reads runs/trial10_<run>_s<seed>/ for every line of configs/trial10/runs.txt
and writes two files to runs/:

    trial10_results.md    one row per seed, the paper's row on top. Paste it
    trial10_results.csv   every number, per-task scores included

A finished run reports its evaluation.json (best.ckpt scored on validation and
test, both protocols). A run still going reports its best val/bacc10 so far
from metrics.csv. Light enough for a login node: it reads JSON and CSV only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ...config import REPO_ROOT, RUNS_DIR
from .evaluate import PAPER_REFERENCE

#markdown columns: header -> (protocol, partition, metric)
COLUMNS = {
    "val ref @10": ("reference", "validation", "bacc10"),
    "TEST ref @10": ("reference", "test", "bacc10"),
    "test ref @1": ("reference", "test", "bacc1"),
    "test Moses @10": ("reference", "test", "moses_bacc10"),
    "test holdout @10": ("holdout", "test", "bacc10"),
    "sentence": ("holdout", "test", "bacc10_sentence"),
    "isolated": ("holdout", "test", "bacc10_isolated"),
    "val holdout @10": ("holdout", "validation", "bacc10"),
}


def parse_args():
    """Command-line arguments for collation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-list", default="configs/trial10/runs.txt", help="repo-relative")
    parser.add_argument("--runs-dir", default=None, help="defaults to the repo's runs/")
    return parser.parse_args()


def read_queue(runs_list):
    """The (run name, seed) pairs of the queue, in order.

    ARGS:
        runs_list (Path): configs/trial10/runs.txt

    RETURNS:
        list of (run_name, seed)
    """
    queue = []
    for line in Path(runs_list).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        config, seed = line.split()

        #the run name from the run file itself, as the PBS script builds it
        with open(REPO_ROOT / config, "r", encoding="utf-8") as handle:
            run_name = yaml.safe_load(handle)["run_name"]
        queue.append((run_name, int(seed)))
    return queue


def progress(run_dir):
    """Best val/bacc10 so far and epochs done, for a run still training.

    ARGS:
        run_dir (Path)

    RETURNS:
        dict with status, epochs, best_val
    """
    #the live file plus every copy a resume set aside
    files = sorted(run_dir.glob("metrics.*.csv")) + sorted(run_dir.glob("metrics.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
    if not frames:
        return {"status": "started, no metrics yet"}

    merged = pd.concat(frames, ignore_index=True)
    if "val/bacc10" not in merged or merged["val/bacc10"].dropna().empty:
        return {"status": "no validation yet"}

    #one row per epoch, each column's last logged value
    table = merged.groupby("epoch").last().dropna(subset=["val/bacc10"])
    return {
        "status": "training",
        "epochs": int(table.index.max()) + 1,
        "best_val": float(table["val/bacc10"].max()),
    }


def summarise(run_dir):
    """One seed's row: evaluation.json when finished, progress otherwise.

    ARGS:
        run_dir (Path)

    RETURNS:
        dict: flat columns for the CSV
    """
    if not run_dir.exists():
        return {"status": "not started"}

    path = run_dir / "evaluation.json"
    if not path.exists():
        return progress(run_dir)

    report = json.loads(path.read_text(encoding="utf-8"))
    row = {"status": "done", "best_epoch": report.get("epoch")}

    #every protocol x partition x metric, flattened; per-task scores too
    for protocol, partitions in report["results"].items():
        for partition, metrics in partitions.items():
            for name, value in metrics.items():
                if name == "by_task":
                    for task, scores in value.items():
                        row[f"{protocol}/{partition}/task/{task}"] = scores.get("bacc10")
                else:
                    row[f"{protocol}/{partition}/{name}"] = value

    #the train-val gap at the selected epoch
    selected = report.get("selected_epoch_metrics", {})
    row["train/bacc10"] = selected.get("train/bacc10")
    return row


def percent(value):
    """A fraction as a percentage cell, or a dash.

    ARGS:
        value (float or None)

    RETURNS:
        str
    """
    if value is None or pd.isna(value):
        return "-"
    return f"{100 * value:.2f}"


def main():
    """Read every run, write the markdown and the CSV."""
    args = parse_args()
    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    queue = read_queue(REPO_ROOT / args.runs_list)

    rows = []
    for run_name, seed in queue:
        row = {"run": run_name, "seed": seed}
        row.update(summarise(runs_dir / f"{run_name}_s{seed}"))
        rows.append(row)

    headers = ["run", "seed", "status", "epoch"] + list(COLUMNS) + ["train @10"]
    lines = [
        "# Trial 10 -- the d'Ascoli reference, reproduced",
        "",
        "BAcc@10 in percent; chance is 20. `ref` is the reference protocol (every word",
        "a token, full 3 s windows); `holdout` is the holdout-matched protocol. The",
        "paper's row is its Table 1, scored on the same test partition.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
        "| paper (d'Ascoli ref) | - | - | - | - | "
        f"**{percent(PAPER_REFERENCE['bacc10'])}** | {percent(PAPER_REFERENCE['bacc1'])} | "
        f"{percent(PAPER_REFERENCE['moses_bacc10'])} | - | - | - | - | - |",
    ]
    for row in rows:
        cells = [row["run"], str(row["seed"]), row["status"]]

        #a run still training shows its best validation so far in the val column
        if row["status"] != "done":
            cells.append(str(row.get("epochs", "-")))
            cells += [percent(row.get("best_val"))] + ["-"] * (len(COLUMNS) - 1) + ["-"]
        else:
            cells.append(str(row.get("best_epoch", "-")))
            cells += [percent(row.get("/".join(key))) for key in COLUMNS.values()]
            cells.append(percent(row.get("train/bacc10")))
        lines.append("| " + " | ".join(cells) + " |")

    markdown = "\n".join(lines) + "\n"
    (runs_dir / "trial10_results.md").write_text(markdown, encoding="utf-8")
    pd.DataFrame(rows).to_csv(runs_dir / "trial10_results.csv", index=False)
    print(markdown)
    print(f"[collate] wrote {runs_dir / 'trial10_results.md'} and trial10_results.csv")


if __name__ == "__main__":
    main()
