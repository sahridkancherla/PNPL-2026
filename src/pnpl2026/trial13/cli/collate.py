"""Collate every trial-13 run into one results file, seed means first.

    PYTHONPATH=src python -m pnpl2026.trial13.cli.collate

Reads runs/<run>_s<seed>/ for every line of configs/trial13/runs.txt and
writes two files to runs/:

    trial13_results.md    per config: the seed mean (and spread), then each
                          seed. The fragment protocol leads: it is the one
                          meant to track the leaderboard. Paste it
    trial13_results.csv   every number of every seed, per-task scores included

A finished run reports its evaluation.json; a run still going reports its best
val/bacc10 so far. Light enough for a login node: JSON and CSV only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...config import REPO_ROOT, RUNS_DIR

#markdown columns: header -> (protocol label, partition, metric)
COLUMNS = {
    "FRAG test": ("fragment", "test", "bacc10"),
    "frag +bn": ("fragment+bn", "test", "bacc10"),
    "sentence": ("fragment", "test", "bacc10_sentence"),
    "isolated": ("fragment", "test", "bacc10_isolated"),
    "frag val": ("fragment", "validation", "bacc10"),
    "ref test": ("reference", "test", "bacc10"),
    "test @1": ("fragment", "test", "bacc1"),
}


def parse_args():
    """Command-line arguments for collation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-list", default="configs/trial13/runs.txt", help="repo-relative")
    parser.add_argument("--runs-dir", default=None, help="defaults to the repo's runs/")
    return parser.parse_args()


def read_queue(runs_list):
    """The (run name, seed) pairs of the queue, in order.

    ARGS:
        runs_list (Path): configs/trial13/runs.txt

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


def summarise(run_dir):
    """One seed's row: evaluation.json when finished, progress otherwise.

    ARGS:
        run_dir (Path)

    RETURNS:
        dict: flat columns, percentages
    """
    if not run_dir.exists():
        return {"status": "not started"}

    path = run_dir / "evaluation.json"
    if not path.exists():
        #the live file plus every copy a resume set aside
        files = sorted(run_dir.glob("metrics.*.csv")) + sorted(run_dir.glob("metrics.csv"))
        frames = [pd.read_csv(file) for file in files if file.stat().st_size > 0]
        merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if "val/bacc10" not in merged or merged["val/bacc10"].dropna().empty:
            return {"status": "no validation yet"}
        return {"status": "training", "frag val": 100 * float(merged["val/bacc10"].max())}

    report = json.loads(path.read_text(encoding="utf-8"))
    row = {"status": "done", "epoch": report.get("epoch")}

    #the markdown columns, then every number for the CSV
    for header, (label, partition, metric) in COLUMNS.items():
        value = report["results"].get(label, {}).get(partition, {}).get(metric)
        row[header] = None if value is None else 100 * value
    for label, partitions in report["results"].items():
        for partition, metrics in partitions.items():
            for name, value in metrics.items():
                if name != "by_task":
                    row[f"{label}/{partition}/{name}"] = value
    train = report.get("selected_epoch_metrics", {}).get("train/bacc10")
    row["train"] = None if train is None else 100 * train
    return row


def cell(value):
    """A percentage cell, or a dash.

    ARGS:
        value (float or None)

    RETURNS:
        str
    """
    if value is None or pd.isna(value):
        return "-"
    return f"{value:.2f}"


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
    table = pd.DataFrame(rows)

    headers = ["run", "seeds", "status"] + list(COLUMNS) + ["train"]
    lines = [
        "# Trial 13 -- F1 trained and judged under holdout conditions",
        "",
        "BAcc@10 in percent; chance 20. FRAG is the fragment protocol, shaped like the",
        "holdout; `+bn` re-estimates BatchNorm on the scored inputs. A config's first",
        "row is the mean over its finished seeds, +/- the spread (max - min) / 2.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for run_name in dict.fromkeys(table["run"]):
        seeds = table[table["run"] == run_name]
        done = seeds[seeds["status"] == "done"]

        #the config's mean row, from finished seeds only
        mean_cells = [f"**{run_name}**", f"{len(done)}/{len(seeds)}", "mean"]
        for header in list(COLUMNS) + ["train"]:
            values = done[header].dropna() if header in done else pd.Series(dtype=float)
            if values.empty:
                mean_cells.append("-")
            else:
                spread = (values.max() - values.min()) / 2
                mean_cells.append(f"**{values.mean():.2f}**" + (f" +/- {spread:.2f}" if len(values) > 1 else ""))
        lines.append("| " + " | ".join(mean_cells) + " |")

        #then each seed as it stands; records, since headers hold spaces
        for values in seeds.to_dict("records"):
            cells = ["", f"s{values['seed']}", values["status"]]
            cells += [cell(values.get(header)) for header in list(COLUMNS) + ["train"]]
            lines.append("| " + " | ".join(cells) + " |")

    markdown = "\n".join(lines) + "\n"
    (runs_dir / "trial13_results.md").write_text(markdown, encoding="utf-8")
    table.to_csv(runs_dir / "trial13_results.csv", index=False)
    print(markdown)
    print(f"[collate] wrote {runs_dir / 'trial13_results.md'} and trial13_results.csv")


if __name__ == "__main__":
    main()
