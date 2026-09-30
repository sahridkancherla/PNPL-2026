"""Collate every trial-8 run into one results file.

    PYTHONPATH=src python -m pnpl2026.trial8.cli.collate

Reads runs/trial8_<run>_s<seed>/ for every line of configs/trial8/runs.txt,
picks each seed's BEST epoch on val/bacc10 -- the epoch best.ckpt holds -- and
writes two files to runs/:

    trial8_results.md    per-run means across seeds, then every seed. Paste it
    trial8_results.csv   one row per seed, every column, for your own analysis

Light enough for a login node: it reads CSVs and nothing else. Safe to rerun
while jobs are still going; a run still training shows as such.

⚠️ A resume makes Lightning wipe metrics.csv, and cli.train copies it aside as
metrics.<stamp>.csv first. Every copy is read and merged here, so a run killed
and resumed still reports its whole history.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...config import REPO_ROOT, RUNS_DIR

#the columns reported, in order; the first is what selects the best epoch
SELECT = "val/bacc10"
COLUMNS = [
    "val/bacc10",
    "val/bacc10_sentence",
    "val/bacc10_isolated",
    "val/bacc10_closed",
    "val/bacc10_semantic",
    "val/bacc1",
    "train/bacc10",
    "train/grad_ratio_semantic_to_closed",
]

#short names for the markdown table's header
SHORT = {
    "val/bacc10": "bacc10",
    "val/bacc10_sentence": "sentence",
    "val/bacc10_isolated": "isolated",
    "val/bacc10_closed": "closed",
    "val/bacc10_semantic": "semantic",
    "val/bacc1": "bacc1",
    "train/bacc10": "train",
    "train/grad_ratio_semantic_to_closed": "grad ratio",
}


def parse_args():
    """Command-line arguments for collation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-list", default="configs/trial8/runs.txt", help="repo-relative")
    parser.add_argument("--runs-dir", default=None, help="defaults to the repo's runs/")
    return parser.parse_args()


def read_queue(runs_list):
    """The (run file, seed) pairs of the queue, in order.

    ARGS:
        runs_list (Path): configs/trial8/runs.txt

    RETURNS:
        list of (str run_name, int seed, str config path)
    """
    queue = []
    for line in Path(runs_list).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        config, seed = line.split()

        #the run name comes from the run file itself, as the PBS script reads it
        with open(REPO_ROOT / config, "r", encoding="utf-8") as handle:
            run_name = yaml.safe_load(handle)["run_name"]
        queue.append((run_name, int(seed), config))
    return queue


def epoch_table(run_dir):
    """Every epoch's metrics for one run, merged across resumes.

    ARGS:
        run_dir (Path): runs/<run_name>_s<seed>

    RETURNS:
        pd.DataFrame indexed by epoch, or None if nothing is logged yet
    """
    #the live metrics.csv plus every copy a resume set aside, oldest first so
    #a later file's value for the same epoch wins
    files = sorted(run_dir.glob("metrics.*.csv")) + sorted(run_dir.glob("metrics.csv"))
    frames = [pd.read_csv(path) for path in files if path.stat().st_size > 0]
    if not frames:
        return None

    merged = pd.concat(frames, ignore_index=True)
    if "epoch" not in merged or SELECT not in merged:
        return None

    #one row per epoch: each column's last logged value. Train and val metrics
    #land on separate rows of the same epoch, and last() skips the gaps
    return merged.groupby("epoch").last()


def summarise_seed(run_dir):
    """The best epoch of one seed, and how far the run has got.

    ARGS:
        run_dir (Path): the seed's run directory

    RETURNS:
        dict of metrics at the best epoch, plus epochs, best_epoch, status
    """
    if not run_dir.exists():
        return {"status": "not started"}

    table = epoch_table(run_dir)
    if table is None or table[SELECT].dropna().empty:
        return {"status": "no validation yet"}

    scored = table.dropna(subset=[SELECT])
    best_epoch = int(scored[SELECT].idxmax())
    row = scored.loc[best_epoch]

    #max_epochs from the run's own saved config, so a finished run is known
    config_path = run_dir / "config.yaml"
    max_epochs = None
    if config_path.exists():
        max_epochs = yaml.safe_load(config_path.read_text(encoding="utf-8")).get("max_epochs")

    epochs = int(scored.index.max()) + 1
    result = {column: row.get(column, np.nan) for column in COLUMNS}

    #the gradient ratio is a train metric logged on only some epochs; take the
    #first epoch's, where the design asks for the balance to be checked
    if "train/grad_ratio_semantic_to_closed" in table:
        ratios = table["train/grad_ratio_semantic_to_closed"].dropna()
        result["train/grad_ratio_semantic_to_closed"] = ratios.iloc[0] if len(ratios) else np.nan

    result.update(
        {
            "epochs": epochs,
            "best_epoch": best_epoch,
            "status": "done" if max_epochs and epochs >= max_epochs else "stopped or running",
        }
    )
    return result


def format_cell(values):
    """Mean+/-SD over seeds, or one value, or a dash.

    ARGS:
        values (list): floats, possibly NaN

    RETURNS:
        str
    """
    values = [value for value in values if value == value]
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f}+/-{np.std(values, ddof=1):.3f}"


def write_markdown(per_seed, path):
    """The results file: summary by run, then every seed.

    ARGS:
        per_seed (pd.DataFrame): one row per (run, seed)
        path (Path): where to write
    """
    lines = ["# Trial 8 results", ""]
    lines.append("Best epoch per seed on val/bacc10; cells are mean+/-SD over seeds. Chance is 0.20.")
    lines.append("")

    header = ["run", "seeds"] + [SHORT[column] for column in COLUMNS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for run, group in per_seed.groupby("run", sort=False):
        scored = group[group[SELECT].notna()]
        cells = [run, f"{len(scored)}/{len(group)}"]
        cells += [format_cell(scored[column].tolist()) for column in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    #every geometry against the full-attention control, with the spread to beat
    lines += ["", "## Against E0: adopt a reach only if it beats E0 by more than the seed spread"]
    lines.append("")
    means = per_seed.dropna(subset=[SELECT]).groupby("run")[SELECT]
    control = "trial8_e0"
    if control not in means.groups:
        lines.append("Not yet: needs scored seeds of E0.")
    else:
        control_scores = means.get_group(control)
        for run in means.groups:
            if run == control:
                continue
            scores = means.get_group(run)
            gap = scores.mean() - control_scores.mean()
            #the larger of the two seed SDs; NaN until both have two seeds
            spreads = [group.std(ddof=1) for group in (scores, control_scores) if len(group) > 1]
            spread = max(spreads) if len(spreads) == 2 else float("nan")
            lines.append(f"- {run} - E0 = {gap:+.3f}; largest seed SD of the two = {spread:.3f}.")

    lines += ["", "## Every seed", ""]
    header = ["run", "seed", "status", "epochs", "best"] + [SHORT[column] for column in COLUMNS]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for _, row in per_seed.iterrows():
        cells = [row["run"], str(row["seed"]), row["status"]]
        cells += [
            str(int(row["epochs"])) if row.get("epochs") == row.get("epochs") else "-",
            str(int(row["best_epoch"])) if row.get("best_epoch") == row.get("best_epoch") else "-",
        ]
        cells += [format_cell([row.get(column, np.nan)]) for column in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Read every seed, write the markdown and the CSV, print the markdown."""
    args = parse_args()
    runs_dir = REPO_ROOT / args.runs_dir if args.runs_dir else RUNS_DIR

    rows = []
    for run_name, seed, config in read_queue(REPO_ROOT / args.runs_list):
        #the PBS script appends the seed to the run file's run_name
        summary = summarise_seed(runs_dir / f"{run_name}_s{seed}")
        rows.append({"run": run_name, "seed": seed, "config": config, **summary})

    #a fixed column set, so a queue with nothing run yet still writes cleanly
    per_seed = pd.DataFrame(rows).reindex(
        columns=["run", "seed", "config", "status", "epochs", "best_epoch"] + COLUMNS
    )

    markdown_path = runs_dir / "trial8_results.md"
    csv_path = runs_dir / "trial8_results.csv"
    write_markdown(per_seed, markdown_path)
    per_seed.to_csv(csv_path, index=False)

    print(markdown_path.read_text(encoding="utf-8"))
    print(f"[collate] wrote {markdown_path} and {csv_path}")


if __name__ == "__main__":
    main()
