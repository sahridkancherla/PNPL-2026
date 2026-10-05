"""Does the chunked score track the leaderboard? Singles and ensembles, side by side.

    PYTHONPATH=src python -m pnpl2026.trial14.cli.calibrate \\
        --runs runs/trial14_calibration/trial10_f1_s0 ... \\
        --ensembles configs/trial14/ensembles.csv --submissions-dir submissions

Each run directory holds an `evaluation.json` and `dumps/` from `cli.evaluate
--dump`. Every run was scored on identical sequences, so the dumps line up
word for word and any group of runs can be ensembled offline, exactly as
`cli.ensemble` averages CSVs. Writes runs/trial14_calibration.md (or --out):

    1. every single run's local scores beside its leaderboard score, and each
       score's Spearman correlation with the leaderboard
    2. every submitted ENSEMBLE (`--ensembles`: name, members) rebuilt from
       its members' dumps, beside its leaderboard score; Spearman over the
       ensembles, and over singles and ensembles together
    3. how often runs agree on their top-1 word, locally and on the holdout

Light: numpy over saved arrays, no model. Safe on a login node.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from ...config import REPO_ROOT, RUNS_DIR
from ...metrics import balanced_topk_accuracy
from ..losses import blend_scores
from .ensemble import read_submission

#BAcc@10 is the scored metric; 50 words in both vocabularies
K = 10
N_WORDS = 50

#the local scores reported for single runs: (header, protocol, partition, metric)
SCORES = [
    ("chunk test", "chunked", "test", "bacc10"),
    ("chunk val", "chunked", "validation", "bacc10"),
    ("chunk isolated", "chunked", "test", "bacc10_isolated"),
    ("frag test", "fragment", "test", "bacc10"),
    ("ref test", "reference", "test", "bacc10"),
]


def parse_args():
    """Command-line arguments for calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="directories with evaluation.json and dumps/")
    parser.add_argument("--leaderboard", default="configs/trial13/leaderboard.csv", help="repo-relative")
    parser.add_argument("--ensembles", default="configs/trial14/ensembles.csv", help="repo-relative; name,members")
    parser.add_argument("--submissions-dir", default=None, help="holdout CSVs, for agreement on the holdout")
    parser.add_argument("--protocol", default="chunked", help="the dumps the ensembles are rebuilt from")
    parser.add_argument("--partition", default="test", choices=("validation", "test"))
    parser.add_argument("--out", default=None, help="default runs/trial14_calibration.md")
    return parser.parse_args()


def bacc(scores, targets, k=K):
    """BAcc@k over the 50 columns.

    ARGS:
        scores (np.ndarray): [n, 50]
        targets (np.ndarray): [n] int

    RETURNS:
        float
    """
    return balanced_topk_accuracy(torch.from_numpy(scores), torch.from_numpy(targets), k, N_WORDS)


def zscore_rows(scores):
    """Each row z-scored across its 50 words, so runs of any scale average fairly.

    ARGS:
        scores (np.ndarray): [n, 50]

    RETURNS:
        [n, 50] float64
    """
    #axis=1 is the word axis; a row's ranking is unchanged
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def markdown_table(frame):
    """A DataFrame as a markdown table, numbers to two decimals, gaps as dashes.

    ARGS:
        frame (pd.DataFrame)

    RETURNS:
        str
    """
    def cell(value):
        """One cell's text."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)

    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule] + body)


def spearman_lines(frame, columns, label):
    """One line per column: its Spearman correlation with the leaderboard.

    ARGS:
        frame (pd.DataFrame): has an "LB" column
        columns (list): the columns to correlate
        label (str): what the rows are, for the heading

    RETURNS:
        list of str
    """
    known = frame.dropna(subset=["LB"])
    lines = [f"Spearman with the leaderboard, {label}:", ""]
    for column in columns:
        pair = known[["LB", column]].dropna()
        if len(pair) < 3:
            continue

        #a column that never varies has no ranking to correlate
        if pair["LB"].nunique() < 2 or pair[column].nunique() < 2:
            lines.append(f"- {column}: undefined, a column is constant (n = {len(pair)})")
            continue
        result = spearmanr(pair["LB"], pair[column])
        lines.append(f"- {column}: {result.statistic:+.2f} (n = {len(pair)}, p = {result.pvalue:.3f})")
    return lines + [""]


def read_leaderboard(path):
    """Leaderboard scores by run or ensemble name, in-context mode only.

    ARGS:
        path (Path): the CSV: run, mode, score, date, file

    RETURNS:
        dict: name -> score in percent
    """
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    table = table[table["mode"] == "in_context"]
    return {row.run: float(row.score) for row in table.itertuples()}


def load_scores(run_dir, protocol, partition):
    """One run's scored words: its own scores (the blend, or retrieval), and layout.

    ARGS:
        run_dir (Path)
        protocol (str)
        partition (str)

    RETURNS:
        dict (scores [n, 50], targets [n], layout [n_tokens, 2]) or None
    """
    path = run_dir / "dumps" / f"{protocol}_{partition}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        scored = stored["primary"] >= 0
        retrieval = torch.from_numpy(stored["primary_logits"][scored].astype(np.float32))

        #a run without a head carries NaN there and blends nothing in
        closed = stored["closed_logits"][scored].astype(np.float32) if "closed_logits" in stored.files else None
        if closed is not None and np.isnan(closed).all():
            closed = None
        weight = float(stored["closed_weight"]) if "closed_weight" in stored.files else 0.0
        blended = blend_scores(retrieval, None if closed is None else torch.from_numpy(closed), weight)
        return {
            "scores": blended.numpy().astype(np.float64),
            "targets": stored["primary"][scored].astype(np.int64),
            "layout": np.stack([stored["sequence"], stored["primary"]], axis=1),
        }


def main():
    """Read every run, build the tables, write the markdown."""
    args = parse_args()
    run_dirs = {Path(path).name: Path(path) for path in args.runs}
    leaderboard = read_leaderboard(REPO_ROOT / args.leaderboard)
    lines = ["# Trial 14 -- calibrating local scores against the leaderboard", ""]

    # --- 1. single runs ---------------------------------------------------------------
    rows = []
    for name, run_dir in run_dirs.items():
        report = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))["results"]
        row = {"run": name, "LB": leaderboard.get(name)}
        for header, label, partition, metric in SCORES:
            value = report.get(label, {}).get(partition, {}).get(metric)
            row[header] = None if value is None else 100 * value
        rows.append(row)
    singles = pd.DataFrame(rows)
    lines += ["## 1. Single runs (BAcc@10, %)", "", markdown_table(singles), ""]
    lines += spearman_lines(singles, [header for header, *_ in SCORES], "single runs")

    # --- 2. ensembles rebuilt from their members' dumps ----------------------------------
    dumps = {name: load_scores(run_dir, args.protocol, args.partition) for name, run_dir in run_dirs.items()}
    present = [name for name in run_dirs if dumps[name] is not None]
    ensemble_rows = []
    if present:
        reference = dumps[present[0]]

        #⚠️ averaging by position is only valid if every run saw the same words
        for name in present[1:]:
            if not np.array_equal(dumps[name]["layout"], reference["layout"]):
                raise SystemExit(f"{name}'s {args.protocol} dump is laid out differently; re-run cli.evaluate.")
        targets = reference["targets"]

        groups_path = REPO_ROOT / args.ensembles
        groups = pd.read_csv(groups_path) if groups_path.exists() else pd.DataFrame(columns=["name", "members"])
        for group in groups.itertuples():
            members = [member.strip() for member in str(group.members).split(";") if member.strip()]
            found = [member for member in members if member in dumps and dumps[member] is not None]
            if not found:
                continue

            #rows z-scored per member, then averaged, as cli.ensemble does
            ensemble = np.mean([zscore_rows(dumps[member]["scores"]) for member in found], axis=0)
            ensemble_rows.append(
                {
                    "ensemble": group.name,
                    "LB": leaderboard.get(group.name),
                    "members": f"{len(found)}/{len(members)}",
                    f"{args.protocol} {args.partition}": 100 * bacc(ensemble, targets),
                    "member mean": float(np.mean([100 * bacc(dumps[member]["scores"], targets) for member in found])),
                }
            )
    if ensemble_rows:
        ensembles = pd.DataFrame(ensemble_rows)
        column = f"{args.protocol} {args.partition}"
        lines += [f"## 2. Ensembles, rebuilt from dumps ({args.protocol}, {args.partition})", "", markdown_table(ensembles), ""]
        lines += spearman_lines(ensembles, [column], "ensembles")

        #singles and ensembles on one scale: each single run as its own row
        together = pd.concat(
            [
                singles.rename(columns={"chunk test": column})[["run", "LB", column]].rename(columns={"run": "name"})
                if args.protocol == "chunked" and args.partition == "test"
                else pd.DataFrame(columns=["name", "LB", column]),
                ensembles.rename(columns={"ensemble": "name"})[["name", "LB", column]],
            ],
            ignore_index=True,
        )
        lines += spearman_lines(together, [column], "singles and ensembles together")

    # --- 3. agreement, locally and on the holdout -----------------------------------------
    lines += ["## 3. How often runs agree on their top-1 word", ""]
    local = [np.mean(dumps[a]["scores"].argmax(1) == dumps[b]["scores"].argmax(1)) for a, b in combinations(present, 2)]
    lines.append(f"- local ({args.protocol}, {args.partition}): mean {np.mean(local):.3f} over {len(local)} pairs" if local else "- local: one run only")
    if args.submissions_dir:
        from pnpl.competition import load_vocabulary

        primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
        holdout = {}
        for name in present:
            path = Path(args.submissions_dir) / f"{name}_in_context.csv"
            if path.exists():
                holdout[name] = read_submission(path, primary_words, moses_words)[1].argmax(axis=1)
        pairs = [np.mean(holdout[a] == holdout[b]) for a, b in combinations(sorted(holdout), 2)]
        if pairs:
            lines.append(f"- holdout (submitted CSVs): mean {np.mean(pairs):.3f} over {len(pairs)} pairs")
    lines.append("")

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial14_calibration.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[calibrate] wrote {out_path}")


if __name__ == "__main__":
    main()
