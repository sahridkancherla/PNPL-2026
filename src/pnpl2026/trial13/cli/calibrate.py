"""Which local score tracks the leaderboard, and what ensembles and switches are worth.

    PYTHONPATH=src python -m pnpl2026.trial13.cli.calibrate \\
        --runs runs/trial13_calibration/trial10_f1_s0 runs/trial13_calibration/trial10_f0_s0 ... \\
        --leaderboard configs/trial13/leaderboard.csv --submissions-dir submissions

Each run directory holds an `evaluation.json` and `dumps/` from `cli.evaluate
--dump --bn-adapt`. Every run was scored on identical sequences, so their
dumps line up word for word and can be averaged offline. Writes
runs/trial13_calibration.md (or --out) with four tables:

    1. every local score beside the leaderboard, and each score's Spearman
       correlation with it over the runs that have one
    2. the local ensemble of all runs, against its members
    3. column normalisation (each word z-scored across words of one source)
       and BatchNorm re-estimation, run by run and for the ensemble
    4. how often runs agree on their top-1 word, locally and on the holdout --
       if the holdout's agreement is far lower, the holdout is further from
       the training data than the local test partition is

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
from .ensemble import normalise_columns, read_submission

#BAcc@10 is the scored metric; 50 words in both vocabularies
K = 10
N_WORDS = 50

#the local scores reported, as (column header, protocol label, partition, metric)
SCORES = [
    ("ref test", "reference", "test", "bacc10"),
    ("frag test", "fragment", "test", "bacc10"),
    ("frag test +bn", "fragment+bn", "test", "bacc10"),
    ("frag sentence", "fragment", "test", "bacc10_sentence"),
    ("frag isolated", "fragment", "test", "bacc10_isolated"),
    ("frag val", "fragment", "validation", "bacc10"),
    ("sparse test", "sparse", "test", "bacc10"),
]


def parse_args():
    """Command-line arguments for calibration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="directories with evaluation.json and dumps/")
    parser.add_argument("--leaderboard", default="configs/trial13/leaderboard.csv", help="repo-relative")
    parser.add_argument("--submissions-dir", default=None, help="holdout CSVs, for agreement on the holdout")
    parser.add_argument("--protocol", default="fragment", help="the dumps the ensemble tables use")
    parser.add_argument("--partition", default="test", choices=("validation", "test"))
    parser.add_argument("--out", default=None, help="default runs/trial13_calibration.md")
    return parser.parse_args()


def bacc(logits, targets, k=K):
    """BAcc@k over the 50 columns.

    ARGS:
        logits (np.ndarray): [n, 50]
        targets (np.ndarray): [n] int

    RETURNS:
        float
    """
    return balanced_topk_accuracy(torch.from_numpy(logits), torch.from_numpy(targets), k, N_WORDS)


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


def zscore_rows(scores):
    """Each row z-scored across its 50 words, so runs of any temperature average fairly.

    ARGS:
        scores (np.ndarray): [n, 50] logits

    RETURNS:
        [n, 50] float64
    """
    #axis=1 is the word axis; a row's ranking is unchanged
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def read_leaderboard(path):
    """Leaderboard scores by run name, sentence mode only.

    ARGS:
        path (Path): the CSV: run, mode, score, date, file

    RETURNS:
        dict: run name -> score in percent
    """
    if not path.exists():
        return {}
    table = pd.read_csv(path)

    #in-context (sentence) mode is what every run here is compared under
    table = table[table["mode"] == "in_context"]
    return {row.run: float(row.score) for row in table.itertuples()}


def load_dump(run_dir, label, partition):
    """One run's per-token dump, scored tokens only.

    ARGS:
        run_dir (Path)
        label (str): e.g. "fragment" or "fragment+bn"
        partition (str)

    RETURNS:
        dict of arrays (logits, targets, isolated, tokens), or None if absent
    """
    path = run_dir / "dumps" / f"{label}_{partition}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        scored = stored["primary"] >= 0
        return {
            "logits": stored["primary_logits"][scored].astype(np.float64),
            "targets": stored["primary"][scored].astype(np.int64),
            "isolated": stored["isolated"][scored].astype(bool),
            #the token layout, to prove every run saw the same words in order
            "layout": np.stack([stored["sequence"], stored["primary"]], axis=1),
        }


def main():
    """Read every run, build the four tables, write the markdown."""
    args = parse_args()
    run_dirs = [Path(path) for path in args.runs]
    names = [path.name for path in run_dirs]
    leaderboard = read_leaderboard(REPO_ROOT / args.leaderboard)
    lines = ["# Trial 13 -- calibrating local scores against the leaderboard", ""]

    # --- 1. every local score beside the leaderboard -----------------------------
    rows = []
    for name, run_dir in zip(names, run_dirs):
        report = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))["results"]
        row = {"run": name, "LB": leaderboard.get(name)}
        for header, label, partition, metric in SCORES:
            value = report.get(label, {}).get(partition, {}).get(metric)
            row[header] = None if value is None else 100 * value
        rows.append(row)
    table = pd.DataFrame(rows)
    lines += ["## 1. Local scores beside the leaderboard (BAcc@10, %)", "", markdown_table(table), ""]

    #Spearman over the runs with a leaderboard score
    known = table.dropna(subset=["LB"])
    lines.append(f"Spearman with the leaderboard over {len(known)} runs:")
    lines.append("")
    for header, *_ in SCORES:
        pair = known[["LB", header]].dropna()
        if len(pair) < 3:
            continue

        #a column that never varies has no ranking to correlate
        if pair["LB"].nunique() < 2 or pair[header].nunique() < 2:
            lines.append(f"- {header}: undefined, a column is constant (n = {len(pair)})")
            continue
        rho = spearmanr(pair["LB"], pair[header]).statistic
        lines.append(f"- {header}: {rho:+.2f} (n = {len(pair)})")
    lines.append("")

    # --- 2 and 3. ensemble, column normalisation, BatchNorm re-estimation ----------
    dumps = {name: load_dump(run_dir, args.protocol, args.partition) for name, run_dir in zip(names, run_dirs)}
    adapted = {name: load_dump(run_dir, args.protocol + "+bn", args.partition) for name, run_dir in zip(names, run_dirs)}
    present = [name for name in names if dumps[name] is not None]
    if present:
        reference = dumps[present[0]]

        #⚠️ averaging by position is only valid if every run saw the same words
        for name in present[1:]:
            if not np.array_equal(dumps[name]["layout"], reference["layout"]):
                raise SystemExit(f"{name}'s {args.protocol} dump is laid out differently; re-run cli.evaluate.")
        targets, groups = reference["targets"], reference["isolated"].astype(int)

        effect = []
        for name in present:
            plain = dumps[name]["logits"]
            entry = {
                "run": name,
                "plain": 100 * bacc(plain, targets),
                "columns normalised": 100 * bacc(normalise_columns(zscore_rows(plain), groups), targets),
            }
            if adapted[name] is not None:
                entry["BatchNorm re-estimated"] = 100 * bacc(adapted[name]["logits"], targets)
            effect.append(entry)

        #the ensemble: rows z-scored per run, then averaged, as cli.ensemble does
        ensemble = np.mean([zscore_rows(dumps[name]["logits"]) for name in present], axis=0)
        entry = {
            "run": f"ENSEMBLE of {len(present)}",
            "plain": 100 * bacc(ensemble, targets),
            "columns normalised": 100 * bacc(normalise_columns(ensemble, groups), targets),
        }
        if all(adapted[name] is not None for name in present):
            adapted_ensemble = np.mean([zscore_rows(adapted[name]["logits"]) for name in present], axis=0)
            entry["BatchNorm re-estimated"] = 100 * bacc(adapted_ensemble, targets)
        effect.append(entry)

        singles = [item["plain"] for item in effect[:-1]]
        lines += [
            f"## 2-3. Ensemble and switches, {args.protocol} protocol, {args.partition} partition (BAcc@10, %)",
            "",
            markdown_table(pd.DataFrame(effect)),
            "",
            f"Members: mean {np.mean(singles):.2f}, best {np.max(singles):.2f}; the ensemble "
            f"{effect[-1]['plain'] - np.max(singles):+.2f} over the best member.",
            "",
        ]

        # --- 4. agreement, locally and on the holdout --------------------------------
        local = [np.mean(dumps[a]["logits"].argmax(1) == dumps[b]["logits"].argmax(1)) for a, b in combinations(present, 2)]
        lines += ["## 4. How often runs agree on their top-1 word", ""]
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

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial13_calibration.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[calibrate] wrote {out_path}")


if __name__ == "__main__":
    main()
