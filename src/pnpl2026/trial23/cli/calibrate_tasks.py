"""Calibrate local scores against the leaderboard, corpus by corpus.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.calibrate_tasks [--protocols reference fragment]

The second architecture note (10 Oct) argues that subject 0's non-Sherlock
material -- MOCHA-TIMIT and TheMoth, both in pnpl's test partition -- is a
better local stand-in for the holdout than Sherlock. This answers that with
the board points we already have, training nothing:

    1. every leaderboard row whose members have local dumps: its ensemble
       rebuilt from the members' TEST dumps -- each member's 50 scores
       z-scored per token, averaged, as `cli.ensemble` averages CSVs
    2. a row whose file was hub-corrected on the holdout (mode "hubh") gets
       the same correction locally: each word's column z-scored over the
       subset's own scored tokens
    3. BAcc@10 on scored sentence tokens: all of them, and each corpus apart
       (the dumps carry `task`)
    4. rows with the same members and correction are ONE point, their board
       scores averaged: the scaler, onset-shift and lone-row variants of a
       file differ on the board by a row or two, and not at all locally
    5. per protocol and subset: Spearman with the board, and the share of
       pairs ordered as the board orders them (the note's bar: 80%)

Dumps are looked for in runs/<run>/dumps/, then in
runs/trial14_calibration/<run>/dumps/ (trial 14's re-scoring of trials 10-13
under the reference and fragment protocols). A row with any member missing,
or whose members' dumps list different tokens, is skipped and named.

No label of the holdout is seen. Writes runs/trial23_calibration.md. Light:
saved arrays only, a minute or two on a login node.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ...config import REPO_ROOT, RUNS_DIR
from ...metrics import balanced_topk_accuracy
from ..hubnorm import apply_correction, fit_zscore

#the dump key holding each token's 50 primary scores; trial 13 wrote "retrieval"
SCORE_KEYS = ("primary_logits", "retrieval")

#BAcc@10 over the 50 scored words
K = 10
N_WORDS = 50

#the note's bar for a proxy: this share of pairs ordered as the board orders them
PAIR_BAR = 0.80

#where trial 14 re-scored trials 10-13, beside each run's own dumps
CALIBRATION_DIR = "trial14_calibration"


def parse_args():
    """Command-line arguments for the calibration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocols", nargs="+", default=["reference", "fragment"], help="dump protocols to score")
    parser.add_argument("--partition", default="test", help="dump partition; test is the one never selected on")
    parser.add_argument("--leaderboard", default="configs/trial13/leaderboard.csv", help="repo-relative")
    parser.add_argument("--ensembles", default="configs/trial14/ensembles.csv", help="repo-relative")
    parser.add_argument("--runs-dir", default=str(RUNS_DIR), help="where run directories live")
    parser.add_argument("--min-points", type=int, default=5, help="fewer covered points than this: no statistics")
    parser.add_argument("--out", default=None, help="default runs/trial23_calibration.md")
    return parser.parse_args()


def zscore_rows(scores):
    """Each row z-scored across its words; a flat row stays flat.

    ARGS:
        scores (np.ndarray): [n, n_words]

    RETURNS:
        [n, n_words] float64
    """
    #axis=1 is the word axis: each token on its own
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def find_dump(run, protocol, partition, runs_dir):
    """A run's dump for one protocol and partition, or None.

    ARGS:
        run (str): run name
        protocol (str), partition (str)
        runs_dir (Path)

    RETURNS:
        Path or None
    """
    name = f"{protocol}_{partition}.npz"
    for path in (runs_dir / run / "dumps" / name, runs_dir / CALIBRATION_DIR / run / "dumps" / name):
        if path.exists():
            return path
    return None


def read_dump(path):
    """One dump's scores, labels, isolated flags and corpus labels.

    ARGS:
        path (Path)

    RETURNS:
        dict: scores [n, 50] float64, primary [n] int64, isolated [n] bool,
        task [n] str or None
    """
    with np.load(path, allow_pickle=True) as stored:
        key = next((name for name in SCORE_KEYS if name in stored.files), None)
        if key is None:
            raise ValueError(f"{path}: none of {SCORE_KEYS} in {stored.files}")
        return {
            "scores": stored[key].astype(np.float64),
            "primary": stored["primary"].astype(np.int64),
            "isolated": stored["isolated"].astype(bool) if "isolated" in stored.files else None,
            "task": stored["task"].astype(str) if "task" in stored.files else None,
        }


def board_points(leaderboard, ensembles):
    """The leaderboard's rows as distinct points: members, correction, mean score.

    ARGS:
        leaderboard (pd.DataFrame): run, mode, score, ...
        ensembles (dict): ensemble name -> [member runs]

    RETURNS:
        list of dict: label, members (tuple), hub (str: "", "hubh" or
        "frozen"), board (float, mean), n_rows (int), files (list)
    """
    points = {}
    for row in leaderboard.itertuples(index=False):
        members = tuple(sorted(ensembles.get(row.run, [row.run])))
        mode = str(row.mode)

        #"hubh" is the holdout-fitted correction; a frozen "_hub" cannot be redone locally
        hub = "hubh" if "hubh" in mode else ("frozen" if "hub" in mode else "")
        key = (members, hub)
        point = points.setdefault(key, {"label": f"{row.run} ({mode})", "members": members, "hub": hub, "scores": [], "files": []})
        point["scores"].append(float(row.score))
        point["files"].append(str(row.file))
    for point in points.values():
        point["board"] = float(np.mean(point["scores"]))
        point["n_rows"] = len(point["scores"])
    return list(points.values())


def point_scores(point, protocol, partition, runs_dir, cache):
    """One board point's local BAcc@10, all scored sentence tokens and per corpus.

    ARGS:
        point (dict): from `board_points`
        protocol (str), partition (str)
        runs_dir (Path)
        cache (dict): (run, protocol, partition) -> dump, filled as read

    RETURNS:
        (scores, reason): scores is a dict subset -> BAcc@10 in percent, or
        None with the reason it could not be computed
    """
    if point["hub"] == "frozen":
        return None, "frozen hub correction: not reproducible from dumps"

    #every member's dump, read once across all points
    dumps = []
    for run in point["members"]:
        key = (run, protocol, partition)
        if key not in cache:
            path = find_dump(run, protocol, partition, runs_dir)
            cache[key] = read_dump(path) if path is not None else None
        if cache[key] is None:
            return None, f"no {protocol} {partition} dump for {run}"
        dumps.append(cache[key])

    #⚠️ the same tokens in the same order, or the average mixes words
    first = dumps[0]
    for run, dump in zip(point["members"], dumps):
        if not np.array_equal(dump["primary"], first["primary"]):
            return None, f"{run}'s dump lists other tokens than {point['members'][0]}'s"

    #the members' corpus labels and isolated flags: any member that has them
    task = next((dump["task"] for dump in dumps if dump["task"] is not None), None)
    isolated = next((dump["isolated"] for dump in dumps if dump["isolated"] is not None), np.zeros(len(first["primary"]), bool))
    ensemble = np.mean([zscore_rows(dump["scores"]) for dump in dumps], axis=0)

    #scored sentence tokens: the board scores no lone rows
    base = (first["primary"] >= 0) & ~isolated
    subsets = {"all": base}
    if task is not None:
        for name in sorted(set(task[base])):
            subsets[name] = base & (task == name)

    results = {}
    for name, mask in subsets.items():
        scores = ensemble[mask]

        #the holdout-fitted correction, fitted here on the subset's own tokens
        if point["hub"] == "hubh":
            scores = apply_correction(scores, fit_zscore(scores))
        targets = torch.from_numpy(first["primary"][mask])
        results[name] = 100.0 * float(balanced_topk_accuracy(torch.from_numpy(scores), targets, K, N_WORDS))
    return results, None


def agreement(local, board):
    """Spearman, and the share of pairs ordered as the board orders them.

    ARGS:
        local (list), board (list): one value per point, the same points

    RETURNS:
        (spearman, pair_share, n_pairs)
    """
    from scipy.stats import spearmanr

    #pairs the board ties carry no order to agree with
    agree = total = 0
    for first, second in combinations(range(len(local)), 2):
        board_gap = board[first] - board[second]
        local_gap = local[first] - local[second]
        if board_gap == 0 or local_gap == 0:
            continue
        total += 1
        agree += (board_gap > 0) == (local_gap > 0)
    rho = float(spearmanr(local, board).correlation) if len(local) > 2 else float("nan")
    return rho, (agree / total if total else float("nan")), total


def main():
    """Rebuild every scorable point locally, compare with the board, report."""
    args = parse_args()
    leaderboard = pd.read_csv(REPO_ROOT / args.leaderboard)
    table = pd.read_csv(REPO_ROOT / args.ensembles)
    ensembles = {str(name): [run for run in str(members).split(";") if run] for name, members in zip(table["name"], table["members"])}
    points = board_points(leaderboard, ensembles)
    runs_dir = Path(args.runs_dir)
    cache = {}

    lines = [
        "# Trial 20 -- do local scores order the leaderboard, corpus by corpus?",
        "",
        f"{len(leaderboard)} leaderboard rows -> {len(points)} distinct points (same members and correction).",
        f"Local: the members' `{args.partition}` dumps, z-scored and averaged; BAcc@10 on scored sentence tokens;",
        "\"hubh\" points get each word's column z-scored on the subset's own tokens, as on the holdout.",
        f"'pairs' is the share of point pairs ordered as the board orders them; the note's bar is {100 * PAIR_BAR:.0f}%.",
        "",
        "| protocol | subset | points | Spearman | pairs ordered as the board | pairs |",
        "|---|---|---|---|---|---|",
    ]
    detail = []
    for protocol in args.protocols:
        scored, skipped = [], []
        for point in points:
            results, reason = point_scores(point, protocol, args.partition, runs_dir, cache)
            if results is None:
                skipped.append((point["label"], reason))
            else:
                scored.append((point, results))

        #every subset any point reached, "all" first
        subsets = ["all"] + sorted({name for _, results in scored for name in results if name != "all"})
        for subset in subsets:
            pairs = [(results[subset], point["board"]) for point, results in scored if subset in results]
            if len(pairs) < args.min_points:
                lines.append(f"| {protocol} | {subset} | {len(pairs)} | - | - | - |")
                continue
            rho, share, n_pairs = agreement([local for local, _ in pairs], [board for _, board in pairs])
            mark = " **passes**" if share >= PAIR_BAR else ""
            lines.append(f"| {protocol} | {subset} | {len(pairs)} | {rho:+.2f} | {100 * share:.0f}%{mark} | {n_pairs} |")

        #the points themselves, best on the board first
        detail += ["", f"## {protocol}: every point", "", "| point | members | board | " + " | ".join(subsets) + " |",
                   "|---|---|---|" + "---|" * len(subsets)]
        for point, results in sorted(scored, key=lambda item: -item[0]["board"]):
            cells = " | ".join(f"{results[name]:.1f}" if name in results else "-" for name in subsets)
            detail.append(f"| {point['label']} | {len(point['members'])} | {point['board']:.2f} | {cells} |")
        if skipped:
            detail += ["", f"Skipped under {protocol} ({len(skipped)}):", ""]
            detail += [f"- {label}: {reason}" for label, reason in skipped]

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial23_calibration.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines + detail) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[calibrate] wrote {out_path}")


if __name__ == "__main__":
    main()
