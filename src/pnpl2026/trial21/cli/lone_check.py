"""Which recipe should fill the holdout's lone-word rows? A local comparison.

    PYTHONPATH=src python -m pnpl2026.trial21.cli.lone_check \\
        --group mix=runs/trial15_alone_check/trial13_g1_s0,runs/trial15_alone_check/trial13_g2_s0,... \\
        --group i3=runs/trial15_i3_s0,runs/trial15_i3_s1,runs/trial15_i3_s2 \\
        --group l=runs/trial21_l_s0,runs/trial21_l_s1,runs/trial21_l_s2

The 92 lone words are scored on the PRIVATE board only, so no submission can
judge them; the isolated protocol serves every validation and test word in
exactly their format (alone, 1 s from onset), on thousands of words. Every
run's `dumps/isolated_{validation,test}.npz` (from `evaluate --dump` or a
training run's end-of-run scoring) is read; then

    - each group is ensembled as cli.ensemble averages CSVs (rows z-scored,
      averaged, z-scored again)
    - every combination of groups -- each alone, and every equal-weight mix,
      as `cli.ensemble --lone-inputs` mixes them (each group re-standardised,
      then averaged) -- is scored with and without the frozen hub correction
      (`hubnorm.fit_zscore` on the combination's validation scores, labels
      never used)
    - the winner is chosen on VALIDATION BAcc@10 by the rule fixed in
      advance: the highest; and its TEST score is the unbiased number

Writes runs/trial21_lone_check.md. Light: numpy over saved arrays, seconds.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from ...config import RUNS_DIR
from ...metrics import balanced_topk_accuracy
from ..hubnorm import K, apply_correction, fit_zscore, top_k_occupancy

#the dump key holding each token's 50 primary scores; trial 13 wrote "retrieval"
SCORE_KEYS = ("primary_logits", "retrieval")

#the 50 scored words
N_WORDS = 50


def parse_args():
    """Command-line arguments for the lone-word comparison."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", required=True,
                        help="NAME=run_dir,run_dir,... ; give it once per group")
    parser.add_argument("--protocol", default="isolated", help="the dumps compared")
    parser.add_argument("--out", default=None, help="default runs/trial21_lone_check.md")
    return parser.parse_args()


def zscore_rows(scores):
    """Each row z-scored across its words; a flat row stays flat.

    ARGS:
        scores (np.ndarray): [n, n_words]

    RETURNS:
        [n, n_words] float64
    """
    #axis=1 is the word axis: each row on its own
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def read_scored(run_dir, protocol, partition):
    """One run's scored rows: scores and true columns.

    ARGS:
        run_dir (str): holds dumps/
        protocol (str)
        partition (str)

    RETURNS:
        scores [n_scored, 50], targets [n_scored]
    """
    path = Path(run_dir) / "dumps" / f"{protocol}_{partition}.npz"
    with np.load(path) as stored:
        key = next((name for name in SCORE_KEYS if name in stored.files), None)
        if key is None:
            raise SystemExit(f"{path}: none of {SCORE_KEYS} in {stored.files}")
        primary = stored["primary"].astype(np.int64)
        scored = primary >= 0
        return stored[key].astype(np.float64)[scored], primary[scored]


def group_scores(runs, protocol, partition, reference_targets=None):
    """One group's ensemble on one partition, as cli.ensemble builds a CSV.

    ARGS:
        runs (list): run directories
        protocol (str)
        partition (str)
        reference_targets (np.ndarray or None): every group must score the
            same words in the same order

    RETURNS:
        scores [n_scored, 50] standardised, targets [n_scored]
    """
    loaded = [read_scored(run, protocol, partition) for run in runs]
    targets = loaded[0][1] if reference_targets is None else reference_targets

    #⚠️ the same words in the same order, or the average mixes words
    for run, (_, these) in zip(runs, loaded):
        if not np.array_equal(these, targets):
            raise SystemExit(f"{run}: its {protocol} {partition} words differ from the first run's.")

    #axis 0 is the run axis; z-scored again, as a CSV is read
    return zscore_rows(np.mean([zscore_rows(scores) for scores, _ in loaded], axis=0)), targets


def bacc(scores, targets):
    """BAcc@10 over the 50 words, in percent.

    ARGS:
        scores (np.ndarray): [n, 50]
        targets (np.ndarray): [n]

    RETURNS:
        float
    """
    return 100.0 * balanced_topk_accuracy(torch.from_numpy(scores), torch.from_numpy(targets), K, N_WORDS)


def main():
    """Ensemble each group, score every combination, choose on validation."""
    args = parse_args()

    #NAME=dir,dir,... into an ordered mapping
    groups = {}
    for spec in args.group:
        name, _, paths = spec.partition("=")
        if not name or not paths:
            raise SystemExit(f"--group {spec!r} is not NAME=run_dir,run_dir,...")
        groups[name] = [path for path in paths.split(",") if path]

    #every group on both partitions, the words checked identical across groups
    data = {}
    for partition in ("validation", "test"):
        reference = None
        for name, runs in groups.items():
            scores, targets = group_scores(runs, args.protocol, partition, reference)
            reference = targets
            data[(name, partition)] = scores
        data[("targets", partition)] = reference

    #every non-empty set of groups, mixed at equal weight
    names = list(groups)
    combos = [combo for size in range(1, len(names) + 1) for combo in combinations(names, size)]
    rows = []
    for combo in combos:
        mixed = {partition: zscore_rows(np.mean([data[(name, partition)] for name in combo], axis=0))
                 for partition in ("validation", "test")}
        hub = fit_zscore(mixed["validation"])
        for corrected in (False, True):
            val = apply_correction(mixed["validation"], hub) if corrected else mixed["validation"]
            test = apply_correction(mixed["test"], hub) if corrected else mixed["test"]
            occupancy = top_k_occupancy(test)
            rows.append({
                "recipe": " + ".join(combo) + (", hub-corrected" if corrected else ""),
                "val": bacc(val, data[("targets", "validation")]),
                "test": bacc(test, data[("targets", "test")]),
                "shares": f"{100 * occupancy.min():.0f}-{100 * occupancy.max():.0f}%",
            })

    #the rule: the highest validation score; test is the honest number
    best = max(rows, key=lambda row: row["val"])
    n_val, n_test = len(data[("targets", "validation")]), len(data[("targets", "test")])
    lines = [
        "# Trial 17 -- which recipe should fill the lone-word rows? (local, isolated protocol)",
        "",
        "Lone words are scored on the private board only. The isolated protocol serves every word",
        f"alone with 1 s, as the holdout does: {n_val} validation and {n_test} test words.",
        "Groups: " + "; ".join(f"{name} ({len(runs)} runs)" for name, runs in groups.items()) + ".",
        "Mixes weigh groups equally; the hub correction is fitted on validation (no labels).",
        "Chosen on validation; test is the unbiased number. BAcc@10 in percent, chance 20.",
        "",
        "| recipe | val | test | top-10 shares (test) |",
        "|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda row: -row["val"]):
        mark = " **(chosen)**" if row is best else ""
        lines.append(f"| {row['recipe']}{mark} | {row['val']:.2f} | {row['test']:.2f} | {row['shares']} |")

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial21_lone_check.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[lone_check] wrote {out_path}")


if __name__ == "__main__":
    main()
