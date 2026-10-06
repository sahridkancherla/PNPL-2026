"""Fuse a submission CSV with the word-pair table along the holdout's runs.

    PYTHONPATH=src python -m pnpl2026.trial15.cli.fuse \\
        --csv submissions/ens_g1g2_f1w1024_in_context.csv \\
        --temperature 0.54 [--strength 0.5] [--transitions gap] \\
        [--table data/trial15_wordpairs_train.npz] --holdout-dir data/holdout

Reads the CSV's 50 primary columns and z-scores each row's log-probabilities,
as `cli.ensemble` does. Every holdout "sentence" is a run of scored words; its
rows, in the loader's word order, are replaced by `wordpairs.fuse_run`'s scores
and turned back into distributions. Isolated words and the Moses columns are
copied unchanged. Take `--temperature` and `--strength` from `cli.fuse_check`,
run on the same models' dumps.

    --transitions gap        (default) each neighbouring pair's chain mixes
                             the adjacent and skip-one tables by P(adjacent |
                             its onset gap); the speech-rate factor and the
                             adjacent share are fitted on the holdout's own
                             gaps, inputs only. Writes <csv stem>_wpgap.csv
    --transitions adjacent   every pair taken as directly adjacent, as first
                             submitted on 2026-10-06 (lost 0.8-2.3 points).
                             Writes <csv stem>_wp.csv

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. Rows are matched to runs through the
holdout loader's own metadata `index`, never by position arithmetic; the output
keeps the input's index column and order, and is checked as cli.predict checks
before it is written.

⚠️ Like the in-context mode itself, this reads the other words of a run -- here,
their scores and onsets -- to score each word. The rules do not settle whether
that is allowed.

Light: CSV and metadata only, seconds on a login node or a laptop.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT
from ..wordpairs import GAP_SHARES, GAP_SHIFTS, adjacent_weight, fit_gap_mixture, fuse_run, load_table, step_transitions
from .ensemble import describe, read_submission, softmax_rows, standardise_rows

#the file each transitions mode writes, beside the input
SUFFIXES = {"gap": "_wpgap", "adjacent": "_wp"}


def parse_args():
    """Command-line arguments for fusing one CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="the submission CSV to fuse")
    parser.add_argument("--temperature", type=float, required=True, help="from cli.fuse_check")
    parser.add_argument("--strength", type=float, default=1.0, help="weight of the neighbours' evidence; 1 is exact Bayes")
    parser.add_argument("--transitions", choices=tuple(SUFFIXES), default="gap", help="gap-aware or adjacency-only chain")
    parser.add_argument("--table", default="data/trial15_wordpairs_train.npz", help="repo-relative word-pair table")
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--track", default="deep", help="competition track")
    parser.add_argument("--out", default=None, help="default <csv stem>_wpgap.csv (or _wp.csv) beside the input")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    return parser.parse_args()


def rank_preserving_distribution(scores):
    """Scores to row distributions that keep every rank readable in a CSV.

    The scorer reads rankings only. Softmaxing fused log-likelihoods directly
    gives very peaked rows, and a CSV's 6 decimals would round ranks 2-10 of a
    confident row to 0.000000 -- ties the scorer breaks arbitrarily. Each row is
    z-scored first, which changes no ranking and keeps every value well above
    the rounding floor, as `cli.ensemble`'s outputs are.

    ARGS:
        scores (np.ndarray): [n_rows, n_words]

    RETURNS:
        [n_rows, n_words] float64, rows summing to 1
    """
    #axis=1 is the word axis: each row on its own
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return softmax_rows(centred / np.where(spread > 0, spread, 1.0))


def holdout_runs(track, holdout_dir, n_rows):
    """The canonical rows and onsets of every holdout run, in word order.

    ARGS:
        track (str): "deep" is subject 0
        holdout_dir (str or None): folder holding COMPETITION_HOLDOUT/
        n_rows (int): rows in the submission, checked against the loader

    RETURNS:
        runs : list of (rows, onsets): row indices [length] and onsets in
            seconds [length] of each run, in word order
        sources : [n_rows] array of str, each row's holdout source
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    holdout = LibriBrainCompetitionHoldout(track=track, data_path=holdout_dir)

    #⚠️ the loader's metadata decides which row is which, never a range
    if len(holdout.metadata) != n_rows:
        raise SystemExit(f"The submission has {n_rows} rows; the holdout loader lists {len(holdout.metadata)}.")
    for position, meta in enumerate(holdout.metadata):
        if meta["index"] != position:
            raise SystemExit(f"Holdout metadata row {position} carries index {meta['index']}.")

    #a run is one sentence epoch; its words in slot order, as cli.predict serves them
    grouped = defaultdict(list)
    for meta in holdout.metadata:
        if meta["source"] == "sentence":
            grouped[(meta["subject"], meta["epoch"])].append(meta)
    runs = []
    for key in sorted(grouped):
        members = sorted(grouped[key], key=lambda meta: meta["word"])
        runs.append((np.array([meta["index"] for meta in members]), np.array([meta["onset_s"] for meta in members])))

    #every row in at most one run, or a word would be fused twice
    members = np.concatenate([rows for rows, _ in runs]) if runs else np.zeros(0, dtype=np.int64)
    if len(np.unique(members)) != len(members):
        raise SystemExit("A holdout row appears in two runs.")
    sources = np.array([meta["source"] for meta in holdout.metadata])
    return runs, sources


def main():
    """Read, fuse every run, check, and write."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}{SUFFIXES[args.transitions]}.csv")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    if args.temperature <= 0:
        raise SystemExit(f"--temperature must be > 0, got {args.temperature}")

    #⚠️ the table's columns are load_vocabulary("primary") order, as the CSV's
    table = load_table(REPO_ROOT / args.table)
    primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
    index, primary, moses = read_submission(in_path, primary_words, moses_words)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{in_path}: the index column is not 0..{len(index) - 1} in order.")

    runs, sources = holdout_runs(args.track, args.holdout_dir, len(index))
    fusable = [(rows, onsets) for rows, onsets in runs if len(rows) >= 2]

    #the gap mixture from every neighbouring pair's gap -- inputs, never labels
    if args.transitions == "gap":
        all_gaps = np.concatenate([np.diff(onsets) for _, onsets in fusable])
        shift, share = fit_gap_mixture(all_gaps, table["gap_models"])
        weights = adjacent_weight(all_gaps, table["gap_models"], shift, share)
        print(f"[fuse] holdout gaps: speech-rate factor x{np.exp(shift):.2f}, adjacent share {share:.2f}; "
              f"P(adjacent) per pair: mean {weights.mean():.2f}, under 0.5 for {100 * np.mean(weights < 0.5):.0f}%")

        #a fit on the grid's edge means the gap models do not describe these gaps
        if shift in (float(GAP_SHIFTS[0]), float(GAP_SHIFTS[-1])) or share in (float(GAP_SHARES[0]), float(GAP_SHARES[-1])):
            print("[fuse] WARNING: the gap mixture sits at the edge of its grid; the table's gap models do not fit "
                  "the holdout's gaps. Do not submit this file before checking cli.wordpairs' gap lines.")

    #Moses is copied, renormalised only: a CSV's rounding can leave row sums a
    #hair off 1, and the rankings do not change; axis=1 is the word axis
    moses = moses / moses.sum(axis=1, keepdims=True)

    #every row as the fusion reads it; rows outside a run keep their ranking
    standardised = standardise_rows(primary)
    fused = standardised / args.temperature
    for rows, onsets in fusable:
        chain = table["transitions"]
        if args.transitions == "gap":
            pair_weights = adjacent_weight(np.diff(onsets), table["gap_models"], shift, share)
            chain = step_transitions(pair_weights, table["transitions"], table["skip_transitions"])
        fused[rows] = fuse_run(standardised[rows], chain, table["stationary"], args.temperature, args.strength)
    fused_primary = rank_preserving_distribution(fused)

    #how much moved, on the rows that could move
    in_runs = np.concatenate([rows for rows, _ in fusable]) if fusable else np.zeros(0, dtype=np.int64)
    top1_changed = np.mean(primary[in_runs].argmax(axis=1) != fused_primary[in_runs].argmax(axis=1))
    before = np.argsort(-primary[in_runs], axis=1)[:, :10]
    after = np.argsort(-fused_primary[in_runs], axis=1)[:, :10]
    overlap = np.mean([len(set(old) & set(new)) for old, new in zip(before, after)])
    print(f"[fuse] {len(in_runs)} rows in {len(fusable)} runs fused ({args.transitions}) at temperature "
          f"{args.temperature:g}, strength {args.strength:g}; table {args.table} ({', '.join(table['partitions'])})")
    print(f"[fuse] top-1 changed on {100 * top1_changed:.1f}% of them; top-10 keeps {overlap:.2f} of 10 words on average")
    print(describe("input", primary, sources))
    print(describe("fused", fused_primary, sources))

    #⚠️ DO NOT REMOVE: the same checks predict makes before writing
    for name, values in (("primary", fused_primary), ("moses", moses)):
        sums = values.sum(axis=1)
        if not np.isfinite(values).all() or not np.allclose(sums, 1.0, atol=1e-6):
            raise SystemExit(f"{name} rows are not distributions (sums {sums.min():.6f}..{sums.max():.6f}).")
    if fused_primary.shape != primary.shape:
        raise SystemExit(f"Fused shape {fused_primary.shape} differs from the input's {primary.shape}.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=index, primary_probs=fused_primary, secondary_probs=moses)
    print(f"[fuse] wrote {out_path}: {len(index)} rows")
    print("[fuse] submission is MANUAL, through the competition web UI.")


if __name__ == "__main__":
    main()
