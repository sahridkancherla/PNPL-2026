"""Average several submission CSVs into one ensemble submission.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.ensemble \\
        --inputs submissions/trial10_f1_s0_in_context.csv \\
                 submissions/trial10_f3_s0_in_context.csv ... \\
        --out submissions/ensemble_t10t12_in_context.csv

    PYTHONPATH=src python -m pnpl2026.trial22.cli.ensemble \\
        --group G=a.csv,b.csv,c.csv --group T19=d.csv,e.csv,f.csv \\
        --weights T19=2 --out submissions/mix.csv

Per input file and per row, the 50 log-probabilities are z-scored across the
words (which removes each model's own temperature), averaged over the files,
and turned back into a distribution with a softmax. The Moses columns are
treated the same way. Ranking is all BAcc@k reads; the softmax only makes the
rows valid distributions again.

    --lone-inputs        a lone-word specialist's CSVs (trial 15's I3). The
                         holdout's lone words (source "word") take the general
                         ensemble and the specialist ensemble at EQUAL weight,
                         each re-standardised first -- a rule fixed in
                         advance, never fitted. Sentence rows are the general
                         ensemble alone, unchanged. Uses each row's source,
                         which the loader states per item: within the
                         holdout principle. Needs the holdout files
    --column-normalise   off by default. After averaging, z-score each word's
                         column across the rows of each holdout source, so no
                         word is in every top 10 and none is in no top 10.
                         ⚠️ It pools all holdout rows: against the holdout
                         principle (CLAUDE.md); never for a final pick

    --group NAME=f,f,...  trial 20: families instead of one flat list. Each
                         group's files are z-scored and averaged, then the
                         groups are averaged with `--weights` (default 1
                         each). With equal weights and equal-sized groups this
                         is exactly the flat average; a family of twelve no
                         longer outvotes one of three by its size alone.
                         Exclusive with --inputs
    --weights NAME=w,... each group's weight; unnamed groups weigh 1

Why ensemble: on the holdout, trial 10/12 models with equal local scores agree
on only 30-48% of their top-1 picks, so their errors are far from identical.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. Every input must carry the canonical
`index` column 0..N-1 in order, and the output keeps it. Any mismatch stops
the script before anything is written.

Light: it reads CSVs only, so it runs on a login node or a laptop.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#the CSV's first column, the canonical row number
INDEX_COLUMN = "index"

#write_submission's prefix for the secondary (Moses) columns
MOSES_PREFIX = "moses_"

#floor before the log: a probability of exactly 0 would give -inf
PROBABILITY_FLOOR = 1e-12


def parse_args():
    """Command-line arguments for ensembling."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", default=None, help="submission CSVs to average, one vote each")
    parser.add_argument(
        "--group",
        action="append",
        default=None,
        help="NAME=csv,csv,... : one family, averaged before the families are; repeat per family",
    )
    parser.add_argument("--weights", default="", help="NAME=w,NAME=w : the families' weights (default 1 each)")
    parser.add_argument("--out", required=True, help="the ensemble CSV to write")
    parser.add_argument(
        "--lone-inputs",
        nargs="+",
        default=None,
        help="a lone-word specialist's CSVs, mixed in at equal weight on the holdout's lone words only",
    )
    parser.add_argument(
        "--column-normalise",
        action="store_true",
        help="z-score each word's column within each holdout source (pools all holdout rows: against the principle)",
    )
    parser.add_argument("--track", default="deep", help="competition track, for the source of each row")
    parser.add_argument(
        "--holdout-dir",
        default=None,
        help="folder holding COMPETITION_HOLDOUT/; needed by --lone-inputs and --column-normalise",
    )
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    return parser.parse_args()


def read_submission(path, primary_words, moses_words):
    """One submission CSV as arrays, after checking its columns.

    ARGS:
        path (Path): a CSV written by write_submission
        primary_words (list): the competition's 50 words, in column order
        moses_words (list): the Moses 50 words, in column order

    RETURNS:
        index : [N] int64, the canonical row numbers
        primary : [N, 50] float64 probabilities
        moses : [N, 50] float64 probabilities
    """
    table = pd.read_csv(path)

    #⚠️ the header must be exactly what write_submission writes, in order: a
    #CSV from another vocabulary would average the wrong words together
    expected = [INDEX_COLUMN] + list(primary_words) + [MOSES_PREFIX + word for word in moses_words]
    if list(table.columns) != expected:
        raise SystemExit(f"{path}: columns are not the competition's index + primary + moses layout.")

    index = table[INDEX_COLUMN].to_numpy(dtype=np.int64)
    primary = table[list(primary_words)].to_numpy(dtype=np.float64)
    moses = table[[MOSES_PREFIX + word for word in moses_words]].to_numpy(dtype=np.float64)
    return index, primary, moses


def standardise_rows(probabilities):
    """Log-probabilities z-scored across each row's words.

    ARGS:
        probabilities (np.ndarray): [N, n_words], each row a distribution

    RETURNS:
        [N, n_words] float64 - zero mean and unit spread per row
    """
    #axis=1 is the word axis: each row is ranked on its own
    logs = np.log(np.maximum(probabilities, PROBABILITY_FLOOR))
    centred = logs - logs.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)

    #a perfectly flat row has no ranking; leave it flat rather than divide by 0
    return centred / np.where(spread > 0, spread, 1.0)


def zscore_scores(scores):
    """Scores (not probabilities) z-scored across each row's words.

    ARGS:
        scores (np.ndarray): [N, n_words]

    RETURNS:
        [N, n_words] float64 - zero mean and unit spread per row
    """
    #axis=1 is the word axis: each row on its own; a flat row stays flat
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def normalise_columns(scores, groups):
    """Z-score each word's column across the rows of each group.

    ARGS:
        scores (np.ndarray): [N, n_words]
        groups (np.ndarray): [N] group label per row (the holdout source)

    RETURNS:
        [N, n_words] float64
    """
    out = np.empty_like(scores)
    for group in np.unique(groups):
        rows = groups == group

        #axis=0 is the row axis: each word is compared with itself across rows
        block = scores[rows]
        centred = block - block.mean(axis=0, keepdims=True)
        spread = centred.std(axis=0, keepdims=True)
        out[rows] = centred / np.where(spread > 0, spread, 1.0)
    return out


def softmax_rows(scores):
    """Scores back to one distribution per row.

    ARGS:
        scores (np.ndarray): [N, n_words]

    RETURNS:
        [N, n_words] float64, rows summing to 1
    """
    #subtract the row maximum first, so exp never overflows; axis=1 is words
    shifted = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def holdout_sources(track, holdout_dir, n_rows):
    """The source ("sentence" or "word") of every canonical row.

    ARGS:
        track (str): "deep" is subject 0
        holdout_dir (str or None): local COMPETITION_HOLDOUT/ parent
        n_rows (int): rows in the submissions, checked against the loader

    RETURNS:
        [n_rows] array of str
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    holdout = LibriBrainCompetitionHoldout(track=track, data_path=holdout_dir)

    #⚠️ the loader's own metadata decides which row is which; never a range
    if len(holdout.metadata) != n_rows:
        raise SystemExit(f"The submissions have {n_rows} rows; the holdout loader lists {len(holdout.metadata)}.")
    for position, meta in enumerate(holdout.metadata):
        if meta["index"] != position:
            raise SystemExit(f"Holdout metadata row {position} carries index {meta['index']}.")
    return np.array([meta["source"] for meta in holdout.metadata])


def describe(name, probabilities, sources=None):
    """One line on how a prediction matrix spreads over the 50 words.

    ARGS:
        name (str): label for the line
        probabilities (np.ndarray): [N, 50]
        sources (np.ndarray or None): [N] source per row, for a per-source split

    RETURNS:
        str
    """
    parts = []
    groups = [("all", np.ones(len(probabilities), dtype=bool))]
    if sources is not None:
        groups += [(source, sources == source) for source in np.unique(sources)]
    for label, rows in groups:
        #share of rows in which each word reaches the top 10; 0.20 when even
        top10 = np.argsort(-probabilities[rows], axis=1)[:, :10]
        occupancy = np.bincount(top10.ravel(), minlength=probabilities.shape[1]) / rows.sum()
        distinct = len(np.unique(probabilities[rows].argmax(axis=1)))
        parts.append(f"{label}: top-1 covers {distinct}, top-10 share {occupancy.min():.3f}-{occupancy.max():.3f}")
    return f"[ensemble] {name}: " + "; ".join(parts)


def parse_groups(group_specs, weight_spec):
    """The families and their weights, from the command line.

    ARGS:
        group_specs (list): "NAME=a.csv,b.csv" strings
        weight_spec (str): "NAME=w,NAME=w", may be empty

    RETURNS:
        list of (name, [paths], weight)
    """
    groups = []
    for spec in group_specs:
        name, _, paths = spec.partition("=")
        files = [path for path in paths.split(",") if path]
        if not name or not files:
            raise SystemExit(f"--group {spec!r} is not NAME=csv,csv,...")
        if name in [group[0] for group in groups]:
            raise SystemExit(f"--group {name} is given twice.")
        groups.append([name, files, 1.0])

    #weights by name; a name not among the groups is a typo, refused
    known = {group[0]: group for group in groups}
    for item in [item for item in weight_spec.split(",") if item]:
        name, _, value = item.partition("=")
        if name not in known:
            raise SystemExit(f"--weights names {name!r}, which is not a group ({sorted(known)}).")
        weight = float(value)
        if weight <= 0:
            raise SystemExit(f"--weights {name}={value}: weights must be > 0.")
        known[name][2] = weight
    return [tuple(group) for group in groups]


def main():
    """Read, check, average, mix lone words if asked, and write."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")

    #one flat list (one vote per file), or families with weights; not both
    if bool(args.inputs) == bool(args.group):
        raise SystemExit("Give either --inputs or --group, not both and not neither.")
    if args.weights and not args.group:
        raise SystemExit("--weights needs --group.")
    groups = parse_groups(args.group, args.weights) if args.group else [("all", list(args.inputs), 1.0)]
    args.inputs = [path for _, files, _ in groups for path in files]
    lone_paths = list(args.lone_inputs or [])
    every_path = list(args.inputs) + lone_paths
    if len(set(every_path)) != len(every_path):
        raise SystemExit("An input is listed twice (or in both lists); it would count double in the average.")
    if args.column_normalise:
        print("[ensemble] WARNING: --column-normalise pools all holdout rows -- against the holdout "
              "principle; never submit this file as a final pick")

    primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
    loaded = [read_submission(Path(path), primary_words, moses_words) for path in args.inputs]
    lone_loaded = [read_submission(Path(path), primary_words, moses_words) for path in lone_paths]

    #⚠️ every file must hold the canonical rows 0..N-1 in order
    index = loaded[0][0]
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{args.inputs[0]}: the index column is not 0..{len(index) - 1} in order.")
    for path, (other, _, _) in zip(every_path, loaded + lone_loaded):
        if not np.array_equal(other, index):
            raise SystemExit(f"{path}: its index column differs from {args.inputs[0]}'s.")

    #the source of each row, as the loader states it item by item
    sources = None
    if args.column_normalise or lone_paths:
        sources = holdout_sources(args.track, args.holdout_dir, len(index))

    #each family's files averaged, then the families by weight; with one
    #family this is the flat [n_files, N, 50] -> [N, 50] mean over files
    primary_parts, moses_parts, weights, start = [], [], [], 0
    for name, files, weight in groups:
        members = loaded[start : start + len(files)]
        start += len(files)
        primary_parts.append(np.mean([standardise_rows(primary) for _, primary, _ in members], axis=0))
        moses_parts.append(np.mean([standardise_rows(moses) for _, _, moses in members], axis=0))
        weights.append(weight)
    shares = np.array(weights) / np.sum(weights)

    #axis 0 is the family axis
    primary_scores = np.tensordot(shares, np.stack(primary_parts, axis=0), axes=1)
    moses_scores = np.tensordot(shares, np.stack(moses_parts, axis=0), axes=1)
    if len(groups) > 1:
        print("[ensemble] families: " + ", ".join(
            f"{name} ({len(files)} files) {100 * share:.1f}%" for (name, files, _), share in zip(groups, shares)
        ))

    #lone words: general and specialist at equal weight, each re-standardised
    #so neither wins by spread; sentence rows stay the general ensemble's
    if lone_paths:
        lone = sources == "word"
        specialist_primary = np.mean([standardise_rows(primary) for _, primary, _ in lone_loaded], axis=0)
        specialist_moses = np.mean([standardise_rows(moses) for _, _, moses in lone_loaded], axis=0)
        primary_scores[lone] = 0.5 * zscore_scores(primary_scores[lone]) + 0.5 * zscore_scores(specialist_primary[lone])
        moses_scores[lone] = 0.5 * zscore_scores(moses_scores[lone]) + 0.5 * zscore_scores(specialist_moses[lone])
        print(f"[ensemble] {int(lone.sum())} lone-word rows mixed with {len(lone_paths)} specialist files at equal weight")
    if args.column_normalise:
        primary_scores = normalise_columns(primary_scores, sources)
        moses_scores = normalise_columns(moses_scores, sources)
    primary, moses = softmax_rows(primary_scores), softmax_rows(moses_scores)

    #how far each input's top-1 sits from the ensemble's
    for path, (_, single, _) in zip(args.inputs, loaded):
        agreement = np.mean(single.argmax(axis=1) == primary.argmax(axis=1))
        print(f"[ensemble] {Path(path).name}: top-1 agrees with the ensemble on {100 * agreement:.1f}% of rows")
    print(describe("ensemble", primary, sources))

    #⚠️ DO NOT REMOVE: the same checks predict makes before writing
    for name, table in (("primary", primary), ("moses", moses)):
        sums = table.sum(axis=1)
        if not np.isfinite(table).all() or not np.allclose(sums, 1.0, atol=1e-6):
            raise SystemExit(f"{name} rows are not distributions (sums {sums.min():.6f}..{sums.max():.6f}).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=index, primary_probs=primary, secondary_probs=moses)
    print(f"[ensemble] wrote {out_path}: {len(index)} rows from {len(args.inputs)} files"
          + (f", lone words with {len(lone_paths)} specialist files" if lone_paths else "")
          + (", columns normalised per source" if args.column_normalise else ""))
    print("[ensemble] submission is MANUAL, through the competition web UI.")


if __name__ == "__main__":
    main()
