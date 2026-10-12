"""Add other subjects' votes to subject 0's rows: change 1 of the Route to 73.

    # route 2: subject 0's own members on the virtual subject (cli.align)
    PYTHONPATH=src python -m pnpl2026.trial22.cli.cross_votes --base submissions/ens_k18_spw_ts.csv \\
        --virtual-dir data/trial22_virtual/mcca --votes submissions/t22_virtual/mcca/*.csv \\
        --weight 0.15 --out submissions/ens_k18_spw_x2w15_ts.csv --holdout-dir data/holdout

    # route 1: the multi-subject model on each subject's own files (cli.predict --subject K)
    PYTHONPATH=src python -m pnpl2026.trial22.cli.cross_votes --base submissions/ens_k18_spw_ts.csv \\
        --votes submissions/trial18_all_g2_s*_subj*_in_context.csv --matches data/trial22_matches.json \\
        --weight 0.15 --out submissions/ens_k18_spw_x1w15_ts.csv --holdout-dir data/holdout

Every CSV is read as cli.ensemble reads it (each row's log-probabilities
z-scored). The votes:

    route 2   the virtual CSVs are in subject 0's own row order (the virtual
              holdout copies its metadata); their average is the vote on the
              sentences `matched.json` lists
    route 1   each subject's seeds averaged, then its rows mapped onto
              subject 0's through the repeat check's match table (epoch, then
              word slot; onsets checked equal); each subject weighted by its
              agreement with the base -- the mean correlation of matched rows,
              floored at 0, as the doc proposes for choosing weights without
              labels

The result, on matched sentence rows only: z(base + weight * z(vote)). Every
other row -- unmatched sentences, lone words -- keeps the base's ranking
exactly. Then `hubnorm_holdout` and `lone_fill`, as for every candidate.
Writes the CSV and runs/trial22_cross_votes_<stem>.md.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR
from ..crosssubject import load_matches
from .ensemble import read_submission, softmax_rows, standardise_rows, zscore_scores

#a route-1 file names its subject: <run>_subjKK_...
SUBJECT_PATTERN = re.compile(r"_subj(\d{2})_")


def parse_args():
    """Command-line arguments for adding cross-subject votes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="subject 0's mix, before the hub correction")
    parser.add_argument("--votes", nargs="+", required=True, help="the vote CSVs, of one route")
    parser.add_argument("--virtual-dir", default=None, help="route 2: the virtual holdout, with matched.json")
    parser.add_argument("--matches", default="data/trial22_matches.json", help="route 1: repo-relative match table")
    parser.add_argument("--weight", type=float, required=True, help="the votes' weight against the base's 1")
    parser.add_argument("--min-subject-weight", type=float, default=0.0, help="route 1: drop subjects agreeing less")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def zscore_rows(scores):
    """Each row centred and scaled across its words.

    ARGS:
        scores (np.ndarray): [N, n_words]

    RETURNS:
        [N, n_words] float64
    """
    #axis=1 is the word axis
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def row_correlation(first, second):
    """Each row's correlation between two score tables.

    ARGS:
        first, second (np.ndarray): [N, n_words]

    RETURNS:
        [N] float64
    """
    #z-scored rows: the mean product is the correlation
    return np.mean(zscore_rows(first) * zscore_rows(second), axis=1)


def top10_overlap(first, second):
    """Mean share of the top 10 two tables agree on.

    ARGS:
        first, second (np.ndarray): [N, n_words]

    RETURNS:
        float in [0, 1]
    """
    #axis=1: each row's ten best words, in any order
    top_first = np.argpartition(-first, 9, axis=1)[:, :10]
    top_second = np.argpartition(-second, 9, axis=1)[:, :10]
    return float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top_first, top_second)]))


def read_standardised(path, words):
    """One CSV's primary scores, as cli.ensemble reads them.

    ARGS:
        path (str)
        words (tuple): (primary, moses) vocabularies

    RETURNS:
        index [N], standardised [N, 50], moses [N, 50]
    """
    index, primary, moses = read_submission(Path(path), *words)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{path}: the index column is not 0..{len(index) - 1} in order.")
    return index, standardise_rows(primary), moses


def subject_metadata(subject, holdout_dir):
    """A subject's canonical rows: (source, epoch, word slot, onset) each.

    ARGS:
        subject (int)
        holdout_dir (str or None)

    RETURNS:
        list of dict, the loader's metadata in canonical order
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    return LibriBrainCompetitionHoldout(subjects=[subject], data_path=holdout_dir).metadata


def virtual_votes(paths, virtual_dir, words, n_rows):
    """Route 2: the virtual CSVs' average and the rows it may vote on.

    ARGS:
        paths (list): the members' CSVs predicted on the virtual holdout
        virtual_dir (str): with matched.json
        words (tuple)
        n_rows (int): the base's rows

    RETURNS:
        votes [N, 50] (zero where no vote), voting [N] bool, lines (list)
    """
    tables = []
    for path in paths:
        index, standardised, _ = read_standardised(path, words)
        if len(index) != n_rows:
            raise SystemExit(f"{path} has {len(index)} rows; the base has {n_rows}.")
        tables.append(standardised)
    votes = zscore_rows(np.mean(tables, axis=0))

    #the matched sentences' rows only, by the virtual holdout's own record
    info = json.loads((Path(virtual_dir) / "matched.json").read_text(encoding="utf-8"))
    matched = set(int(epoch) for epoch in info["matched_epochs"])
    from pnpl.competition import LibriBrainCompetitionHoldout

    metadata = LibriBrainCompetitionHoldout(track="deep", data_path=virtual_dir).metadata
    voting = np.array([meta["source"] == "sentence" and meta["epoch"] in matched for meta in metadata])
    lines = [f"Route 2: {len(paths)} members on the virtual holdout `{Path(virtual_dir).name}` "
             f"({info['mode']}, C = {info.get('components')}, {len(info['subjects'])} subjects)."]
    return votes, voting, lines


def subject_votes(paths, matches_path, holdout_dir, words, base, min_weight):
    """Route 1: every subject's votes mapped onto subject 0's rows, weighted by agreement.

    ARGS:
        paths (list): `<run>_subjKK_...csv` files, any seeds, any subjects
        matches_path (Path): the repeat check's table
        holdout_dir (str or None)
        words (tuple)
        base (np.ndarray): [N, 50] subject 0's standardised scores
        min_weight (float): subjects agreeing less are dropped

    RETURNS:
        votes [N, 50], voting [N] bool, lines (list)
    """
    by_subject = defaultdict(list)
    for path in paths:
        found = SUBJECT_PATTERN.search(Path(path).name)
        if not found:
            raise SystemExit(f"{path}: no _subjKK_ in its name, so no subject to map from.")
        by_subject[int(found.group(1))].append(path)
    matches = load_matches(matches_path)

    #subject 0's rows by (epoch, word slot), sentences only
    reference = subject_metadata(0, holdout_dir)
    row_of = {(meta["epoch"], meta["word"]): meta["index"] for meta in reference if meta["source"] == "sentence"}
    onset_of = {meta["index"]: meta["onset_s"] for meta in reference}

    mapped, agreements = {}, {}
    for subject, files in sorted(by_subject.items()):
        tables = [read_standardised(path, words)[1] for path in files]
        own = zscore_rows(np.mean(tables, axis=0))
        to_reference = {other: reference_epoch for reference_epoch, other in matches.get(subject, [])}

        #each of this subject's sentence rows onto subject 0's row of the same word
        table = np.full_like(base, np.nan)
        for meta in subject_metadata(subject, holdout_dir):
            if meta["source"] != "sentence" or meta["epoch"] not in to_reference:
                continue
            row = row_of.get((to_reference[meta["epoch"]], meta["word"]))
            #⚠️ the same audio has the same onsets: anything else is a broken map
            if row is None or abs(onset_of[row] - meta["onset_s"]) > 1e-6:
                raise SystemExit(f"Subject {subject} epoch {meta['epoch']} slot {meta['word']} does not map onto subject 0.")
            table[row] = own[meta["index"]]
        present = ~np.isnan(table).any(axis=1)
        agreements[subject] = float(np.mean(row_correlation(base[present], table[present])))
        mapped[subject] = table

    #weights by agreement with subject 0, floored at zero
    weights = {subject: max(value, 0.0) for subject, value in agreements.items() if value > min_weight}
    if not weights or sum(weights.values()) == 0:
        raise SystemExit(f"No subject agrees with the base above {min_weight}: {agreements}")
    total = sum(weights.values())
    stacked = np.stack([np.nan_to_num(mapped[subject]) * weights[subject] / total for subject in weights], axis=0)
    covered = np.stack([~np.isnan(mapped[subject]).any(axis=1) for subject in weights], axis=0)
    voting = covered.all(axis=0)
    votes = zscore_rows(stacked.sum(axis=0))
    lines = [f"Route 1: {len(by_subject)} subjects, {len(paths)} files; agreement with the base (mean row "
             "correlation) and weight:", ""]
    lines += [f"- subject {subject}: {agreements[subject]:+.3f}" + (f", weight {weights[subject] / total:.3f}" if subject in weights else ", dropped")
              for subject in sorted(agreements)]
    return votes, voting, lines


def main():
    """Read the base and the votes, combine the matched rows, write the CSV and report."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    index, base, moses = read_standardised(args.base, words)

    #one route per call: the virtual holdout's rows, or each subject's own
    if args.virtual_dir:
        votes, voting, lines = virtual_votes(args.votes, args.virtual_dir, words, len(index))
    else:
        votes, voting, lines = subject_votes(args.votes, REPO_ROOT / args.matches, args.holdout_dir, words, base, args.min_subject_weight)

    #the combination on voting rows; every other row exactly the base
    combined = base.copy()
    combined[voting] = zscore_rows(base[voting] + args.weight * votes[voting])
    lines += ["", f"Weight {args.weight:g} against the base's 1, on {int(voting.sum())} of {len(index)} rows.",
              f"Votes against the base on those rows: mean row correlation {np.mean(row_correlation(base[voting], votes[voting])):+.3f}, "
              f"top-10 overlap {100 * top10_overlap(base[voting], votes[voting]):.1f}%.",
              f"Result against the base: top-10 overlap {100 * top10_overlap(base[voting], combined[voting]):.1f}%."]

    #⚠️ rows that do not vote must keep their ranking exactly
    still = ~voting
    if not np.array_equal(np.argsort(-combined[still], axis=1, kind="stable"), np.argsort(-base[still], axis=1, kind="stable")):
        raise SystemExit("A row without votes changed its ranking.")
    primary = softmax_rows(zscore_scores(combined))
    moses = moses / moses.sum(axis=1, keepdims=True)
    for name, values in (("primary", primary), ("moses", moses)):
        sums = values.sum(axis=1)
        if not np.isfinite(values).all() or not np.allclose(sums, 1.0, atol=1e-6):
            raise SystemExit(f"{name} rows are not distributions.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=index, primary_probs=primary, secondary_probs=moses)
    report = RUNS_DIR / f"trial22_cross_votes_{out_path.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join([f"# Trial 22 -- cross-subject votes: `{out_path.name}`", ""] + lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[cross_votes] wrote {out_path} and {report}")


if __name__ == "__main__":
    main()
