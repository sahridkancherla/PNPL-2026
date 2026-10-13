"""The holdout's cross-subject votes, timing removed: P1 graph, SNF, P3 LDA, P8a product.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.cross_graph --base submissions/ens_t22_tc1b75_ts.csv \\
        --holdout-dir data/holdout [--methods graph lda] [--settings data/trial23_testbed_settings.json]

The 13 Oct research note's rank-2 proposals on the holdout, through exactly
the pipeline the testbed measured (`crossvotes`):

    1   subject 0 and subjects 1-39: every matched sentence epoch at 50 Hz,
        robust-scaled per subject, low-passed at 20 Hz, MCCA'd into C shared
        components (C from the testbed)
    2   every row's window of each subject's shared series, residualized on
        the timing design every file shares
    3   each method's vote from the 39 OTHER subjects only, seeded by the base
        mix's posteriors; z(base) + w * z(vote) on the matched sentence rows,
        w from the testbed; every other row keeps its ranking exactly

With `--features <file>` -- P2's row-encoder embeddings (cli/row_encoder.py)
or P4's MEG-XL features (cli/megxl_features.py) -- each subject's rows are
that file's instead of steps 1-2, and the settings are that tool's own
testbed settings; the candidates are `<base stem>_x<method>_enc.csv` (P2) or
`_megxl.csv` (P4).

`--base` is a mix BEFORE the hub correction (the 70.68 file's is
`ens_t22_tc1b75_ts.csv`). Writes `<base stem>_x<method>.csv` per method and
runs/trial23_cross_graph_<stem>.md with the note's label-free diagnostics:
split-half reliability, the echo with the base and with the gap voter, the
subjects' reliability weights. Then hubnorm_holdout and lone_fill as for
every candidate. CPU; each subject's sentence file is read once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR
from .. import crossfeatures as cf
from .. import crossvotes as cv
from ..crosssubject import downsample, load_matches, read_sentences, robust_stats, sentence_file
from .ensemble import read_submission, softmax_rows, standardise_rows, zscore_scores
from .repeat_check import parse_subjects


def parse_args():
    """Command-line arguments for the holdout's cross-subject votes."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="a mix CSV before the hub correction")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--matches", default="data/trial22_matches.json", help="repo-relative, from cli.repeat_check")
    parser.add_argument("--subjects", default="1-39")
    parser.add_argument("--settings", default="data/trial23_testbed_settings.json", help="repo-relative, from cli.testbed")
    parser.add_argument("--methods", nargs="+", default=list(cv.METHODS), choices=cv.METHODS)
    parser.add_argument("--features", default="mcca",
                        help="'mcca' (residual MCCA windows), or a features file from cli.row_encoder (P2) or "
                             "cli.megxl_features (P4) -- then --settings is that tool's settings file, and the "
                             "candidates end in _enc or _megxl")
    parser.add_argument("--gap-voter", default="submissions/gap_voter_ts.csv", help="for the echo check; skipped if absent")
    parser.add_argument("--report", default=None, help="default runs/trial23_cross_graph_<base stem>.md")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def subject_pieces(holdout_dir, subject, epoch_order, own_epoch, lengths):
    """One subject's matched epochs at 50 Hz, robust-scaled, in subject 0's epoch order.

    ARGS:
        holdout_dir (str)
        subject (int)
        epoch_order (list): subject 0's matched epochs
        own_epoch (dict): subject 0's epoch -> this subject's
        lengths (dict): subject 0's epoch -> common length at 50 Hz

    RETURNS:
        list of [306, n] float64, one per epoch of `epoch_order`
    """
    data = read_sentences(sentence_file(holdout_dir, subject))
    pieces = []
    for epoch in epoch_order:
        #this subject's recording of the sentence, cut to the common length
        raw = data["meg"][own_epoch[epoch], :, : int(data["n_times"][own_epoch[epoch]])]
        pieces.append(downsample(raw)[:, : lengths[epoch]])

    #one scaler per subject, pooled over its matched sentences
    median, iqr = robust_stats(pieces)
    return [(piece - median[:, None]) / iqr[:, None] for piece in pieces]


def load_holdout(holdout_dir, matches_path, subject_spec):
    """Every subject's matched sentence epochs and subject 0's rows in them.

    Shared with cli.row_encoder (P2), so both read the holdout identically.

    ARGS:
        holdout_dir (str)
        matches_path (Path): from cli.repeat_check
        subject_spec (str): e.g. "1-39"

    RETURNS:
        dict: pieces (per subject, 0 first, of [306, n] robust-scaled 50 Hz
        epochs in `epoch_order`), others (list of int), epoch_order, metadata,
        voting [n_rows] bool, groups, onsets_s, piece_of_row, onset_in_piece
        (one entry per voting row, in canonical order)
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    matches = load_matches(matches_path)
    others = [subject for subject in parse_subjects(subject_spec) if subject != 0 and matches.get(subject)]
    epoch_order = sorted(set.intersection(*[{pair[0] for pair in matches[subject]} for subject in others]))

    #subject 0's rows in the matched epochs, in canonical order
    metadata = LibriBrainCompetitionHoldout(track="deep", data_path=holdout_dir).metadata
    voting = np.array([meta["source"] == "sentence" and meta["epoch"] in set(epoch_order) for meta in metadata])
    rows = [meta for meta, keep in zip(metadata, voting) if keep]
    groups = np.array([meta["epoch"] for meta in rows])
    onsets_s = np.array([meta["onset_s"] for meta in rows])

    #every subject's matched epochs, cut to the length all subjects share
    reference = read_sentences(sentence_file(holdout_dir, 0))
    lengths = {epoch: int(reference["n_times"][epoch]) // 5 for epoch in epoch_order}
    for subject in others:
        with np.load(sentence_file(holdout_dir, subject), allow_pickle=False) as stored:
            n_times = np.asarray(stored["sentence_n_times"]).astype(np.int64)
        for reference_epoch, own in matches[subject]:
            if reference_epoch in lengths:
                lengths[reference_epoch] = min(lengths[reference_epoch], int(n_times[own]) // 5)
    pieces = [subject_pieces(holdout_dir, 0, epoch_order, {epoch: epoch for epoch in epoch_order}, lengths)]
    for subject in others:
        pieces.append(subject_pieces(holdout_dir, subject, epoch_order, dict(matches[subject]), lengths))
        print(f"[holdout] subject {subject} read")
    return {"pieces": pieces, "others": others, "epoch_order": epoch_order, "metadata": metadata, "voting": voting,
            "groups": groups, "onsets_s": onsets_s, "piece_of_row": np.searchsorted(np.array(epoch_order), groups),
            "onset_in_piece": np.round(onsets_s * cf.RATE).astype(np.int64)}


def encoder_features(path, holdout):
    """P2's embeddings of the voting rows, checked against this holdout read.

    ARGS:
        path (Path): from cli.row_encoder
        holdout (dict): from `load_holdout`

    RETURNS:
        list (per subject, 0 first) of [n_rows, D]
    """
    with np.load(path) as stored:
        #⚠️ the same subjects and rows in the same order, or a vote mixes rows
        if [int(subject) for subject in stored["subjects"]] != [0] + holdout["others"]:
            raise SystemExit(f"{path}: its subjects are not this run's 0 + {holdout['others'][:3]}...")
        if not np.array_equal(stored["rows"], np.flatnonzero(holdout["voting"])):
            raise SystemExit(f"{path}: its rows are not this run's voting rows.")
        return [table.astype(np.float64) for table in stored["embeddings"]]


def main():
    """Shared series, residual windows, every method's vote, the candidates and the report."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    stored_settings = json.loads((REPO_ROOT / args.settings).read_text(encoding="utf-8"))
    settings = stored_settings["methods"]
    holdout = load_holdout(args.holdout_dir, REPO_ROOT / args.matches, args.subjects)
    metadata, voting, groups, others = holdout["metadata"], holdout["voting"], holdout["groups"], holdout["others"]
    epoch_order = holdout["epoch_order"]
    n_rows = int(voting.sum())

    #the features: each subject's residual MCCA windows, or P2's embeddings of them
    if args.features == "mcca":
        largest = max(settings[method]["components"] for method in args.methods)
        series, eigenvalues = cv.shared_series(holdout["pieces"], largest)
        projector = cf.residual_projector(cf.timing_design(groups, holdout["onsets_s"]))
        residual = [cf.residualize(cv.row_features(subject_series, holdout["piece_of_row"], holdout["onset_in_piece"]), projector)
                    .reshape(n_rows, largest, -1) for subject_series in series]
        feature_note = "MCCA eigenvalues " + ", ".join(f"{value:.1f}" for value in eigenvalues[:6]) + " ..."
    else:
        residual = [table[:, :, None] for table in encoder_features(REPO_ROOT / args.features, holdout)]
        feature_note = f"{stored_settings.get('features', 'row_encoder')} features ({residual[0].shape[1]}-d) from `{args.features}`"

    #each feature set's candidates by their own name: P2's _enc, P4's _megxl
    suffix = {"mcca": "", "megxl": "_megxl"}.get("mcca" if args.features == "mcca" else stored_settings.get("features"), "_enc")

    #the base, read as cli.ensemble reads a CSV
    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    index, primary, moses = read_submission(Path(args.base), *words)
    if not np.array_equal(index, np.arange(len(metadata))):
        raise SystemExit(f"{args.base}: its rows are not the holdout's 0..{len(metadata) - 1}.")
    base = standardise_rows(primary)
    gap_path = Path(args.gap_voter)
    gaps = standardise_rows(read_submission(gap_path, *words)[1])[voting] if gap_path.exists() else None

    lines = [f"# Trial 23 -- cross-subject votes, timing removed: `{Path(args.base).name}`", "",
             f"{len(others)} other subjects; {n_rows} rows in {len(epoch_order)} matched sentences; {feature_note}", "",
             "| method | C | setting | weight | testbed ses-12 gain | echo with the base | echo with the gap voter | file |",
             "|---|---|---|---|---|---|---|---|"]
    valid = cf.valid_pairs(groups)
    for method in args.methods:
        chosen = settings[method]
        #the encoder's settings have no C: every embedding dimension is used
        components = chosen.get("components")
        params = dict(chosen["params"])
        voters = [subject[:, :components, :].reshape(n_rows, -1) for subject in residual[1:]]
        if "temperature_scale" in params:
            params["temperature"] = params["temperature_scale"] / np.sqrt(len(voters))
        vote, info = cv.vote(method, voters, groups, base[voting], params)

        #the candidate: matched sentence rows voted on, every other row exactly the base
        combined = base.copy()
        combined[voting] = cv.blend(base[voting], vote, chosen["weight"])
        still = ~voting
        if not np.array_equal(np.argsort(-combined[still], axis=1, kind="stable"), np.argsort(-base[still], axis=1, kind="stable")):
            raise SystemExit("A row without votes changed its ranking.")
        out_path = Path(args.base).with_name(f"{Path(args.base).stem}_x{method}{suffix}.csv")
        if out_path.exists() and not args.force:
            raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
        write_submission(str(out_path), indices=index, primary_probs=softmax_rows(zscore_scores(combined)),
                         secondary_probs=moses / moses.sum(axis=1, keepdims=True))
        echo_gap = f"{cf.row_specific_correlation(vote, gaps):+.3f}" if gaps is not None else "-"
        lines.append(f"| {method} | {components} | {json.dumps(chosen['params'])} | {chosen['weight']} | {chosen['test_gain']:+.2f} | "
                     f"{cf.row_specific_correlation(vote, base[voting]):+.3f} | {echo_gap} | `{out_path.name}` |")
        if method == "graph":
            weights = info["weights"]
            lines.append(f"|  | reliability weights: median {np.median(weights):.3f}, range {weights.min():.3f}..{weights.max():.3f}; "
                         f"{int((weights == 0).sum())} subjects at 0 | | | | | | |")

    #split-half reliability of the graph, at the graph's C
    components = settings.get("graph", settings[args.methods[0]]).get("components")
    matrices = cv.similarity_matrices([subject[:, :components, :].reshape(n_rows, -1) for subject in residual[1:]], valid)
    halves = cf.split_half_reliability(matrices, valid)
    verdict = "near chance: the note says stop" if halves["neighbour_overlap"] < 2 * halves["chance"] else "above chance"
    lines += ["", f"Split-half reliability (the {len(others)} others in two halves): halves r = {halves['half_correlation']:+.3f}, "
              f"Spearman-Brown {halves['spearman_brown']:+.3f}; top-10 neighbour overlap {halves['neighbour_overlap']:.3f} "
              f"against chance {halves['chance']:.3f} ({verdict})."]
    report = Path(args.report) if args.report else RUNS_DIR / f"trial23_cross_graph_{Path(args.base).stem}{suffix}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[cross_graph] wrote the candidates beside {args.base} and {report}")


if __name__ == "__main__":
    main()
