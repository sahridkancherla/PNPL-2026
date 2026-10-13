"""Write the doc's per-word correction variants of one mix: change 4 of the Route to 73.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.correct --csv submissions/ens_k18_spw_ts.csv \\
        --runs trial13_g1_s0 ... --holdout-dir data/holdout [--variants hubw emsk hubpos quota qbnorm]

The input is a mix BEFORE any hub correction (cli.ensemble's output). Each
variant (`corrections.py`) is fitted on the holdout's 868 sentence rows --
except qbnorm, frozen on a validation bank -- and written as
`<stem>_<variant>.csv`; lone rows keep their ranking exactly (lone_fill
replaces them next). The members' validation dumps (`--runs`, reference
protocol) give what needs labels or a bank: the temperature, the BCTS
calibration, the validation word shares, QB-Norm's bank.

The report holds the doc's two checks:

    label-free   how often each word enters a top 10 -- the spread should
                 narrow without pushing frequent words far below their
                 estimated share
    local        every variant and the current z-score (and balance_half,
                 the pending hubb) fitted transductively on `--check` draws
                 of the members' reference TEST tokens, non-Sherlock,
                 resampled to the holdout's estimated word mix; gains are
                 against the z-score. The doc: drop any variant that loses

Seconds to minutes on a login node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...config import RUNS_DIR
from ..corrections import emsk, fit_bcts, hubpos, hubw, position_groups, qbnorm, quota, softmax_rows
from ..hubnorm import apply_correction, fit_balance, fit_zscore, top_k_occupancy
from ..local_checks import bacc, eligible_tokens, ensemble, member_dumps, summarise_gains, uneven_draws
from ..transduction import fit_temperature_nll
from .ensemble import read_submission, standardise_rows, zscore_scores

#every variant this tool writes
VARIANTS = ("hubw", "emsk", "hubpos", "quota", "qbnorm")


def parse_args():
    """Command-line arguments for the correction variants."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="a mix CSV before any hub correction")
    parser.add_argument("--runs", nargs="+", required=True, help="the mix's members (names or directories), for their dumps")
    parser.add_argument("--protocol", default="reference", help="the dumps read")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument("--check", type=int, default=200, help="local draws (0 skips the check)")
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--track", default="deep")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


class Calibration:
    """What the variants need from the members' validation dumps.

    ATTRS:
        temperature (float): NLL-fitted, for hubw
        bcts (tuple): (temperature, biases [50]) for emsk and quota
        source_shares (np.ndarray): [50] the validation word shares
        bank (np.ndarray): [B, 50] the validation ensemble's sentence rows, for qbnorm
    """

    def __init__(self, runs, protocol):
        """
        ARGS:
            runs (list): the members
            protocol (str)
        """
        dumps = member_dumps(runs, protocol, "validation")
        missing = [run for run, dump in zip(runs, dumps) if dump is None]
        if missing:
            raise SystemExit(f"No {protocol} validation dump for {missing}: they are needed for calibration.")

        #the validation ensemble, built as a CSV mix is
        scores = ensemble([dump["scores"] for dump in dumps])
        keep = (dumps[0]["primary"] >= 0) & ~dumps[0]["isolated"]
        labels = dumps[0]["primary"][keep]
        self.bank = scores[keep]
        self.temperature = fit_temperature_nll(self.bank, labels)
        self.bcts = fit_bcts(self.bank, labels)
        self.source_shares = np.bincount(labels, minlength=50) / len(labels)


def apply_variant(name, scores, calibration, groups):
    """One variant fitted on the rows it corrects (qbnorm: on its frozen bank).

    ARGS:
        name (str): one of VARIANTS, "zscore" or "balance_half"
        scores (np.ndarray): [N, 50] standardised
        calibration (Calibration)
        groups (np.ndarray): [N] position group of each row

    RETURNS:
        [N, 50]
    """
    if name == "zscore":
        return apply_correction(scores, fit_zscore(scores))
    if name == "balance_half":
        return apply_correction(scores, fit_balance(scores), 0.5)
    if name == "hubw":
        return hubw(scores, calibration.temperature)
    if name == "emsk":
        return emsk(scores, *calibration.bcts, calibration.source_shares)[0]
    if name == "hubpos":
        return hubpos(scores, groups)
    if name == "quota":
        shares = emsk(scores, *calibration.bcts, calibration.source_shares)[1]
        return quota(scores, shares)
    if name == "qbnorm":
        return qbnorm(scores, calibration.bank)
    raise ValueError(f"Unknown variant {name!r}")


def local_check(args, calibration, shares, lines):
    """Every variant against the z-score on uneven draws of the members' test tokens.

    ARGS:
        args (argparse.Namespace)
        calibration (Calibration)
        shares (np.ndarray): [50] the holdout's estimated shares, for the draws
        lines (list): report lines, appended to
    """
    tests = member_dumps(args.runs, args.protocol, "test")
    if any(test is None for test in tests):
        lines.append("Local check skipped: a member has no test dump.")
        return
    scores, labels = ensemble([test["scores"] for test in tests]), tests[0]["primary"]
    sequence = tests[0]["sequence"] if tests[0]["sequence"] is not None else np.arange(len(labels))
    groups = position_groups(sequence)
    names = ["balance_half"] + list(args.variants)
    gains = {name: [] for name in names}
    baseline_scores = []
    for draw in uneven_draws(labels, eligible_tokens(tests[0]), shares, n_draws=args.check):
        #each draw is its own "holdout": every fit made on it
        baseline = bacc(apply_variant("zscore", scores[draw], calibration, groups[draw]), labels[draw])
        baseline_scores.append(baseline)
        for name in names:
            gains[name].append(bacc(apply_variant(name, scores[draw], calibration, groups[draw]), labels[draw]) - baseline)
    lines += ["", f"## Local check: {len(baseline_scores)} uneven draws of 868 {args.protocol} test tokens, non-Sherlock",
              "", f"The z-score (the current hubh) scores {np.mean(baseline_scores):.2f} on average. Gains against it:", "",
              "| variant | gain (mean +/- sd, share of draws won) | verdict |", "|---|---|---|"]
    for name in names:
        verdict = "keep" if np.mean(gains[name]) >= 0 else "DROP (loses locally)"
        lines.append(f"| {name} | {summarise_gains(gains[name])} | {verdict} |")


def main():
    """Fit every variant on the holdout's sentence rows, write each, report."""
    args = parse_args()
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary, write_submission

    in_path = Path(args.csv)
    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    index, primary, moses = read_submission(in_path, *words)
    metadata = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir).metadata
    if not np.array_equal(index, np.arange(len(metadata))):
        raise SystemExit(f"{in_path}: its rows are not the holdout's 0..{len(metadata) - 1}.")

    #sentence rows, contiguous per sentence in canonical order, and their positions
    sentence = np.array([meta["source"] == "sentence" for meta in metadata])
    epochs = np.array([meta["epoch"] for meta in metadata])[sentence]
    groups = position_groups(epochs)
    standardised = standardise_rows(primary)
    scores = standardised[sentence]
    calibration = Calibration(args.runs, args.protocol)

    #the holdout's estimated shares: the label-free check's reference and the draws' mix
    _, shares = emsk(scores, *calibration.bcts, calibration.source_shares)
    vocabulary = list(words[0])
    frequent = np.argsort(-shares)[:5]
    lines = [f"# Trial 22 -- correction variants of `{in_path.name}`", "",
             f"Validation calibration: temperature {calibration.temperature:.3f}; BCTS temperature {calibration.bcts[0]:.3f}; "
             f"bank {len(calibration.bank)} rows.",
             "Holdout shares estimated (BCTS + EM), highest: "
             + ", ".join(f"{vocabulary[word]} {100 * shares[word]:.1f}%" for word in frequent), "",
             "| variant | top-10 shares (min..max) | " + " | ".join(f"{vocabulary[word]} (share {100 * shares[word]:.0f}%)" for word in frequent) + " | file |",
             "|---|---|" + "---|" * len(frequent) + "---|"]

    for name in ["zscore", "balance_half"] + list(args.variants):
        corrected = apply_variant(name, scores, calibration, groups)
        occupancy = top_k_occupancy(corrected)
        row = f"| {name} | {100 * occupancy.min():.0f}..{100 * occupancy.max():.0f}% | " + " | ".join(f"{100 * occupancy[word]:.0f}%" for word in frequent)

        #the reference rows (zscore, balance_half) are reported, not written
        if name not in args.variants:
            lines.append(row + " | (hubnorm_holdout) |")
            continue
        out_path = in_path.with_name(f"{in_path.stem}_{name}.csv")
        if out_path.exists() and not args.force:
            raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
        full = standardised.copy()
        full[sentence] = corrected

        #⚠️ the lone rows' rankings must come through untouched
        if not np.array_equal(np.argsort(-full[~sentence], axis=1, kind="stable"), np.argsort(-standardised[~sentence], axis=1, kind="stable")):
            raise SystemExit("A lone row's ranking changed; only sentence rows may be corrected.")
        fused = softmax_rows(zscore_scores(full))
        moses_out = moses / moses.sum(axis=1, keepdims=True)
        if not np.allclose(fused.sum(axis=1), 1.0, atol=1e-6) or not np.isfinite(fused).all():
            raise SystemExit(f"{name}: rows are not distributions.")
        write_submission(str(out_path), indices=index, primary_probs=fused, secondary_probs=moses_out)
        lines.append(row + f" | `{out_path.name}` |")

    #the doc's local check: drop what loses
    if args.check > 0:
        local_check(args, calibration, shares, lines)
    report = RUNS_DIR / f"trial23_correct_{in_path.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[correct] wrote the variants beside {in_path} and {report}")


if __name__ == "__main__":
    main()
