"""Correct a submission's hub words with adjustments fitted on the holdout's own rows.

    PYTHONPATH=src python -m pnpl2026.trial17.cli.hubnorm_holdout \\
        --csv submissions/ens_g1g2kn_f1w1024_ts.csv --holdout-dir data/holdout \\
        --check-runs runs/trial14_calibration/trial13_g1_s0 ...  [--check-protocol chunked] \\
        [--method auto]

`cli.hubnorm`'s correction fitted TRANSDUCTIVELY: the bank is the holdout's
own sentence rows instead of a validation partition. Allowed since the
holdout principle was withdrawn (CLAUDE.md, 2026-10-08). Two reasons it can
beat the frozen version: the holdout's sources are not ours, so its hubs may
be other words; and it needs no dumps, so it corrects any mix.

    1. the CSV's sentence rows, z-scored as cli.ensemble reads them, are the
       bank: zscore (column normalisation) and balance (offsets giving every
       word the fair top-10 share, in full and at half strength), labels
       never seen
    2. the local check, with --check-runs: the same procedure on local test
       run words -- fitted on those very rows, scored on them -- against the
       uncorrected scores and the frozen validation-bank fit. It checks the
       PROCEDURE, so the runs need not be the CSV's members
    3. the rule (`--method auto`): the variant with the largest local
       transductive gain, refused when none gains
    4. sentence rows corrected; lone rows left as they are (92 rows are too
       few to fit; cli.lone_fill fills them afterwards). Written as
       `<stem>_hubh.csv`, checked as cli.predict checks; Moses copied

Writes runs/trial17_hubnorm_holdout.md. Light: seconds on a login node.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT: rows keep the input's index column and
order, and each row's source comes from the holdout loader's own metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...config import RUNS_DIR
from ..hubnorm import apply_correction, fit_balance, fit_zscore
from .ensemble import describe, holdout_sources, read_submission, softmax_rows, standardise_rows, zscore_scores
from .hubnorm import bacc, ensemble_partition, share_range

#the variants compared, in the order they are reported
VARIANTS = ("zscore", "balance", "balance_half")


def parse_args():
    """Command-line arguments for the holdout-fitted hub correction."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="the submission CSV to correct")
    parser.add_argument("--check-runs", nargs="+", default=None, help="directories holding dumps/, for the local check")
    parser.add_argument("--check-protocol", default="chunked", help="the dumps the local check reads")
    parser.add_argument(
        "--method",
        default="auto",
        choices=("auto",) + VARIANTS,
        help="auto: the best local transductive gain (needs --check-runs)",
    )
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--track", default="deep", help="competition track")
    parser.add_argument("--out", default=None, help="default <csv stem>_hubh.csv beside the input")
    parser.add_argument("--report", default=None, help="default runs/trial17_hubnorm_holdout.md")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    return parser.parse_args()


def fit_variants(bank):
    """Every variant fitted on one bank of standardised scores.

    ARGS:
        bank (np.ndarray): [n_bank, 50]

    RETURNS:
        dict: variant -> (correction, strength), for `apply_correction`
    """
    zscore, balance = fit_zscore(bank), fit_balance(bank)
    return {"zscore": (zscore, 1.0), "balance": (balance, 1.0), "balance_half": (balance, 0.5)}


def local_check(runs, protocol, lines):
    """The transductive procedure on local test run words, beside the frozen one.

    ARGS:
        runs (list): run directories holding dumps/
        protocol (str): whose validation and test dumps are read
        lines (list): report lines, appended to

    RETURNS:
        dict: variant -> transductive BAcc@10 gain on the test run words
    """
    val_scores, val_primary, val_isolated = ensemble_partition(runs, protocol, "validation")
    test_scores, test_primary, test_isolated = ensemble_partition(runs, protocol, "test")

    #run words only: the public board scores nothing else
    val_run = (val_primary >= 0) & ~val_isolated
    test_run = (test_primary >= 0) & ~test_isolated
    scores, targets = test_scores[test_run], test_primary[test_run]
    before = bacc(scores, targets)
    lines.append(f"| none | - | {len(targets)} | {before:.2f} | - | {share_range(scores)} |")

    #transductive: each fit made on the very rows it then corrects
    gains = {}
    for name, (correction, strength) in fit_variants(scores).items():
        corrected = apply_correction(scores, correction, strength)
        after = bacc(corrected, targets)
        gains[name] = after - before
        lines.append(f"| {name} | the test rows themselves | {len(targets)} | {after:.2f} | {gains[name]:+.2f} | "
                     f"{share_range(corrected)} |")

    #frozen, as cli.hubnorm fits it: validation rows, then test corrected
    for name, (correction, strength) in fit_variants(val_scores[val_run]).items():
        corrected = apply_correction(scores, correction, strength)
        after = bacc(corrected, targets)
        lines.append(f"| {name} | validation (frozen) | {len(targets)} | {after:.2f} | {after - before:+.2f} | "
                     f"{share_range(corrected)} |")
    return gains


def main():
    """Check the procedure locally, fit on the holdout's sentence rows, correct them."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_hubh.csv")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    if args.method == "auto" and not args.check_runs:
        raise SystemExit("--method auto chooses by the local check: pass --check-runs, or name a method.")

    lines = [
        "# Trial 17 -- hub correction fitted on the holdout's own rows",
        "",
        f"CSV `{in_path.name}`. The fit's bank is the CSV's sentence rows (labels never seen).",
    ]

    #the local check of the procedure, when dumps are given
    gains = {}
    if args.check_runs:
        lines += [
            f"Local check on {len(args.check_runs)} runs' `{args.check_protocol}` test run words: "
            + ", ".join(Path(run).name for run in args.check_runs) + ".",
            "BAcc@10 in percent, chance 20; 'top-10 shares' is the smallest and largest share of rows any",
            "word reaches the top 10 of (fair: 20%).",
            "",
            "| fit | fitted on | test words | BAcc@10 | gain | top-10 shares |",
            "|---|---|---|---|---|---|",
        ]
        gains = local_check(args.check_runs, args.check_protocol, lines)

    #the rule: the best local transductive gain, refused when none gains
    choice = args.method
    if choice == "auto":
        choice = max(gains, key=lambda name: gains[name])
        if gains[choice] <= 0:
            print("\n".join(lines))
            raise SystemExit("[hubnorm_holdout] no fit gains on the local run words; nothing written.")
    gain_text = f", {gains[choice]:+.2f} on the local test run words" if choice in gains else ", chosen by hand"
    lines += ["", "## Chosen", "", f"- sentence rows: **{choice}**{gain_text}", "- lone rows: left as they are"]

    #the CSV: every row z-scored as cli.ensemble reads it
    primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
    index, primary, moses = read_submission(in_path, primary_words, moses_words)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{in_path}: the index column is not 0..{len(index) - 1} in order.")
    sources = holdout_sources(args.track, args.holdout_dir, len(index))
    sentence = sources == "sentence"
    standardised = standardise_rows(primary)

    #⚠️ fitted on ALL the holdout's sentence rows at once: the transductive step
    correction, strength = fit_variants(standardised[sentence])[choice]
    corrected = standardised.copy()
    corrected[sentence] = apply_correction(standardised[sentence], correction, strength)
    lines.append(f"- holdout sentence rows ({int(sentence.sum())}): top-10 shares "
                 f"{share_range(standardised[sentence])} before, {share_range(corrected[sentence])} after")

    #z-scored before the softmax, so a CSV's 6 decimals keep every rank readable
    fused_primary = softmax_rows(zscore_scores(corrected))
    moses = moses / moses.sum(axis=1, keepdims=True)
    print(describe("input", primary, sources))
    print(describe("corrected", fused_primary, sources))

    #⚠️ DO NOT REMOVE: the same checks predict makes before writing
    for name, values in (("primary", fused_primary), ("moses", moses)):
        sums = values.sum(axis=1)
        if not np.isfinite(values).all() or not np.allclose(sums, 1.0, atol=1e-6):
            raise SystemExit(f"{name} rows are not distributions (sums {sums.min():.6f}..{sums.max():.6f}).")
    if fused_primary.shape != primary.shape:
        raise SystemExit(f"Corrected shape {fused_primary.shape} differs from the input's {primary.shape}.")

    #the lone rows' rankings must come through untouched
    alone = ~sentence
    #stable sorts, so tied words keep one order on both sides
    ranked_after = np.argsort(-fused_primary[alone], axis=1, kind="stable")
    if not np.array_equal(ranked_after, np.argsort(-primary[alone], axis=1, kind="stable")):
        raise SystemExit("A lone row's ranking changed; only sentence rows may be corrected.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=index, primary_probs=fused_primary, secondary_probs=moses)
    lines += ["", f"Wrote `{out_path.name}`."]
    report = Path(args.report) if args.report else RUNS_DIR / "trial17_hubnorm_holdout.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[hubnorm_holdout] wrote {out_path} and {report}")
    print("[hubnorm_holdout] submission is MANUAL, through the competition web UI.")


if __name__ == "__main__":
    main()
