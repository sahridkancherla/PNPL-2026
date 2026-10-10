"""Correct a submission's hub words with frozen per-word adjustments from a validation bank.

    PYTHONPATH=src python -m pnpl2026.trial19.cli.hubnorm \\
        --csv submissions/ens_g1g2_f1w1024_ts.csv \\
        --runs runs/trial14_calibration/trial13_g1_s0 ...   (the CSV's members) \\
        [--bank-protocol chunked] [--method auto] --holdout-dir data/holdout

What it does (see `hubnorm.py` for why):

    1. the members' per-word dumps under `--bank-protocol`, validation and
       test, are ensembled exactly as `cli.ensemble` averages CSVs: each run's
       rows z-scored, averaged, z-scored again
    2. two banks from VALIDATION, split by each dump row's isolated flag:
       scored words inside runs, and scored words served alone. Each is fitted
       three ways -- zscore, balance, balance at half strength -- labels never
       used
    3. a local check on TEST: BAcc@10 before and after every fit, run words
       and lone words apart, and how evenly the words share the top 10
    4. the rule, fixed in advance (`--method auto`): runs take the fit with the
       largest test gain on run words, refused when none gains, warned when
       the gain is under 0.5; lone words -- scored on the PRIVATE board only --
       take a fit only if it gains more than 1.0 on lone words (fewer words,
       more noise), else stay as they are
    5. the chosen fits applied to the CSV, each holdout row on its own:
       sentence rows with the run fit, lone rows with the lone fit. Written as
       `<stem>_hub.csv`, checked as cli.predict checks; Moses copied

Writes runs/trial19_hubnorm.md. Light: numpy over saved arrays and one CSV,
seconds on a login node.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT: rows keep the input's index column and
order, and each row's source comes from the holdout loader's own metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ...config import RUNS_DIR
from ...metrics import balanced_topk_accuracy
from ..hubnorm import K, apply_correction, fit_balance, fit_zscore, top_k_occupancy
from .ensemble import describe, holdout_sources, read_submission, softmax_rows, standardise_rows, zscore_scores

#the dump key holding each token's 50 primary scores; trial 13 wrote "retrieval"
SCORE_KEYS = ("primary_logits", "retrieval")

#the 50 scored words
N_WORDS = 50

#a lone-word bank smaller than this is not fitted: too few rows per word
MIN_LONE_BANK = 100

#the rule's thresholds, in BAcc@10 points on the local test words
RUN_WARN_GAIN = 0.5
LONE_MIN_GAIN = 1.0

#the fits compared: name -> (fit kind, strength)
VARIANTS = {"zscore": ("zscore", 1.0), "balance": ("balance", 1.0), "balance_half": ("balance", 0.5)}


def parse_args():
    """Command-line arguments for the hub correction."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="the submission CSV to correct")
    parser.add_argument("--runs", nargs="+", required=True, help="the CSV's members: directories holding dumps/")
    parser.add_argument("--bank-protocol", default="chunked", help="the dumps the bank and the check come from")
    parser.add_argument(
        "--method",
        default="auto",
        choices=("auto", "none") + tuple(VARIANTS),
        help="the run-word fit; auto applies the rule above",
    )
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--track", default="deep", help="competition track")
    parser.add_argument("--out", default=None, help="default <csv stem>_hub.csv beside the input")
    parser.add_argument("--report", default=None, help="default runs/trial19_hubnorm.md")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
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


def read_dump(run_dir, protocol, partition):
    """One run's dump for one protocol and partition.

    ARGS:
        run_dir (str): holds dumps/
        protocol (str): e.g. "chunked"
        partition (str): "validation" or "test"

    RETURNS:
        dict: scores [n_tokens, 50], primary [n_tokens], isolated [n_tokens] bool
    """
    path = Path(run_dir) / "dumps" / f"{protocol}_{partition}.npz"
    with np.load(path) as stored:
        key = next((name for name in SCORE_KEYS if name in stored.files), None)
        if key is None:
            raise SystemExit(f"{path}: none of {SCORE_KEYS} in {stored.files}")
        return {
            "scores": stored[key].astype(np.float64),
            "primary": stored["primary"].astype(np.int64),
            "isolated": stored["isolated"].astype(bool),
        }


def ensemble_partition(runs, protocol, partition):
    """The members' ensemble on one partition, as cli.ensemble builds a CSV.

    ARGS:
        runs (list): run directories
        protocol (str)
        partition (str)

    RETURNS:
        standardised [n_tokens, 50], primary [n_tokens], isolated [n_tokens]
    """
    dumps = [read_dump(run, protocol, partition) for run in runs]

    #⚠️ the same tokens in the same order, or the average mixes words
    first = dumps[0]
    for run, dump in zip(runs, dumps):
        if not (np.array_equal(dump["primary"], first["primary"]) and np.array_equal(dump["isolated"], first["isolated"])):
            raise SystemExit(f"{run}: its {protocol} {partition} dump lists other tokens than {runs[0]}'s.")

    #axis 0 is the run axis; z-scored again, as standardise_rows reads a CSV
    average = np.mean([zscore_rows(dump["scores"]) for dump in dumps], axis=0)
    return zscore_rows(average), first["primary"], first["isolated"]


def bacc(scores, targets):
    """BAcc@10 over the 50 words, in percent.

    ARGS:
        scores (np.ndarray): [n, 50]
        targets (np.ndarray): [n]

    RETURNS:
        float, nan without rows
    """
    if len(targets) == 0:
        return float("nan")
    return 100.0 * balanced_topk_accuracy(torch.from_numpy(scores), torch.from_numpy(targets), K, N_WORDS)


def share_range(scores):
    """The smallest and largest top-10 share any word gets, as text.

    ARGS:
        scores (np.ndarray): [n, 50]

    RETURNS:
        str
    """
    occupancy = top_k_occupancy(scores)
    return f"{100 * occupancy.min():.0f}-{100 * occupancy.max():.0f}%"


def check_group(label, bank, test_scores, test_targets, lines):
    """Fit every variant on one bank and score it on the matching test words.

    ARGS:
        label (str): "run words" or "lone words"
        bank (np.ndarray): [n_bank, 50] validation scores
        test_scores (np.ndarray): [n_test, 50]
        test_targets (np.ndarray): [n_test]
        lines (list): report lines, appended to

    RETURNS:
        fits : dict, variant -> (correction, strength)
        gains : dict, variant -> test BAcc@10 gain in points
    """
    zscore, balance = fit_zscore(bank), fit_balance(bank)
    fits = {"zscore": (zscore, 1.0), "balance": (balance, 1.0), "balance_half": (balance, 0.5)}

    #the uncorrected baseline, then each fit, on the same test words
    before = bacc(test_scores, test_targets)
    lines.append(f"| {label} | none | {len(bank)} | {len(test_targets)} | {before:.2f} | - | {share_range(test_scores)} |")
    gains = {}
    for name, (correction, strength) in fits.items():
        corrected = apply_correction(test_scores, correction, strength)
        after = bacc(corrected, test_targets)
        gains[name] = after - before
        lines.append(f"| {label} | {name} | {len(bank)} | {len(test_targets)} | {after:.2f} | {gains[name]:+.2f} | "
                     f"{share_range(corrected)} |")
    lines.append(f"| {label} | (balance left a largest share error of {100 * balance['share_error']:.1f} points on the bank) | | | | | |")
    return fits, gains


def main():
    """Ensemble the dumps, fit, check, choose by the rule, correct the CSV."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_hub.csv")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")

    #validation is the bank; test is the check; both split runs from lone words
    val_scores, val_primary, val_isolated = ensemble_partition(args.runs, args.bank_protocol, "validation")
    test_scores, test_primary, test_isolated = ensemble_partition(args.runs, args.bank_protocol, "test")
    val_run, val_lone = (val_primary >= 0) & ~val_isolated, (val_primary >= 0) & val_isolated
    test_run, test_lone = (test_primary >= 0) & ~test_isolated, (test_primary >= 0) & test_isolated

    lines = [
        "# Trial 17 -- hub correction (querybank normalisation), checked locally",
        "",
        f"CSV `{in_path.name}`; bank and check from {len(args.runs)} runs' `{args.bank_protocol}` dumps: "
        + ", ".join(Path(run).name for run in args.runs) + ".",
        "Fitted on validation (labels never used), scored on test. BAcc@10 in percent, chance 20;",
        "'top-10 shares' is the smallest and largest share of test rows any word reaches the top 10 of (fair: 20%).",
        "",
        "| words | fit | bank rows | test words | BAcc@10 | gain | top-10 shares |",
        "|---|---|---|---|---|---|---|",
    ]
    run_fits, run_gains = check_group("run words", val_scores[val_run], test_scores[test_run], test_primary[test_run], lines)
    lone_fits, lone_gains = {}, {}
    if val_lone.sum() >= MIN_LONE_BANK:
        lone_fits, lone_gains = check_group("lone words", val_scores[val_lone], test_scores[test_lone],
                                            test_primary[test_lone], lines)
    else:
        lines.append(f"| lone words | not fitted: {int(val_lone.sum())} bank rows, fewer than {MIN_LONE_BANK} | | | | | |")

    #the rule: runs by their own test gain, lone words only past a higher bar
    run_choice = args.method
    if run_choice == "auto":
        run_choice = max(run_gains, key=lambda name: run_gains[name])
        if run_gains[run_choice] <= 0:
            print("\n".join(lines))
            raise SystemExit("[hubnorm] no fit gains on the local run words; nothing written (pass --method to force one).")
    lone_choice = None
    if lone_gains:
        best = max(lone_gains, key=lambda name: lone_gains[name])
        lone_choice = best if lone_gains[best] > LONE_MIN_GAIN else None

    lines += ["", "## Chosen", ""]
    if run_choice == "none":
        lines.append("- run words: none")
    else:
        warn = " (under 0.5 points: little to expect)" if run_gains[run_choice] < RUN_WARN_GAIN else ""
        lines.append(f"- run words: **{run_choice}**, {run_gains[run_choice]:+.2f} on the local test run words{warn}")
    lines.append(f"- lone words: **{lone_choice}**, {lone_gains[lone_choice]:+.2f} on the local test lone words"
                 if lone_choice else "- lone words: left as they are (no fit gained more than 1.0 locally)")

    #the CSV: every row z-scored as cli.ensemble reads it, then its own group's fit
    primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
    index, primary, moses = read_submission(in_path, primary_words, moses_words)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{in_path}: the index column is not 0..{len(index) - 1} in order.")
    sources = holdout_sources(args.track, args.holdout_dir, len(index))
    sentence, alone = sources == "sentence", sources == "word"
    standardised = standardise_rows(primary)
    corrected = standardised.copy()
    if run_choice != "none":
        correction, strength = run_fits[run_choice]
        corrected[sentence] = apply_correction(standardised[sentence], correction, strength)
    if lone_choice is not None:
        correction, strength = lone_fits[lone_choice]
        corrected[alone] = apply_correction(standardised[alone], correction, strength)

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

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=index, primary_probs=fused_primary, secondary_probs=moses)
    lines += ["", f"Wrote `{out_path.name}`."]
    report = Path(args.report) if args.report else RUNS_DIR / "trial19_hubnorm.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[hubnorm] wrote {out_path} and {report}")
    print("[hubnorm] submission is MANUAL, through the competition web UI.")


if __name__ == "__main__":
    main()
