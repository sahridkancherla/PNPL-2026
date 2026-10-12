"""Transduce every member over the holdout's sentence rows, then mix: change 2 of the Route to 73.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.transduce \\
        --inputs submissions/trial13_g1_s0_in_context_ts.csv ... \\
        --runs trial13_g1_s0 ... --pull 1.0 --out submissions/ens_t22_tc1_ts.csv --holdout-dir data/holdout

Per member (one CSV each, `--runs` naming each CSV's run in the same order):

    features   v1: the member's 50 standardised word scores per row (the CSVs
               carry no embeddings; `cli/embed.py` adds them for v2)
    yhat       softmax(features / tau), tau fitted by NLL on the run's
               VALIDATION dump (labels: allowed), else the label-free
               confidence rule
    prototypes each word's mean feature on the validation dump (3+ tokens),
               else the rows' own yhat-weighted mean (TransCLIP's default)
    transduce  `transduction.transduce` on the 868 sentence rows: the KL term
               weighted `--pull` (1 = TransCLIP; larger stays closer), the
               cross-sentence graph, floored word shares
    blend      `--blend` of the transduced log-posterior, the estimated share
               divided out, into the features

then the members are averaged as cli.ensemble averages CSVs. Lone rows are
the plain mix (lone_fill replaces them anyway). Run `hubnorm_holdout` and
`lone_fill` on the result as on any mix -- the doc submits both with and
without the hub correction.

The report: the doc's label-free checks -- before, whether rows sharing a
top-1 word sit closer than other cross-sentence pairs; after, whether each
family's seeds agree more (A0's damage showed as less), and whether the
estimated shares put function words high -- plus `--check` draws: the same
procedure on the members' reference TEST dumps, non-Sherlock, resampled to
the holdout's estimated word mix, against the plain mix. Informational only.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from ...config import RUNS_DIR
from ..local_checks import bacc, eligible_tokens, ensemble, member_dumps, summarise_gains, uneven_draws
from ..transduction import (
    blend,
    fit_temperature_confidence,
    fit_temperature_nll,
    same_top1_separation,
    softmax_rows,
    transduce,
    zscore_rows,
)
from .ensemble import read_submission, standardise_rows, zscore_scores

#a CSV's run family: its name without the seed and the mode
FAMILY_PATTERN = re.compile(r"^(.*?)_s\d+(_|$)")

#a word needs this many validation tokens for a prototype of its own
MIN_PROTOTYPE_TOKENS = 3


def parse_args():
    """Command-line arguments for transduction."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="the members' holdout CSVs")
    parser.add_argument("--runs", nargs="+", default=None, help="each CSV's run (name or directory), same order: dumps")
    parser.add_argument("--protocol", default="reference", help="the dumps read for tau, prototypes and the check")
    parser.add_argument("--pull", type=float, default=1.0, help="the KL term's weight: 1 is TransCLIP's, larger stays closer")
    parser.add_argument("--blend", type=float, default=0.5, help="share of the transduced scores in each member")
    parser.add_argument("--neighbours", type=int, default=3, help="graph neighbours per row, other sentences only")
    parser.add_argument("--share-floor", type=float, default=0.005, help="least share a word may take (uniform 0.02)")
    parser.add_argument("--check", type=int, default=20, help="local draws (0 skips the check)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--track", default="deep")
    parser.add_argument("--report", default=None, help="default runs/trial22_transduce_<stem>.md")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def family_of(path):
    """A CSV's family: its file name without seed and mode.

    ARGS:
        path (str)

    RETURNS:
        str
    """
    name = Path(path).name
    found = FAMILY_PATTERN.match(name)
    return found.group(1) if found else Path(path).stem


def prototypes_from(dump):
    """Each word's mean standardised score row on a labelled dump.

    ARGS:
        dump (dict): from `local_checks.read_dump`

    RETURNS:
        [50, 50] float64, NaN rows for words with too few tokens
    """
    scores = zscore_rows(dump["scores"])
    keep = (dump["primary"] >= 0) & ~dump["isolated"]
    prototypes = np.full((50, 50), np.nan)
    for word in range(50):
        #a word's tokens on the dump, if enough exist
        rows = keep & (dump["primary"] == word)
        if rows.sum() >= MIN_PROTOTYPE_TOKENS:
            prototypes[word] = scores[rows].mean(axis=0)
    return prototypes


def member_setup(validation):
    """A member's temperature and prototypes from its validation dump.

    ARGS:
        validation (dict or None): from `member_dumps`

    RETURNS:
        temperature (float or None), prototypes ([50, 50] or None), how (str)
    """
    if validation is None:
        return None, None, "no validation dump: confidence-rule tau, TransCLIP's own starting means"
    keep = (validation["primary"] >= 0) & ~validation["isolated"]
    temperature = fit_temperature_nll(zscore_rows(validation["scores"][keep]), validation["primary"][keep])
    prototypes = prototypes_from(validation)
    return temperature, prototypes, f"validation tau {temperature:.3f}, {int(np.isfinite(prototypes).all(axis=1).sum())} prototypes"


def run_member(features, groups, temperature, prototypes, args):
    """One member's transduced, blended scores on a set of rows.

    ARGS:
        features (np.ndarray): [N, 50] standardised scores
        groups (np.ndarray): [N] sentence ids
        temperature (float or None): None uses the confidence rule
        prototypes (np.ndarray or None)
        args (argparse.Namespace)

    RETURNS:
        blended [N, 50], result dict from `transduce`
    """
    tau = temperature if temperature is not None else fit_temperature_confidence(features)
    zero_shot = softmax_rows(features / tau)
    result = transduce(features, zero_shot, groups, pull=args.pull, prototypes=prototypes,
                       n_neighbours=args.neighbours, share_floor=args.share_floor)
    return blend(features, result["posterior"], result["shares"], args.blend), result


def seed_agreement(tables_by_family):
    """Each family's mean pairwise top-10 overlap between seeds.

    ARGS:
        tables_by_family (dict): family -> list of [N, 50]

    RETURNS:
        dict: family -> overlap in percent (families of one seed left out)
    """
    agreement = {}
    for family, tables in tables_by_family.items():
        if len(tables) < 2:
            continue
        #axis=1: each row's ten best words
        tops = [np.argpartition(-table, 9, axis=1)[:, :10] for table in tables]
        pairs = [np.mean([len(set(a) & set(b)) / 10 for a, b in zip(first, second)]) for first, second in combinations(tops, 2)]
        agreement[family] = 100.0 * float(np.mean(pairs))
    return agreement


def local_check(args, runs, validations, setups, shares, lines):
    """The same procedure on the members' test dumps, uneven draws, against the plain mix.

    ARGS:
        args (argparse.Namespace)
        runs (list): run names
        validations (list): validation dumps, for the setups
        setups (list): (temperature, prototypes) per member
        shares (np.ndarray): [50] the holdout's estimated shares
        lines (list): report lines, appended to
    """
    tests = member_dumps(runs, args.protocol, "test")
    if any(test is None for test in tests):
        lines.append("Local check skipped: a member has no test dump.")
        return
    labels, groups = tests[0]["primary"], tests[0]["sequence"]
    if groups is None:
        groups = np.arange(len(labels))
    eligible = eligible_tokens(tests[0])
    gains = []
    for draw in uneven_draws(labels, eligible, shares, n_draws=args.check):
        #a resampled token keeps its sentence: its copies never neighbour each other
        draw_groups = groups[draw]
        plain = ensemble([test["scores"][draw] for test in tests])
        transduced = [run_member(zscore_rows(test["scores"][draw]), draw_groups, temperature, prototypes, args)[0]
                      for test, (temperature, prototypes) in zip(tests, setups)]
        gains.append(bacc(ensemble(transduced), labels[draw]) - bacc(plain, labels[draw]))
    lines.append(f"Local check ({args.protocol} test dumps, non-Sherlock, {len(gains)} uneven draws of 868): "
                 f"transduced mix minus plain mix {summarise_gains(gains)}.")


def main():
    """Transduce each member, mix, write the CSV and the report."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    if args.runs is not None and len(args.runs) != len(args.inputs):
        raise SystemExit(f"{len(args.inputs)} CSVs but {len(args.runs)} runs: name one run per CSV.")
    words = (load_vocabulary("primary"), load_vocabulary("moses"))

    #the holdout's rows: which are sentences, and which sentence each belongs to
    from pnpl.competition import LibriBrainCompetitionHoldout

    metadata = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir).metadata
    sentence = np.array([meta["source"] == "sentence" for meta in metadata])
    groups = np.array([meta["epoch"] if meta["source"] == "sentence" else -1 - meta["epoch"] for meta in metadata])

    #the members' dumps, when runs are named
    validations = member_dumps(args.runs, args.protocol, "validation") if args.runs else [None] * len(args.inputs)
    lines = [f"# Trial 22 -- transduction v1 (scores): `{out_path.name}`", "",
             f"{len(args.inputs)} members; pull {args.pull:g}, blend {args.blend:g}, {args.neighbours} cross-sentence "
             f"neighbours, share floor {args.share_floor:g}; {int(sentence.sum())} sentence rows.", ""]

    before, after, moses_parts, setups, all_shares = [], [], [], [], []
    by_family_before, by_family_after = defaultdict(list), defaultdict(list)
    separations = []
    for path, validation in zip(args.inputs, validations):
        index, primary, moses = read_submission(Path(path), *words)
        if not np.array_equal(index, np.arange(len(metadata))):
            raise SystemExit(f"{path}: its rows are not the holdout's 0..{len(metadata) - 1}.")
        features = standardise_rows(primary)
        temperature, prototypes, how = member_setup(validation)
        setups.append((temperature, prototypes))

        #the sentence rows transduced; the lone rows as they are
        blended, result = run_member(features[sentence], groups[sentence], temperature, prototypes, args)
        transduced = features.copy()
        transduced[sentence] = blended
        before.append(features)
        after.append(transduced)
        moses_parts.append(standardise_rows(moses))
        all_shares.append(result["shares"])
        by_family_before[family_of(path)].append(features[sentence])
        by_family_after[family_of(path)].append(blended)
        separations.append(same_top1_separation(features[sentence], features[sentence], groups[sentence]))
        lines.append(f"- `{Path(path).name}`: {how}; likelihood temperature {result['temperature']:.3g}")

    #the mix, as cli.ensemble builds it, before and after
    plain, mixed = ensemble(before), ensemble(after)
    shares = np.mean(all_shares, axis=0)

    #the doc's label-free checks
    same, other = np.nanmean([pair[0] for pair in separations]), np.mean([pair[1] for pair in separations])
    agree_before, agree_after = seed_agreement(by_family_before), seed_agreement(by_family_after)
    order = np.argsort(-shares)
    vocabulary = list(words[0])
    occupancy = lambda scores: np.bincount(np.argpartition(-scores, 9, axis=1)[:, :10].ravel(), minlength=50) / len(scores)
    lines += ["", "## Label-free checks", "",
              f"- before: rows sharing a top-1 word, cosine {same:.3f}; other cross-sentence pairs {other:.3f} "
              f"({'clustered' if same > other else 'NOT clustered'})",
              "- seeds agreeing (top-10 overlap), before -> after: "
              + "; ".join(f"{family} {agree_before[family]:.1f} -> {agree_after[family]:.1f}" for family in agree_before),
              "- estimated shares, highest: " + ", ".join(f"{vocabulary[word]} {100 * shares[word]:.1f}%" for word in order[:8]),
              "- estimated shares, lowest: " + ", ".join(f"{vocabulary[word]} {100 * shares[word]:.1f}%" for word in order[-5:]),
              f"- top-10 shares, smallest..largest: before {100 * occupancy(plain[sentence]).min():.0f}..{100 * occupancy(plain[sentence]).max():.0f}%, "
              f"after {100 * occupancy(mixed[sentence]).min():.0f}..{100 * occupancy(mixed[sentence]).max():.0f}%",
              f"- top-10 overlap with the plain mix: "
              f"{100 * np.mean([len(set(a) & set(b)) / 10 for a, b in zip(np.argpartition(-plain[sentence], 9, axis=1)[:, :10], np.argpartition(-mixed[sentence], 9, axis=1)[:, :10])]):.1f}%"]

    #the informational local check
    if args.runs and args.check > 0:
        lines += [""]
        local_check(args, args.runs, validations, setups, shares, lines)

    #the CSV: z-scored before the softmax, so six decimals keep every rank
    primary_out = softmax_rows(zscore_scores(mixed))
    moses_out = softmax_rows(zscore_scores(np.mean(moses_parts, axis=0)))
    for name, values in (("primary", primary_out), ("moses", moses_out)):
        sums = values.sum(axis=1)
        if not np.isfinite(values).all() or not np.allclose(sums, 1.0, atol=1e-6):
            raise SystemExit(f"{name} rows are not distributions.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=np.arange(len(metadata)), primary_probs=primary_out, secondary_probs=moses_out)
    report = Path(args.report) if args.report else RUNS_DIR / f"trial22_transduce_{out_path.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[transduce] wrote {out_path} and {report}")


if __name__ == "__main__":
    main()
