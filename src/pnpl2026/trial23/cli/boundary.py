"""P6 of the 13 Oct research note: is a sentence's edge harder, and does a specialist fix it?

    # 1. the diagnosis: BAcc@10 by position, MOCHA and TheMoth chains38 rows
    PYTHONPATH=src python -m pnpl2026.trial23.cli.boundary diagnose --members trial13_g1_s0 ...

    # 2. the specialists on the holdout's first and last rows only
    PYTHONPATH=src python -m pnpl2026.trial23.cli.boundary apply --members ... \\
        --specialists trial23_p6_g1_s0 ... --base submissions/ens_t22_tc1b75_ts.csv

`diagnose` scores the plain and the transduced mix (the 70.68 file's recipe:
v1, pull 1, blend 0.75, setup from the other partition) on every chains38
row of MOCHA-TIMIT and TheMoth -- subject 0's clean short-sentence and
podcast sessions; TIMIT's are all in training -- for first, middle and last
words apart, validation and test pooled, with bootstrap intervals over
chains. On the holdout, where there are no labels, it compares how well each
family's seeds agree on first, middle and last rows. The note: no gap, no P6.

`apply` chooses the specialists' weight on MOCHA VALIDATION boundary rows --
z(base) + w * z(specialist mix), the base being the transduced mix there as
on the holdout -- then writes `<base stem>_p6.csv`: the holdout's first and
last rows blended at that weight, every middle row exactly the base. Then
hubnorm_holdout and lone_fill, as for every candidate.
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...config import RUNS_DIR
from .. import crossfeatures as cf
from ..corrections import position_groups
from ..hubnorm import apply_correction, fit_zscore
from ..local_checks import bacc, ensemble, member_dumps
from .ensemble import read_submission, softmax_rows, standardise_rows, zscore_scores
from .knn_vote import load as load_dump
from .transduce import member_setup, run_member

#the clean short-sentence and podcast tasks of subject 0's held-out sessions
TASKS = ("MOCHATIMIT", "TheMoth")

#the partner partition each partition's transduction setup comes from
OTHER = {"validation": "test", "test": "validation"}

#position groups, as `corrections.position_groups` numbers them
GROUP_NAMES = {0: "first", 1: "middle", 2: "last"}

#the specialists' weight grid, fixed in advance
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)

#bootstrap draws over chains
DRAWS = 1000

#a CSV's family: its name without the seed
FAMILY_PATTERN = re.compile(r"^(.*?)_s\d+(_|$)")


def parse_args():
    """Command-line arguments: a mode, then its options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("diagnose", "apply"))
    parser.add_argument("--members", nargs="+", required=True, help="the base mix's members, each with chains38 dumps")
    parser.add_argument("--specialists", nargs="+", default=None, help="apply: the P6 specialists, with chains38 dumps and holdout CSVs")
    parser.add_argument("--base", default=None, help="apply: the holdout base mix before the hub correction")
    parser.add_argument("--submissions", default="submissions", help="where the members' and specialists' CSVs are")
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--report", default=None, help="default runs/trial23_boundary_diagnose.md or _apply_<base stem>.md")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def chain_rows(runs, partition, tasks=TASKS, suffix=""):
    """Every member's chains38 scores on one partition's rows of the tasks given.

    ARGS:
        runs (list): run names
        partition (str): "validation" or "test"
        tasks (tuple or None): the tasks kept; None keeps every task
        suffix (str): the dumps' name suffix -- "_rc" for P8b's re-centred ones

    RETURNS:
        rows (dict: labels, groups, positions, task; and onset, table_index
        when the dumps have them), scores (list of [n, 50] standardised)
    """
    dumps = [load_dump(run, f"chains38_{partition}{suffix}") for run in runs]
    for dump in dumps[1:]:
        #⚠️ the same tokens in the same order, or the mix mixes words
        if not np.array_equal(dump["primary"], dumps[0]["primary"]):
            raise SystemExit("The members' chains38 dumps list different tokens; re-run cli.embed.")
    task = dumps[0]["task"].astype(str)
    keep = (np.isin(task, tasks) if tasks is not None else np.ones(len(task), dtype=bool)) & (dumps[0]["primary"] >= 0)
    groups = dumps[0]["sequence"][keep].astype(np.int64)
    rows = {"labels": dumps[0]["primary"][keep].astype(np.int64), "groups": groups,
            "positions": position_groups(groups), "task": task[keep]}

    #trial 23's dump fields: P5 reads gaps from onsets and skips from table rows
    for name in ("onset", "table_index"):
        if name in dumps[0]:
            rows[name] = dumps[0][name][keep].astype(np.int64)
    return rows, [cf.zscore_rows(dump["logits"][keep].astype(np.float64)) for dump in dumps]


def transduced_mix(runs, scores, groups, partition):
    """The 70.68 file's recipe on dump rows: v1 per member, setup from the partner partition.

    ARGS:
        runs (list), scores (list), groups (np.ndarray), partition (str)

    RETURNS:
        [n, 50] standardised, before the hub correction
    """
    setups = member_dumps(runs, "reference", OTHER[partition])
    settings = SimpleNamespace(pull=1.0, blend=0.75, neighbours=3, share_floor=0.005)
    transduced = []
    for member_scores, dump in zip(scores, setups):
        temperature, prototypes, _ = member_setup(dump)
        transduced.append(run_member(member_scores, member_scores, groups, temperature, prototypes, settings)[0])
    return ensemble(transduced)


def hubbed(scores):
    """The hub correction fitted on the rows themselves, as on the holdout.

    ARGS:
        scores (np.ndarray): [n, 50]

    RETURNS:
        [n, 50]
    """
    return apply_correction(scores, fit_zscore(scores))


def group_bacc(scores, rows, draws=DRAWS, seed=0):
    """BAcc@10 per position group, with bootstrap intervals over chains.

    ARGS:
        scores (np.ndarray): [n, 50], already corrected
        rows (dict)
        draws (int), seed (int)

    RETURNS:
        dict: group name -> (estimate, low, high); "gap" -> middle minus the
        boundary rows' estimate, with its interval
    """
    boundary = rows["positions"] != 1
    estimates = {name: bacc(scores[rows["positions"] == group], rows["labels"][rows["positions"] == group])
                 for group, name in GROUP_NAMES.items()}
    estimates["boundary"] = bacc(scores[boundary], rows["labels"][boundary])
    chains = np.unique(rows["groups"])
    members = {chain: np.flatnonzero(rows["groups"] == chain) for chain in chains}
    generator = np.random.default_rng(seed)
    samples = defaultdict(list)
    for _ in range(draws):
        #whole chains drawn with replacement, so neighbouring words stay together
        chosen = np.concatenate([members[chain] for chain in generator.choice(chains, len(chains), replace=True)])
        positions, labels = rows["positions"][chosen], rows["labels"][chosen]
        values = {name: bacc(scores[chosen][positions == group], labels[positions == group]) for group, name in GROUP_NAMES.items()}
        values["boundary"] = bacc(scores[chosen][positions != 1], labels[positions != 1])
        values["gap"] = values["middle"] - values["boundary"]
        for name, value in values.items():
            samples[name].append(value)
    estimates["gap"] = estimates["middle"] - estimates["boundary"]
    return {name: (estimates[name], *np.percentile(samples[name], [2.5, 97.5])) for name in estimates}


def pooled(runs):
    """MOCHA and TheMoth rows of validation and test, pooled, with both mixes.

    ARGS:
        runs (list)

    RETURNS:
        rows (dict), plain [n, 50], transduced [n, 50] -- hub-corrected per partition
    """
    parts = []
    for partition in ("validation", "test"):
        rows, scores = chain_rows(runs, partition)
        plain = hubbed(ensemble(scores))
        transduced = hubbed(transduced_mix(runs, scores, rows["groups"], partition))
        #groups made unique across the two partitions
        rows["groups"] = rows["groups"] + (0 if partition == "validation" else 10 ** 6)
        parts.append((rows, plain, transduced))
    rows = {name: np.concatenate([part[0][name] for part in parts]) for name in parts[0][0]}
    return rows, np.concatenate([part[1] for part in parts]), np.concatenate([part[2] for part in parts])


def holdout_agreement(runs, submissions, holdout_dir):
    """Each family's seed agreement (top-10 overlap) on the holdout's first, middle and last rows.

    ARGS:
        runs (list), submissions (str), holdout_dir (str or None)

    RETURNS:
        dict: family -> {group name: overlap in percent}
    """
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary

    metadata = LibriBrainCompetitionHoldout(track="deep", data_path=holdout_dir).metadata
    sentence = np.array([meta["source"] == "sentence" for meta in metadata])
    positions = position_groups(np.array([meta["epoch"] for meta in metadata])[sentence])
    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    families = defaultdict(list)
    for run in runs:
        #a run may be named or given as its directory: the CSV is named after the run
        name = Path(run).name
        found = FAMILY_PATTERN.match(name)
        table = standardise_rows(read_submission(Path(submissions) / f"{name}_in_context_ts.csv", *words)[1])[sentence]
        families[found.group(1) if found else name].append(np.argpartition(-table, 9, axis=1)[:, :10])
    agreement = {}
    for family, tops in families.items():
        if len(tops) < 2:
            continue
        overlap = np.mean([[len(set(a) & set(b)) / 10 for a, b in zip(first, second)] for first, second in combinations(tops, 2)], axis=0)
        agreement[family] = {name: 100.0 * float(overlap[positions == group].mean()) for group, name in GROUP_NAMES.items()}
    return agreement


def diagnose(args):
    """The boundary gap, locally with labels and on the holdout without."""
    rows, plain, transduced = pooled(args.members)
    lines = ["# Trial 23 -- P6 diagnosis: are a sentence's first and last words harder?", "",
             f"{len(args.members)} members; chains38 rows of {', '.join(TASKS)}, validation and test pooled: "
             f"{len(rows['labels'])} rows in {len(np.unique(rows['groups']))} chains. BAcc@10 after the hub correction, "
             f"95% bootstrap intervals over chains ({DRAWS} draws).", "",
             "| mix | first | middle | last | boundary (first + last) | middle minus boundary |", "|---|---|---|---|---|---|"]
    verdicts = []
    for name, scores in (("plain", plain), ("transduced (70.68 recipe)", transduced)):
        result = group_bacc(scores, rows)
        cells = [f"{result[group][0]:.2f} [{result[group][1]:.1f}, {result[group][2]:.1f}]" for group in ("first", "middle", "last", "boundary", "gap")]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
        verdicts.append(result["gap"][1] > 0)
    lines += ["", f"Verdict: {'a real boundary gap (the interval excludes 0): P6 goes ahead' if all(verdicts) else 'no clear boundary gap: the note says skip P6'}.", ""]

    #the holdout, without labels: do seeds agree less at the edges?
    agreement = holdout_agreement(args.members, args.submissions, args.holdout_dir)
    lines += ["Holdout, label-free: each family's seed agreement (top-10 overlap, %) by position:", "",
              "| family | first | middle | last |", "|---|---|---|---|"]
    lines += [f"| {family} | {values['first']:.1f} | {values['middle']:.1f} | {values['last']:.1f} |" for family, values in agreement.items()]
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_boundary_diagnose.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[boundary] wrote {report}")


def apply(args):
    """Choose the specialists' weight on MOCHA validation boundary rows; write the holdout file."""
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary, write_submission

    if not args.specialists or not args.base:
        raise SystemExit("apply needs --specialists and --base.")

    #the weight, on MOCHA validation's first and last rows
    rows, scores = chain_rows(args.members, "validation")
    base_validation = transduced_mix(args.members, scores, rows["groups"], "validation")
    specialist_rows, specialist_scores = chain_rows(args.specialists, "validation")
    if not np.array_equal(specialist_rows["labels"], rows["labels"]):
        raise SystemExit("The specialists' chains38 rows differ from the members'.")
    specialist_validation = ensemble(specialist_scores)
    mocha = rows["task"] == "MOCHATIMIT"
    boundary = mocha & (rows["positions"] != 1)
    lines = ["# Trial 23 -- P6 applied: the specialists on the holdout's first and last rows", "",
             f"{len(args.specialists)} specialists; the weight chosen on {int(boundary.sum())} MOCHA validation boundary rows "
             "(BAcc@10 after the hub correction fitted on those rows):", "", "| weight | BAcc@10 |", "|---|---|"]
    results = {}
    for weight in WEIGHTS:
        blended = cf.zscore_rows(cf.zscore_rows(base_validation[boundary]) + weight * cf.zscore_rows(specialist_validation[boundary]))
        results[weight] = bacc(hubbed(blended), rows["labels"][boundary])
        lines.append(f"| {weight:g} | {results[weight]:.2f} |")
    chosen = max(results, key=results.get)
    lines += ["", f"Chosen weight: **{chosen:g}** ({results[chosen] - results[0.0]:+.2f} over the base alone)."]

    #the holdout: first and last rows blended, every middle row exactly the base
    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    index, primary, moses = read_submission(Path(args.base), *words)
    metadata = LibriBrainCompetitionHoldout(track="deep", data_path=args.holdout_dir).metadata
    if not np.array_equal(index, np.arange(len(metadata))):
        raise SystemExit(f"{args.base}: its rows are not the holdout's 0..{len(metadata) - 1}.")
    sentence = np.array([meta["source"] == "sentence" for meta in metadata])
    positions = np.full(len(metadata), 1)
    positions[sentence] = position_groups(np.array([meta["epoch"] for meta in metadata])[sentence])
    edge = sentence & (positions != 1)
    base = standardise_rows(primary)
    specialists = ensemble([standardise_rows(read_submission(Path(args.submissions) / f"{Path(run).name}_in_context_ts.csv", *words)[1])
                            for run in args.specialists])
    combined = base.copy()
    combined[edge] = cf.zscore_rows(cf.zscore_rows(base[edge]) + chosen * cf.zscore_rows(specialists[edge]))
    out_path = Path(args.base).with_name(f"{Path(args.base).stem}_p6.csv")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    write_submission(str(out_path), indices=index, primary_probs=softmax_rows(zscore_scores(combined)),
                     secondary_probs=moses / moses.sum(axis=1, keepdims=True))
    lines.append(f"Holdout: {int(edge.sum())} of {int(sentence.sum())} sentence rows are first or last words; wrote `{out_path.name}`.")
    report = Path(args.report) if args.report else RUNS_DIR / f"trial23_boundary_apply_{Path(args.base).stem}.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[boundary] wrote {out_path} and {report}")


def main():
    """Diagnose or apply."""
    args = parse_args()
    diagnose(args) if args.mode == "diagnose" else apply(args)


if __name__ == "__main__":
    main()
