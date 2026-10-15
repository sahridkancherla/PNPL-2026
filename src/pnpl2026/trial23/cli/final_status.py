"""Where the final night's members stand: files on disk, local scores, verdicts.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.final_status

One row per line of configs/trial23/newmembers.txt:

    csv       submissions/<run>_in_context_ts.csv, which the robust step mixes
    dump      runs/<run>/dumps/reference_validation.npz, which it fits on
    fragment  BAcc@10 on the fragment protocol, validation / test
    against   the parent's (anneal and head members, from anneal.json), or the
              same SPW seed at the trained tower weight 0.5 (tower members)
    verdict   anneal.json's: "keep" within 1 point of the parent, else "drop"

Then the EXCLUDE regex that leaves the dropped members out, and which
groups are complete enough for the robust step's INCLUDE.

    --exclude-below   print ONLY an EXCLUDE regex (one line): the members
                      anneal.json dropped, plus every other member whose
                      fragment BAcc@10 (mean of validation and test) sits more
                      than --margin points below the --reference-runs' mean.
                      "^$" when nothing is excluded. The robust step's SANITY=1
                      reads it (pbs/trial23_cpu.pbs); members without a score
                      pass (All-0's scores are not clean, and are not judged)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from ...config import REPO_ROOT, RUNS_DIR, SUBMISSIONS_DIR

#the robust step's INCLUDE prefixes, in the order they are submitted
GROUPS = (
    ("anneal", "trial23_an_"),
    ("heads", "trial23_hg_|trial23_hq_"),
    ("tower", "_tw07"),
    ("sam", "trial23_sam_"),
    ("f0fixed", "trial18_f0fixed"),
    ("anneal last epochs", "trial23_anlast_"),
)


def parse_args():
    """Command-line arguments for the status table."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", default="configs/trial23/newmembers.txt", help="repo-relative")
    parser.add_argument("--runs-dir", default=None, help="default runs/; tests point elsewhere")
    parser.add_argument("--submissions-dir", default=None, help="default submissions/")
    parser.add_argument("--exclude-below", action="store_true",
                        help="print only the EXCLUDE regex of the members failing the sanity filter")
    parser.add_argument("--margin", type=float, default=1.0, help="--exclude-below: BAcc@10 points below the reference")
    parser.add_argument("--reference-runs", nargs="+", default=None,
                        help="--exclude-below: run directories whose fragment scores' mean is the reference")
    return parser.parse_args()


def fragment_mean(pair):
    """The mean of validation and test BAcc@10, in points, or None.

    ARGS:
        pair (tuple or None): from `fragment_pair`

    RETURNS:
        float or None
    """
    return None if pair is None else 0.5 * (pair[0] + pair[1])


def exclude_regex(members, runs_dir, annealed, reference_runs, margin):
    """The EXCLUDE regex of the members that fail the sanity filter.

    A member anneal.json judged keeps its verdict; any other member with a
    fragment score more than `margin` points below the reference runs' mean
    is excluded. A member without a score passes.

    ARGS:
        members (list): run names
        runs_dir (Path)
        annealed (dict): from `anneal_records`
        reference_runs (list or None): run directories, e.g. G2's three seeds
        margin (float): points

    RETURNS:
        str: anchored names joined by |, or "^$" when nothing is excluded
    """
    #the references as anneal.json scored them, under this trial's protocol: a parent's
    #own evaluation.json may come from an older protocol and not compare
    parents = parent_scores(runs_dir)
    references = [fragment_mean(parents.get(Path(run).name)) for run in reference_runs or []]
    references = [value for value in references if value is not None]
    threshold = sum(references) / len(references) - margin if references else None

    excluded = []
    for member in members:
        #anneal.json already compared these with their own parent
        if member in annealed:
            if annealed[member][2] == "drop":
                excluded.append(member)
            continue
        mine = fragment_mean(fragment_pair(read_json(runs_dir / member / "evaluation.json")))
        if mine is not None and threshold is not None and mine < threshold:
            excluded.append(member)
    return "|".join(f"^{re.escape(member)}$" for member in excluded) or "^$"


def read_members(path):
    """The member names, comment lines skipped.

    ARGS:
        path (Path)

    RETURNS:
        list of str
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def fragment_pair(report):
    """Fragment BAcc@10 on validation and test from an evaluation report, or None.

    ARGS:
        report (dict or None)

    RETURNS:
        (validation, test) in points, or None
    """
    fragment = (report or {}).get("results", {}).get("fragment")
    if not fragment or "validation" not in fragment or "test" not in fragment:
        return None
    return 100 * fragment["validation"]["bacc10"], 100 * fragment["test"]["bacc10"]


def read_json(path):
    """A JSON file's content, or None when it is not there.

    ARGS:
        path (Path)

    RETURNS:
        dict or None
    """
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def anneal_records(runs_dir):
    """Every anneal.json, by the member it judges: the averaged run and its last epoch.

    ARGS:
        runs_dir (Path)

    RETURNS:
        dict: member -> (its fragment scores, the parent's, its verdict), points
    """
    found = {}
    for path in sorted(runs_dir.glob("*/anneal.json")):
        record = read_json(path)
        scores = record["fragment_bacc10"]
        parent = (100 * scores["parent"]["validation"], 100 * scores["parent"]["test"])
        for key, member in (("averaged", record["run_name"]), ("last", record["fallback"])):
            mine = (100 * scores[key]["validation"], 100 * scores[key]["test"])
            found[member] = (mine, parent, record["verdict"][key])
    return found


def parent_scores(runs_dir):
    """Each anneal parent's fragment scores, as anneal.json measured them.

    ARGS:
        runs_dir (Path)

    RETURNS:
        dict: parent run name -> (validation, test) in points
    """
    found = {}
    for path in sorted(runs_dir.glob("*/anneal.json")):
        record = read_json(path)
        parent = record["fragment_bacc10"]["parent"]
        #the parent's run directory is the checkpoint's folder
        found[Path(record["parent"]).parent.name] = (100 * parent["validation"], 100 * parent["test"])
    return found


def describe(pair):
    """Validation / test as text, or a dash.

    ARGS:
        pair (tuple or None)

    RETURNS:
        str
    """
    return "-" if pair is None else f"{pair[0]:.2f} / {pair[1]:.2f}"


def main():
    """Print the table, the EXCLUDE regex, and the groups ready to include."""
    args = parse_args()
    runs_dir = Path(args.runs_dir) if args.runs_dir else RUNS_DIR
    submissions_dir = Path(args.submissions_dir) if args.submissions_dir else SUBMISSIONS_DIR
    members = read_members(REPO_ROOT / args.members)
    annealed = anneal_records(runs_dir)

    #one line for the robust step's SANITY switch, nothing else
    if args.exclude_below:
        print(exclude_regex(members, runs_dir, annealed, args.reference_runs, args.margin))
        return

    print("| member | csv | dump | fragment val / test | against | verdict |")
    print("|---|---|---|---|---|---|")
    ready, dropped = {}, []
    for member in members:
        has_csv = (submissions_dir / f"{member}_in_context_ts.csv").exists()
        has_dump = (runs_dir / member / "dumps" / "reference_validation.npz").exists()
        ready[member] = has_csv and has_dump
        mine, against, verdict = None, None, "-"

        #anneal and head members: anneal.json judged them against their parent
        if member in annealed:
            mine, against, verdict = annealed[member]
            against_text = f"parent {describe(against)}"
            if verdict == "drop":
                dropped.append(member)
        else:
            mine = fragment_pair(read_json(runs_dir / member / "evaluation.json"))
            #tower members: the same seed at the trained weight, scored beside it
            plain = fragment_pair(read_json(runs_dir / member / "evaluation_plain.json"))
            against_text = f"weight 0.5 {describe(plain)}" if plain else "-"
        print(f"| {member} | {'yes' if has_csv else 'NO'} | {'yes' if has_dump else 'NO'} | {describe(mine)} | "
              f"{against_text} | {verdict} |")

    #the robust step's arguments
    print()
    if dropped:
        print(f"Dropped by the sanity check: EXCLUDE='{'|'.join(dropped)}'")
    else:
        print("No member dropped by the sanity check (yet).")
    for label, pattern in GROUPS:
        chosen = [member for member in members if any(part in member for part in pattern.split("|"))]
        complete = sum(ready[member] for member in chosen)
        state = "ready" if chosen and complete == len(chosen) else f"{complete} of {len(chosen)} ready"
        print(f"  {label:20s} INCLUDE='{pattern}'  {state}")


if __name__ == "__main__":
    main()
