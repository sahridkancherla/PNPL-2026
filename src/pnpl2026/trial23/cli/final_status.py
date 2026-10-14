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
"""

from __future__ import annotations

import argparse
import json
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
    return parser.parse_args()


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
