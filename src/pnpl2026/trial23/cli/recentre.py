"""P8b of the 13 Oct research note: the reference covariance, and the gate on subject 0's held-out sessions.

    # 1. the training recordings' mean covariance (CPU job: pbs/trial23_recentre.pbs)
    PYTHONPATH=src python -m pnpl2026.trial23.cli.recentre reference

    # 2. every member's chains38 dumps re-centred, beside the plain ones:
    #    qsub -J 1-26 -v EMBED_ARGS="--recentre data/trial23_recentre_reference.npz" pbs/trial23_embed.pbs

    # 3. the gate: with against without, every task of both held-out partitions
    PYTHONPATH=src python -m pnpl2026.trial23.cli.recentre check --members trial13_g1_s0 ...

`reference` averages the covariance of every TRAINING recording of the
reference cache -- per-recording robust-scaled, 50 Hz, exactly what the
models read -- into data/trial23_recentre_reference.npz, and reports its
spectrum and how far each recording sits from it.

`check` scores each task's chains38 rows (subject 0's validation and test
sessions: Sherlock, MOCHA-TIMIT, TheMoth -- different days and speakers, as
the note asks; TIMIT is all in training) with and without re-centring, under
the plain mix and the 70.68 recipe (transduction v1, setup from the partner
partition's plain reference dumps), each hub-corrected on its own rows. The
note: submit only if it helps here -- the verdict reads the 70.68 recipe's
pooled test gain. Writes runs/trial23_recentre_check.md. CPU, a minute.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR
from ..data.dataset import load_manifest
from ..local_checks import bacc, ensemble
from ..recentre import N_DIRECTIONS, recording_covariance
from .boundary import chain_rows, hubbed, transduced_mix


def parse_args():
    """Command-line arguments: a mode, then its options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("reference", "check"))
    parser.add_argument("--cache-dir", default="data/trial10_reference", help="reference: the cache, repo-relative")
    parser.add_argument("--partition", default="train", help="reference: whose recordings are averaged")
    parser.add_argument("--directions", type=int, default=N_DIRECTIONS, help="reference: the eigendirections every transform keeps")
    parser.add_argument("--out", default="data/trial23_recentre_reference.npz", help="reference: repo-relative")
    parser.add_argument("--members", nargs="+", default=None, help="check: runs with chains38 dumps, plain and _rc")
    parser.add_argument("--report", default=None, help="each mode's default under runs/")
    return parser.parse_args()


def reference(args):
    """The training recordings' mean covariance, written with its spectrum."""
    cache_dir = REPO_ROOT / args.cache_dir
    records = [record for record in load_manifest(cache_dir)["recordings"] if record["partition"] == args.partition]
    if not records:
        raise SystemExit(f"No {args.partition} recordings in {cache_dir}.")

    #every recording's covariance from evenly spaced samples, then their mean
    covariances = []
    for index, record in enumerate(records):
        covariances.append(recording_covariance(np.load(cache_dir / f"meg_{record['key']}.npy", mmap_mode="r")))
        if (index + 1) % 20 == 0 or index + 1 == len(records):
            print(f"[recentre] {index + 1}/{len(records)} recordings read", flush=True)
    mean = np.mean(covariances, axis=0)
    eigenvalues = np.sort(np.linalg.eigvalsh(mean))[::-1]

    #how far each recording sits from the mean, relative to the mean's size
    distances = np.array([np.linalg.norm(covariance - mean) / np.linalg.norm(mean) for covariance in covariances])
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, covariance=mean, n_directions=args.directions, n_recordings=len(records), eigenvalues=eigenvalues)
    shown = [0, 72, args.directions - 1, args.directions, 95, 99]
    lines = ["# Trial 23 -- P8b's reference covariance", "",
             f"{len(records)} {args.partition} recordings of `{args.cache_dir}`; transforms keep {args.directions} directions.", "",
             "Spectrum of the mean: " + ", ".join(f"eig[{index}] = {eigenvalues[index]:.3g}" for index in shown) + ".",
             f"Each recording's distance from the mean (Frobenius, relative): 5th / 50th / 95th percentile "
             f"{np.percentile(distances, 5):.3f} / {np.median(distances):.3f} / {np.percentile(distances, 95):.3f}."]
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_recentre_reference.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[recentre] wrote {out_path} and {report}")


def task_bacc(rows, scores):
    """BAcc@10 per task and pooled, each task hub-corrected on its own rows.

    ARGS:
        rows (dict), scores (np.ndarray): [n, 50]

    RETURNS:
        dict: task (and "pooled") -> BAcc@10
    """
    corrected = np.zeros_like(scores)
    for task in np.unique(rows["task"]):
        corrected[rows["task"] == task] = hubbed(scores[rows["task"] == task])
    result = {str(task): bacc(corrected[rows["task"] == task], rows["labels"][rows["task"] == task]) for task in np.unique(rows["task"])}
    result["pooled"] = bacc(corrected, rows["labels"])
    return result


def recipes(members, rows, scores, partition):
    """The plain mix and the 70.68 recipe on one partition's rows, each task on its own.

    ARGS:
        members (list), rows (dict), scores (list of [n, 50]), partition (str)

    RETURNS:
        dict: recipe name -> [n, 50] before the hub correction
    """
    transduced = np.zeros((len(rows["labels"]), 50))
    for task in np.unique(rows["task"]):
        of_task = rows["task"] == task
        transduced[of_task] = transduced_mix(members, [table[of_task] for table in scores], rows["groups"][of_task], partition)
    return {"plain mix": ensemble(scores), "70.68 recipe": transduced}


def check(args):
    """Every task's rows with and without re-centring; the note's gate."""
    if not args.members:
        raise SystemExit("check needs --members.")
    lines = ["# Trial 23 -- P8b: session re-centring, checked on subject 0's held-out sessions", "",
             f"{len(args.members)} members; chains38 rows of every task. BAcc@10 after each task's own hub correction.", ""]
    gains = {}
    for partition in ("validation", "test"):
        rows, scores = chain_rows(args.members, partition, tasks=None)
        recentred_rows, recentred_scores = chain_rows(args.members, partition, tasks=None, suffix="_rc")
        #⚠️ the same tokens in both, or the comparison compares different words
        if not np.array_equal(rows["labels"], recentred_rows["labels"]):
            raise SystemExit(f"The {partition} dumps with and without re-centring list different tokens.")
        plain, recentred = recipes(args.members, rows, scores, partition), recipes(args.members, rows, recentred_scores, partition)
        tasks = [str(task) for task in np.unique(rows["task"])] + ["pooled"]
        lines += [f"## {partition}", "", "| recipe | " + " | ".join(tasks) + " |", "|---" * (len(tasks) + 1) + "|"]
        for recipe in plain:
            without, with_it = task_bacc(rows, plain[recipe]), task_bacc(rows, recentred[recipe])
            lines.append(f"| {recipe} | " + " | ".join(f"{without[task]:.2f} -> {with_it[task]:.2f} ({with_it[task] - without[task]:+.2f})"
                                                     for task in tasks) + " |")
            gains[(partition, recipe)] = with_it["pooled"] - without["pooled"]
        lines.append("")
    passes = gains[("test", "70.68 recipe")] > 0
    lines += [f"Verdict: re-centring {'helps' if passes else 'does not help'} the 70.68 recipe on pooled test "
              f"({gains[('test', '70.68 recipe')]:+.2f}; validation {gains[('validation', '70.68 recipe')]:+.2f}) -- "
              + ("worth a slot (predict every member with --recentre)." if passes else "the note says do not submit it."),
              "The transduction setup is the plain reference dumps' in both columns."]
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_recentre_check.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[recentre] wrote {report}")


def main():
    """The reference, or the check."""
    args = parse_args()
    reference(args) if args.mode == "reference" else check(args)


if __name__ == "__main__":
    main()
