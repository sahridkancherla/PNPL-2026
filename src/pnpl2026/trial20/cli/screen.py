"""Screen candidate models before spending submissions on them.

    PYTHONPATH=src python -m pnpl2026.trial20.cli.screen --holdout-dir data/holdout \\
        --group G1=trial13_g1_s0,trial13_g1_s1,trial13_g1_s2 \\
        --group L=trial20_l_s0,trial20_l_s1,trial20_l_s2

No local score has ranked recipes as the board does, and L lost 24 points on
the board (39.83 against G2's 63.41) while leading G2 on every local test. A
label-free question can still be asked: does a model behave on the holdout as
it does on local data? Each group's ensemble is compared with the REFERENCE
group's (G2, the best recipe on the board), twice:

    holdout   the holdout's sentence rows, from the compliant CSVs
              `submissions/<run><suffix>.csv`
    local     the fragment protocol's test words (runs of list words, as the
              holdout's), from `runs/<run>/dumps/<protocol>_test.npz`

and reports, for each, top-1 agreement and the mean overlap of the two top-10
sets, then how far the holdout overlap falls short of the local one. Also,
within each group, how much its own seeds agree with each other, holdout and
local, and how many distinct words reach the holdout top 1.

How to read it: a group that agrees with G2 about as much on the holdout as
locally behaves normally there. A large shortfall flags a model whose
behaviour shifts on the holdout's sources; agreement near chance (top-1 2%,
top-10 overlap 20%) flags a broken prediction. G1 (62.84 on the board) is the
known-good comparison. Nothing here sees a label of the holdout; local labels
give only the groups' own local BAcc@10, for reference.

Writes runs/trial20_screen.md. Light: CSVs and saved arrays, seconds.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from ...config import RUNS_DIR, SUBMISSIONS_DIR
from ...metrics import balanced_topk_accuracy
from .ensemble import holdout_sources, read_submission, standardise_rows
from .lone_check import zscore_rows

#the dump key holding each token's 50 primary scores; trial 13 wrote "retrieval"
SCORE_KEYS = ("primary_logits", "retrieval")

#BAcc@10's k, and the 50 scored words
K = 10
N_WORDS = 50

#a holdout top-10 overlap this far below the local one is flagged
SHORTFALL_FLAG = 0.15


def parse_args():
    """Command-line arguments for the screen."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="trial13_g2_s0,trial13_g2_s1,trial13_g2_s2",
                        help="the reference group's run names, comma-separated (default G2)")
    parser.add_argument("--group", action="append", required=True, help="NAME=run,run,... ; once per candidate group")
    parser.add_argument("--suffix", default="_in_context_ts", help="holdout CSVs are submissions/<run><suffix>.csv")
    parser.add_argument("--protocol", default="fragment", help="local dumps: runs/<run>/dumps/<protocol>_test.npz")
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--track", default="deep", help="competition track")
    parser.add_argument("--out", default=None, help="default runs/trial20_screen.md")
    return parser.parse_args()


def holdout_scores(run, suffix, primary_words, moses_words):
    """One run's holdout rows, z-scored as cli.ensemble reads a CSV.

    ARGS:
        run (str): run name
        suffix (str): the CSV's mode suffix
        primary_words (list), moses_words (list): the vocabularies, column order

    RETURNS:
        [n_rows, 50] float64
    """
    index, primary, _ = read_submission(SUBMISSIONS_DIR / f"{run}{suffix}.csv", primary_words, moses_words)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{run}{suffix}.csv: the index column is not 0..{len(index) - 1} in order.")
    return standardise_rows(primary)


def local_scores(run, protocol):
    """One run's scored local test words, z-scored, and their true columns.

    ARGS:
        run (str): run name; reads runs/<run>/dumps/<protocol>_test.npz
        protocol (str)

    RETURNS:
        scores [n_scored, 50], targets [n_scored]
    """
    path = RUNS_DIR / run / "dumps" / f"{protocol}_test.npz"
    with np.load(path) as stored:
        key = next((name for name in SCORE_KEYS if name in stored.files), None)
        if key is None:
            raise SystemExit(f"{path}: none of {SCORE_KEYS} in {stored.files}")
        primary = stored["primary"].astype(np.int64)
        scored = primary >= 0
        return zscore_rows(stored[key].astype(np.float64)[scored]), primary[scored]


def ensemble(score_sets):
    """Score sets averaged after z-scoring, then z-scored again, as cli.ensemble.

    ARGS:
        score_sets (list): of [n, 50] arrays, the same rows

    RETURNS:
        [n, 50] float64
    """
    #axis 0 is the member axis
    return zscore_rows(np.mean([zscore_rows(scores) for scores in score_sets], axis=0))


def agreement(first, second):
    """Top-1 agreement and mean top-10 overlap of two score sets, row by row.

    ARGS:
        first (np.ndarray), second (np.ndarray): [n, 50], the same rows

    RETURNS:
        (top1, top10): shares in [0, 1]; chance is 0.02 and 0.2
    """
    #axis=1 is the word axis: each row's best word, and its ten best
    top1 = float(np.mean(first.argmax(axis=1) == second.argmax(axis=1)))
    first_top = np.argsort(-first, axis=1)[:, :K]
    second_top = np.argsort(-second, axis=1)[:, :K]
    overlap = np.mean([len(set(a) & set(b)) / K for a, b in zip(first_top, second_top)])
    return top1, float(overlap)


def seed_agreement(score_sets):
    """Mean top-10 overlap between every pair of a group's members.

    ARGS:
        score_sets (list): of [n, 50] arrays

    RETURNS:
        float, or nan with one member
    """
    pairs = [agreement(a, b)[1] for a, b in combinations(score_sets, 2)]
    return float(np.mean(pairs)) if pairs else float("nan")


def main():
    """Load every group on the holdout and locally, compare with the reference, report."""
    args = parse_args()
    from pnpl.competition import load_vocabulary

    primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
    groups = {"reference": [run for run in args.reference.split(",") if run]}
    for spec in args.group:
        name, _, runs = spec.partition("=")
        if not name or not runs:
            raise SystemExit(f"--group {spec!r} is not NAME=run,run,...")
        groups[name] = [run for run in runs.split(",") if run]

    #the holdout's sentence rows, from the loader's own metadata
    sentence = None

    #every group: its members' holdout rows and local words
    loaded = {}
    for name, runs in groups.items():
        holdout_sets = [holdout_scores(run, args.suffix, primary_words, moses_words) for run in runs]
        if sentence is None:
            sentence = holdout_sources(args.track, args.holdout_dir, len(holdout_sets[0])) == "sentence"
        local_sets, targets = [], None
        for run in runs:
            scores, these = local_scores(run, args.protocol)
            if targets is not None and not np.array_equal(these, targets):
                raise SystemExit(f"{run}: its {args.protocol} test words differ from {runs[0]}'s.")
            targets = these
            local_sets.append(scores)
        loaded[name] = {
            "holdout": [scores[sentence] for scores in holdout_sets],
            "local": local_sets,
            "targets": targets,
        }

    #⚠️ every group must score the same local words as the reference
    reference = loaded["reference"]
    for name, data in loaded.items():
        if not np.array_equal(data["targets"], reference["targets"]):
            raise SystemExit(f"Group {name}: its local {args.protocol} test words differ from the reference's.")

    reference_holdout = ensemble(reference["holdout"])
    reference_local = ensemble(reference["local"])
    lines = [
        "# Trial 17 -- do the candidates behave on the holdout as they do locally?",
        "",
        f"Reference: {', '.join(groups['reference'])}. Holdout: {int(sentence.sum())} sentence rows "
        f"(`<run>{args.suffix}.csv`); local: {len(reference['targets'])} {args.protocol}-protocol test words.",
        "Agreement with the reference's ensemble: top-1 share and mean top-10 overlap (chance 2% and 20%).",
        f"'shortfall' is local minus holdout top-10 overlap; above {SHORTFALL_FLAG:.2f} is flagged.",
        "",
        "| group | runs | holdout top-1 | holdout top-10 | local top-1 | local top-10 | shortfall | "
        "seeds agree (holdout / local) | holdout top-1 words | local BAcc@10 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, data in loaded.items():
        holdout = ensemble(data["holdout"])
        local = ensemble(data["local"])
        holdout_top1, holdout_top10 = agreement(holdout, reference_holdout)
        local_top1, local_top10 = agreement(local, reference_local)
        shortfall = local_top10 - holdout_top10
        flag = " FLAG" if name != "reference" and shortfall > SHORTFALL_FLAG else ""
        local_bacc = 100.0 * balanced_topk_accuracy(torch.from_numpy(local), torch.from_numpy(data["targets"]), K, N_WORDS)
        lines.append(
            f"| {name} | {len(groups[name])} | {100 * holdout_top1:.0f}% | {100 * holdout_top10:.0f}% | "
            f"{100 * local_top1:.0f}% | {100 * local_top10:.0f}% | {shortfall:+.2f}{flag} | "
            f"{100 * seed_agreement(data['holdout']):.0f}% / {100 * seed_agreement(data['local']):.0f}% | "
            f"{len(np.unique(holdout.argmax(axis=1)))} | {local_bacc:.1f} |"
        )

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial20_screen.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[screen] wrote {out_path}")


if __name__ == "__main__":
    main()
