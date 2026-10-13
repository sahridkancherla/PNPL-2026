"""The kNN memory voter: change 2 of the Route to 73, second half.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.knn_vote --runs trial13_g2_s0 ... \\
        --out submissions/knn_mix_ts.csv --holdout-dir data/holdout

Every member scores the holdout by comparing its embedding with the 50 words'
TEXT vectors. The memory voter compares it with the member's own embeddings
of real training occurrences of the words instead, cut as the holdout is cut
(`cli/embed.py`'s training chains): each holdout row votes for the words of
its K nearest stored tokens, weighted by exp(cosine / temperature), each
word's vote divided by how many tokens of it the memory holds -- balanced,
as the metric is. Heads are scored apart and averaged z-scored, as the
members combine their heads. No holdout statistics at all.

The members' kNN scores are averaged as cli.ensemble averages CSVs into one
file, which joins a stack as ONE extra voter, as the doc proposes:

    cli.ensemble --group REST=<best>.csv --group KNN=knn_mix_ts.csv --weights REST=23,KNN=1

The report's local check: each member's TEST chains queried against its
training memory, with labels -- the kNN vote alone, the member's own scores
alone, and the two averaged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...config import RUNS_DIR
from ..local_checks import bacc, zscore_rows
from ..transduction import softmax_rows

#heads in the order the members' dumps may carry them
HEADS = ("t5", "second", "tower")

#a word with no vote this close to zero, inside the logarithm
FLOOR = 1e-6


def parse_args():
    """Command-line arguments for the memory voter."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", required=True, help="members, each with runs/<run>/embeddings/")
    parser.add_argument("--neighbours", type=int, default=32, help="K, stored tokens consulted per row")
    parser.add_argument("--temperature", type=float, default=0.05, help="cosine temperature of the votes")
    parser.add_argument("--out", required=True, help="the members' kNN mix, as a submission CSV")
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--track", default="deep")
    parser.add_argument("--chunk", type=int, default=512, help="query rows per similarity block")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def embeddings_dir(run):
    """Where a member's dumps live.

    ARGS:
        run (str): a run name or directory

    RETURNS:
        Path
    """
    path = Path(run)
    return path / "embeddings" if path.is_dir() else RUNS_DIR / path.name / "embeddings"


def load(run, name):
    """One of a member's dumps.

    ARGS:
        run (str)
        name (str): "holdout", "chains_train", "chains_test", ...

    RETURNS:
        dict of numpy arrays
    """
    path = embeddings_dir(run) / f"{name}.npz"
    if not path.exists():
        raise SystemExit(f"No {path}: run cli.embed (pbs/trial23_embed.pbs) for {run} first.")
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def knn_scores(queries, memory, labels, n_words=50, neighbours=32, temperature=0.05, chunk=512):
    """Class-balanced soft kNN votes.

    ARGS:
        queries (np.ndarray): [n, D] unit vectors
        memory (np.ndarray): [m, D] unit vectors
        labels (np.ndarray): [m] word of each stored token
        n_words (int), neighbours (int), temperature (float), chunk (int)

    RETURNS:
        [n, n_words] float64: log of each word's balanced vote
    """
    counts = np.maximum(np.bincount(labels, minlength=n_words), 1).astype(np.float64)
    memory = memory.astype(np.float32)
    scores = np.empty((len(queries), n_words))
    keep = min(neighbours, len(memory))
    for start in range(0, len(queries), chunk):
        #cosines of a block of rows against the whole memory
        block = queries[start : start + chunk].astype(np.float32) @ memory.T
        nearest = np.argpartition(-block, keep - 1, axis=1)[:, :keep]
        similarity = np.take_along_axis(block, nearest, axis=1).astype(np.float64)

        #each neighbour's vote, summed per word, divided by the word's count
        weights = np.exp((similarity - similarity.max(axis=1, keepdims=True)) / temperature)
        votes = np.zeros((len(block), n_words))
        np.add.at(votes, (np.repeat(np.arange(len(block)), keep), labels[nearest].ravel()), weights.ravel())
        scores[start : start + len(block)] = np.log(votes / counts[None, :] + FLOOR)
    return scores


def member_votes(run, queries_name, args):
    """A member's kNN scores for one dump's rows, every head averaged.

    ARGS:
        run (str)
        queries_name (str): "holdout" or "chains_test"
        args (argparse.Namespace)

    RETURNS:
        scores [n, 50] (row z-scored, heads averaged), the query dump
    """
    memory = load(run, "chains_train")
    queries = load(run, queries_name)
    stored = memory["primary"] >= 0
    heads = [head for head in HEADS if head in memory and head in queries]
    if not heads:
        raise SystemExit(f"{run}: its dumps share no head.")
    per_head = []
    for head in heads:
        #the stored tokens of scored words are the memory
        bank = memory[head][stored].astype(np.float32)
        bank /= np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-12)
        rows = queries[head].astype(np.float32)
        rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
        per_head.append(zscore_rows(knn_scores(rows, bank, memory["primary"][stored].astype(np.int64),
                                               neighbours=args.neighbours, temperature=args.temperature, chunk=args.chunk)))
    return zscore_rows(np.mean(per_head, axis=0)), queries


def local_check(run, args):
    """kNN alone, the member alone and both, on the member's test chains.

    ARGS:
        run (str)
        args (argparse.Namespace)

    RETURNS:
        str: one report line
    """
    knn, test = member_votes(run, "chains_test", args)
    scored = test["primary"] >= 0
    labels = test["primary"][scored].astype(np.int64)
    own = zscore_rows(test["logits"][scored].astype(np.float64))
    vote = knn[scored]
    return (f"| {Path(run).name} | {int(scored.sum())} | {bacc(vote, labels):.2f} | {bacc(own, labels):.2f} | "
            f"{bacc(zscore_rows(own + vote), labels):.2f} |")


def main():
    """Every member's memory votes on the holdout, mixed; the local check; the CSV."""
    args = parse_args()
    from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    n_rows = len(LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir))
    lines = [f"# Trial 22 -- kNN memory voter: `{out_path.name}`", "",
             f"{len(args.runs)} members; K = {args.neighbours}, temperature {args.temperature:g}; memory = each "
             "member's training chains, capped per word; votes balanced by each word's count.", "",
             "Local check, test chains (BAcc@10, chance 20):", "",
             "| member | test tokens | kNN alone | member alone | both averaged |", "|---|---|---|---|---|"]

    tables = []
    for run in args.runs:
        votes, holdout = member_votes(run, "holdout", args)
        if len(votes) != n_rows:
            raise SystemExit(f"{run}: its holdout dump has {len(votes)} rows; the holdout has {n_rows}.")
        tables.append(votes)
        lines.append(local_check(run, args))
        print(lines[-1])

    #the members' votes, averaged as cli.ensemble averages CSVs
    mixed = zscore_rows(np.mean(tables, axis=0))
    primary = softmax_rows(mixed)
    write_submission(str(out_path), indices=np.arange(n_rows), primary_probs=primary, secondary_probs=np.full_like(primary, 1.0 / 50))
    report = RUNS_DIR / f"trial23_knn_{out_path.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[knn] wrote {out_path} and {report}")


if __name__ == "__main__":
    main()
