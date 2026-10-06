"""Build the word-pair table from subject 0's own transcripts.

    PYTHONPATH=src python -m pnpl2026.trial15.cli.wordpairs \\
        [--partitions train] [--holdout-dir data/holdout]

Reads the reference cache's manifest.json and `words_<key>.npz` tables -- text
and onsets only, never MEG -- counts how often each scored word directly
follows each other within a sentence, chooses the smoothing that best predicts
the held-out partition's pairs, and writes

    data/trial15_wordpairs_<partitions>.npz     read by cli.fuse and cli.fuse_check

It also prints what the table knows:

    - how far the previous word alone ranks the next one, balanced over words,
      when the previous word is known (an upper bound: the holdout's neighbours
      are only ever guessed)
    - onset gaps of adjacent scored pairs beside gaps across one unscored word,
      and, with --holdout-dir, the holdout's own gaps inside its runs: the
      fusion assumes the holdout's runs are directly adjacent words

⚠️ Keep the default `--partitions train` for anything `cli.fuse_check` scores:
a table counted on the validation or test transcripts has seen the very word
sequences it would be judged on.

Light: reads a few MB of tables, seconds on a login node.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from ...config import REPO_ROOT
from ...metrics import balanced_topk_accuracy
from ..config import Trial15Config
from ..wordpairs import (
    N_WORDS,
    choose_smoothing,
    count_word_pairs,
    save_table,
    stationary_distribution,
    transition_matrix,
)

#the cache's sample rate: onsets in the word tables are 50 Hz samples
SAMPLE_RATE = 50.0

#the k BAcc@k is scored at
K = 10


def parse_args():
    """Command-line arguments for building the table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=Trial15Config.cache_dir, help="repo-relative reference cache")
    parser.add_argument("--partitions", nargs="+", default=["train"], help="pnpl partitions to count")
    parser.add_argument("--held-out", default="validation", help="partition the smoothing is chosen on")
    parser.add_argument("--out", default=None, help="default data/trial15_wordpairs_<partitions>.npz")
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/, for its gaps")
    parser.add_argument("--track", default="deep", help="competition track")
    return parser.parse_args()


def read_tables(cache_dir, partition):
    """Every word table of one pnpl partition.

    ARGS:
        cache_dir (Path): the reference cache
        partition (str): "train", "validation" or "test"

    RETURNS:
        list of dict, one per recording: primary, sentence, onset_sample
    """
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    tables = []
    for record in manifest["recordings"]:
        if record["partition"] != partition:
            continue

        #only the three columns the counts need; the strings stay on disk
        with np.load(cache_dir / f"words_{record['key']}.npz") as stored:
            tables.append({name: stored[name] for name in ("primary", "sentence", "onset_sample")})
    return tables


def previous_word_bacc(transitions, stationary, held_out_pairs):
    """BAcc@10 of naming a word from the TRUE previous word alone.

    Ranks the 50 words by transitions[previous] / stationary -- the same
    prior-free ratio the fusion uses -- for every held-out pair.

    ARGS:
        transitions (np.ndarray): [N_WORDS, N_WORDS]
        stationary (np.ndarray): [N_WORDS]
        held_out_pairs (np.ndarray): [N_WORDS, N_WORDS] counts

    RETURNS:
        float, or nan without pairs
    """
    #one row per held-out pair occurrence: its previous word's ratios
    previous, following = np.nonzero(held_out_pairs)
    repeats = held_out_pairs[previous, following].astype(np.int64)
    if repeats.sum() == 0:
        return float("nan")
    scores = np.log(transitions[np.repeat(previous, repeats)]) - np.log(stationary)[None, :]
    targets = np.repeat(following, repeats)
    return balanced_topk_accuracy(torch.from_numpy(scores), torch.from_numpy(targets), K, N_WORDS)


def holdout_gaps(holdout_dir, track):
    """Onset gaps between consecutive words inside the holdout's runs.

    ARGS:
        holdout_dir (str or None): folder holding COMPETITION_HOLDOUT/
        track (str): "deep"

    RETURNS:
        np.ndarray of gaps in seconds
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    holdout = LibriBrainCompetitionHoldout(track=track, data_path=holdout_dir)

    #each run is one sentence epoch; its words in the loader's slot order
    runs = defaultdict(list)
    for meta in holdout.metadata:
        if meta["source"] == "sentence":
            runs[(meta["subject"], meta["epoch"])].append(meta)
    gaps = []
    for members in runs.values():
        onsets = np.array([meta["onset_s"] for meta in sorted(members, key=lambda meta: meta["word"])])
        gaps.append(np.diff(onsets))
    return np.concatenate(gaps) if gaps else np.zeros(0)


def quantiles(values):
    """5th, 25th, 50th, 75th and 95th percentiles as one line.

    ARGS:
        values (np.ndarray)

    RETURNS:
        str
    """
    if len(values) == 0:
        return "none"
    points = np.percentile(values, [5, 25, 50, 75, 95])
    return " ".join(f"{point:.2f}" for point in points) + f"  (n={len(values)})"


def main():
    """Count, smooth, save, and report."""
    args = parse_args()
    from pnpl.competition import load_vocabulary

    cache_dir = REPO_ROOT / args.cache_dir
    words = list(load_vocabulary("primary"))

    #counts from the chosen partitions; the held-out one picks the smoothing
    counts = count_word_pairs(table for partition in args.partitions for table in read_tables(cache_dir, partition))
    held_out = count_word_pairs(read_tables(cache_dir, args.held_out))
    if counts["pairs"].sum() == 0:
        raise SystemExit(f"No adjacent scored pairs in {args.partitions}; is the cache at {cache_dir}?")
    if args.held_out in args.partitions:
        print(f"[wordpairs] WARNING: {args.held_out} is also counted; the smoothing choice is optimistic")

    smoothing, smoothing_scores = choose_smoothing(counts["pairs"], held_out["pairs"])
    out_path = REPO_ROOT / (args.out or f"data/trial15_wordpairs_{'_'.join(args.partitions)}.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_table(out_path, counts, smoothing, list(args.partitions), smoothing_scores)

    #what the table knows, in a few lines
    transitions = transition_matrix(counts["pairs"], smoothing)
    stationary = stationary_distribution(transitions)
    n_tokens = counts["unigram"].sum()
    print(f"[wordpairs] {int(counts['pairs'].sum())} adjacent scored pairs from {int(n_tokens)} scored tokens "
          f"in {args.partitions}; {int(held_out['pairs'].sum())} held-out pairs in {args.held_out}")
    for value, nats in smoothing_scores.items():
        print(f"[wordpairs]   smoothing {value:6.1f}: held-out {nats:.4f} nats per pair" + ("  <- chosen" if value == smoothing else ""))
    print(f"[wordpairs] uniform would be {-np.log(N_WORDS):.4f} nats per pair")

    #the upper bound: the previous word known exactly
    bacc = previous_word_bacc(transitions, stationary, held_out["pairs"])
    print(f"[wordpairs] BAcc@{K} naming a word from the TRUE previous word alone: {100 * bacc:.2f} (chance 20)")

    #the commonest pairs, as words
    order = np.argsort(-counts["pairs"], axis=None)[:15]
    common = [f"{words[index // N_WORDS]} {words[index % N_WORDS]} ({int(counts['pairs'].flat[index])})" for index in order]
    print("[wordpairs] commonest pairs: " + ", ".join(common))

    #onset gaps in seconds: the fusion assumes the holdout's runs are adjacent words
    print(f"[wordpairs] onset gaps, s, 5/25/50/75/95%:")
    print(f"[wordpairs]   adjacent scored pairs        {quantiles(counts['adjacent_gaps'] / SAMPLE_RATE)}")
    print(f"[wordpairs]   across one unscored word     {quantiles(counts['skip_gaps'] / SAMPLE_RATE)}")
    if args.holdout_dir is not None:
        print(f"[wordpairs]   holdout, inside its runs     {quantiles(holdout_gaps(args.holdout_dir, args.track))}")
    print(f"[wordpairs] wrote {out_path}")


if __name__ == "__main__":
    main()
