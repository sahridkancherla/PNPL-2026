"""Local, labelled checks of a transductive procedure, on held-out dumps resampled to an uneven word mix.

The Route to 73 asks every correction variant to be run "on local MOCHA,
TIMIT and Moth fragments resampled to an uneven word mix", and dropped if it
loses there. The fragment protocol holds only 128 test words, so the checks
read the reference protocol's dumps (every test word) instead, restricted to
the non-Sherlock tasks -- the subsets the 11 Oct calibration found to track
the board at all (Sherlock: a coin flip).

Each draw takes N rows with word counts ~ Multinomial(N, shares) -- by
default the holdout's own estimated shares, so the local set is as uneven
as the holdout -- sampling every word's local tokens with replacement. A
procedure is fitted on each draw ITSELF, as it is fitted on the holdout.
Seeded: the same call gives the same draws.

These checks catch damage; they do not rank recipes (no local proxy has).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR
from ..metrics import balanced_topk_accuracy

#where trial 14's calibration kept the older runs' dumps
CALIBRATION_DIR = "trial14_calibration"

#the dump key holding each token's 50 primary scores; trial 13 wrote "retrieval"
SCORE_KEYS = ("primary_logits", "retrieval")

#the scored words and the metric's k
N_WORDS = 50
K = 10


def find_dump(run, protocol, partition, runs_dir=RUNS_DIR):
    """A run's dump for one protocol and partition, or None.

    ARGS:
        run (str): a run name, or a run directory
        protocol (str), partition (str)
        runs_dir (Path)

    RETURNS:
        Path or None
    """
    name = f"{protocol}_{partition}.npz"
    run_path = Path(run)
    candidates = [run_path / "dumps" / name] if run_path.is_dir() else []
    candidates += [Path(runs_dir) / run_path.name / "dumps" / name, Path(runs_dir) / CALIBRATION_DIR / run_path.name / "dumps" / name]
    return next((path for path in candidates if path.exists()), None)


def read_dump(path):
    """One dump's scores and token labels.

    ARGS:
        path (Path)

    RETURNS:
        dict: scores [n, 50], primary [n], isolated [n] bool, task [n] str or
        None, sequence [n] int or None
    """
    with np.load(path, allow_pickle=True) as stored:
        key = next((name for name in SCORE_KEYS if name in stored.files), None)
        if key is None:
            raise ValueError(f"{path}: none of {SCORE_KEYS} in {stored.files}")
        n_tokens = len(stored["primary"])
        return {
            "scores": stored[key].astype(np.float64),
            "primary": stored["primary"].astype(np.int64),
            "isolated": stored["isolated"].astype(bool) if "isolated" in stored.files else np.zeros(n_tokens, dtype=bool),
            "task": stored["task"].astype(str) if "task" in stored.files else None,
            "sequence": stored["sequence"].astype(np.int64) if "sequence" in stored.files else None,
        }


def member_dumps(runs, protocol, partition):
    """Every member's dump of one partition, checked to list the same tokens.

    ARGS:
        runs (list): run names or directories, one per member
        protocol (str), partition (str)

    RETURNS:
        list of dict (None for a member without the dump); the tasks and
        sequences of whichever dump has them are copied to every dump
    """
    dumps = []
    for run in runs:
        path = find_dump(run, protocol, partition)
        dumps.append(read_dump(path) if path is not None else None)
    present = [dump for dump in dumps if dump is not None]
    if not present:
        return dumps

    #⚠️ the same tokens in the same order, or an average mixes words
    first = present[0]
    for dump in present[1:]:
        if not (np.array_equal(dump["primary"], first["primary"]) and np.array_equal(dump["isolated"], first["isolated"])):
            raise SystemExit(f"The members' {protocol} {partition} dumps list different tokens.")

    #tasks and sequences, from any dump that kept them
    for field in ("task", "sequence"):
        source = next((dump[field] for dump in present if dump[field] is not None), None)
        for dump in present:
            if dump[field] is None:
                dump[field] = source
    return dumps


def zscore_rows(scores):
    """Each row centred and scaled across its words.

    ARGS:
        scores (np.ndarray): [N, 50]

    RETURNS:
        [N, 50] float64
    """
    #axis=1 is the word axis; a flat row stays flat
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def ensemble(score_tables):
    """Members' scores averaged as cli.ensemble averages CSVs.

    ARGS:
        score_tables (list): [N, 50] each

    RETURNS:
        [N, 50] float64
    """
    #axis 0 is the member axis; z-scored before and after
    return zscore_rows(np.mean([zscore_rows(table) for table in score_tables], axis=0))


def bacc(scores, labels):
    """BAcc@10 over the 50 words, in percent.

    ARGS:
        scores (np.ndarray): [n, 50]
        labels (np.ndarray): [n]

    RETURNS:
        float
    """
    return 100.0 * float(balanced_topk_accuracy(torch.from_numpy(scores), torch.from_numpy(labels), K, N_WORDS))


def eligible_tokens(dump, exclude_sherlock=True):
    """Tokens a check may draw: scored, in a sentence, outside Sherlock if asked.

    ARGS:
        dump (dict): from `read_dump`
        exclude_sherlock (bool)

    RETURNS:
        [n] bool
    """
    keep = (dump["primary"] >= 0) & ~dump["isolated"]
    if exclude_sherlock and dump["task"] is not None:
        #the calibration's finding: Sherlock's local scores do not track the board
        keep &= ~np.char.startswith(dump["task"].astype(str), "Sherlock")
    return keep


def uneven_draws(labels, eligible, shares, n_rows=868, n_draws=200, seed=0):
    """Token indices for each draw, with word counts as uneven as `shares`.

    ARGS:
        labels (np.ndarray): [n] word of each token
        eligible (np.ndarray): [n] bool, tokens that may be drawn
        shares (np.ndarray): [50] target word shares; words with no eligible
            token are left out and the rest renormalised
        n_rows (int): rows per draw, the holdout's 868 by default
        n_draws (int)
        seed (int)

    RETURNS:
        list of [n_rows] int arrays
    """
    generator = np.random.default_rng(seed)
    pools = {word: np.flatnonzero(eligible & (labels == word)) for word in range(N_WORDS)}
    available = np.array([len(pools[word]) > 0 for word in range(N_WORDS)])
    weights = np.where(available, shares, 0.0)
    weights = weights / weights.sum()
    draws = []
    for _ in range(n_draws):
        #how many rows each word gets, then which of its tokens fill them
        counts = generator.multinomial(n_rows, weights)
        chosen = [generator.choice(pools[word], size=count, replace=True) for word, count in enumerate(counts) if count > 0]
        draws.append(np.concatenate(chosen))
    return draws


def summarise_gains(gains):
    """Mean, spread and win rate of a variant's gains over draws.

    ARGS:
        gains (list): per draw, variant minus baseline, in BAcc points

    RETURNS:
        str
    """
    gains = np.asarray(gains, dtype=np.float64)
    return f"{gains.mean():+.2f} +/- {gains.std():.2f}, wins {100 * np.mean(gains > 0):.0f}%"
