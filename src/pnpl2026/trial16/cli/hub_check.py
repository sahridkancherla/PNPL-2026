"""Which t5 layer gives Variant K's second head different hub words?

    PYTHONPATH=src python -m pnpl2026.trial16.cli.hub_check \\
        --reference data/word_embeddings_t5_all.npz \\
        --candidates data/word_embeddings_t5_layer3.npz \\
                     data/word_embeddings_t5_layer6.npz \\
                     data/word_embeddings_t5_layer18.npz

A noisy brain-decoded vector drifts towards the centre of the target space, so
the scored words nearest that centre fill the top-10 lists of uncertain rows:
the hubs. They are set by the table's geometry, so every model trained on one
table shares them. The second head is only worth having if its table's hubs
differ from the first table's.

For each table this measures, on the 50 competition words:

    centrality    cosine of each word to the centroid of EVERY word in the
                  table (the training targets), ranked
    top-10 share  how often each word reaches the top 10 of simulated
                  uncertain predictions: the centroid direction plus isotropic
                  noise of the same norm, 20,000 draws. A fair share is 20%

and compares each candidate with the reference: Spearman correlation of the
centrality ranks, and how many of the ten most central words they share.

Rule fixed in advance: pick the candidate with the lowest Spearman that shares
at most 5 of the top 10. If none qualifies, the layers share their geometry;
use another language model's middle layer instead (see info/variant-k.md).

Reads tables only: CPU, seconds, a login node is fine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT
from ..embeddings import TargetTable

#how many scored words count as the hub list
TOP_K = 10

#simulated uncertain predictions per table
N_DRAWS = 20_000

#the most top-10 places a candidate may share with the reference and qualify
MAX_SHARED = 5


def parse_args():
    """Command-line arguments for the hub check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", default="data/word_embeddings_t5_all.npz", help="the first head's table")
    parser.add_argument("--candidates", nargs="+", required=True, help="tables the second head could use")
    parser.add_argument("--seed", type=int, default=0, help="for the simulated predictions")
    return parser.parse_args()


def unit_rows(matrix):
    """Rows scaled to unit length.

    ARGS:
        matrix (np.ndarray): [n_rows, n_dims]

    RETURNS:
        [n_rows, n_dims] float64
    """
    #axis=1 is the embedding axis: each row is one word's vector
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def hub_profile(path, rng):
    """Centrality and simulated top-10 share of the 50 scored words.

    ARGS:
        path (Path): an all-word table from build_word_embeddings
        rng (np.random.Generator)

    RETURNS:
        dict: "words" [50] str, "centrality" [50], "share" [50]
    """
    table = TargetTable(path)
    everything = unit_rows(table.table.numpy().astype(np.float64))
    candidates = everything[table.primary_rows.numpy()]

    #the centre of the training targets: where uncertain predictions drift
    centre = everything.mean(axis=0)
    centre = centre / np.linalg.norm(centre)
    centrality = candidates @ centre

    #uncertain predictions: the centre plus noise of equal norm, per draw
    n_dims = everything.shape[1]
    noise = rng.standard_normal((N_DRAWS, n_dims)) / np.sqrt(n_dims)
    predictions = unit_rows(centre[None, :] + noise)

    #[draws, 50] cosines; each draw's ten best words, axis=1 the word axis
    top = np.argsort(-(predictions @ candidates.T), axis=1)[:, :TOP_K]
    share = np.bincount(top.ravel(), minlength=len(candidates)) / N_DRAWS

    words = [table.words[row] for row in table.primary_rows.tolist()]
    return {"words": words, "centrality": centrality, "share": share}


def ranks(values):
    """Rank of each value, 0 for the largest.

    ARGS:
        values (np.ndarray): [n]

    RETURNS:
        [n] int64
    """
    order = np.argsort(-values)
    result = np.empty(len(values), dtype=np.int64)
    result[order] = np.arange(len(values))
    return result


def spearman(first, second):
    """Spearman correlation of two score vectors over the same words.

    ARGS:
        first (np.ndarray): [n]
        second (np.ndarray): [n]

    RETURNS:
        float
    """
    return float(np.corrcoef(ranks(first), ranks(second))[0, 1])


def hub_list(profile):
    """The TOP_K most central words, most central first.

    ARGS:
        profile (dict): from hub_profile

    RETURNS:
        list of str
    """
    return [profile["words"][index] for index in np.argsort(-profile["centrality"])[:TOP_K]]


def describe(name, profile):
    """One block of printed lines about a table's hubs.

    ARGS:
        name (str): the table's file name
        profile (dict): from hub_profile

    RETURNS:
        str
    """
    worst = int(np.argmax(profile["share"]))
    return (
        f"{name}\n"
        f"  most central: {', '.join(hub_list(profile))}\n"
        f"  simulated top-10 share: max {100 * profile['share'].max():.0f}% ({profile['words'][worst]}), "
        f"min {100 * profile['share'].min():.0f}%; a fair share is {100 * TOP_K / len(profile['words']):.0f}%"
    )


def main():
    """Profile every table and apply the fixed rule."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    reference_path = REPO_ROOT / args.reference
    reference = hub_profile(reference_path, rng)
    print(describe(f"reference: {reference_path.name}", reference))

    results = []
    for relative in args.candidates:
        path = REPO_ROOT / relative
        profile = hub_profile(path, rng)

        #⚠️ the same 50 words in the same column order, or the ranks compare nonsense
        if profile["words"] != reference["words"]:
            raise SystemExit(f"{path.name} orders the scored words differently from the reference.")

        rho = spearman(profile["centrality"], reference["centrality"])
        shared = len(set(hub_list(profile)) & set(hub_list(reference)))
        results.append((relative, rho, shared))
        print()
        print(describe(path.name, profile))
        print(f"  against the reference: Spearman {rho:+.2f}, {shared} of the top {TOP_K} shared")

    #the rule, fixed before any table was built
    qualifying = [result for result in results if result[2] <= MAX_SHARED]
    print()
    if qualifying:
        relative, rho, shared = min(qualifying, key=lambda result: result[1])
        print(f"[hub_check] pick {relative}: lowest Spearman ({rho:+.2f}) with {shared} <= {MAX_SHARED} shared.")
        print(f"[hub_check] set `second_embeddings: {relative}` in configs/trial16/k.yaml if it is not already.")
    else:
        print(f"[hub_check] no candidate shares {MAX_SHARED} or fewer of the top {TOP_K}: the layers share")
        print("[hub_check] their hubs. Use another language model's middle layer (info/variant-k.md).")


if __name__ == "__main__":
    main()
