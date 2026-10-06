"""Measure the word-pair fusion on per-word dumps before spending a submission.

    PYTHONPATH=src python -m pnpl2026.trial15.cli.fuse_check \\
        --runs runs/trial14_calibration/trial13_g1_s0 runs/trial14_calibration/trial13_g1_s1 ... \\
        [--table data/trial15_wordpairs_train.npz] [--csv submissions/<ensemble>.csv]

Each run directory holds `dumps/<protocol>_<partition>.npz` from an
`evaluate --dump` (trial 13's, 14's or 15's): every token of every sequence in
loader order, scored or not, so the dump says which scored words are directly
adjacent and which have unscored words between them. The runs' ensemble is
rebuilt exactly as `cli.ensemble` averages CSVs and `cli.fuse` reads one: each
run's rows z-scored, averaged, z-scored again. Then:

    1. the temperature is fitted on the VALIDATION words of `--fit-protocol`
       (chunked by default; trial 13/15 dumps have no chunked protocol, so
       pass reference for them): the one that makes softmax(z / temperature)
       the best-calibrated probability of the true word
    2. section A, directly adjacent runs only -- what the first fusion assumed
    3. section B, runs that may skip ONE unscored word, as about a third of the
       holdout's pairs appear to. Each pair gets an onset gap drawn from the
       table's gap model of its true kind, and the gap mixture is fitted on
       those gaps exactly as `cli.fuse` fits it on the holdout's. Scored with
       no fusion, the adjacency-only chain (the failure seen on the
       leaderboard), the gap-aware chain, and an oracle told each pair's kind
    4. a gap-aware setting is chosen on section B's fitting-protocol
       VALIDATION context words alone, by the rule in `choose_setting`, and
       the matching `cli.fuse` command printed
    5. each run alone, gap-aware at the chosen setting: is the gain the same
       in every model, or carried by a few?

"context" is the words with at least one scored neighbour in their run, the
only ones a fusion can move.

⚠️ The table must not have counted the validation or test transcripts
(`cli.wordpairs --partitions train`, the default), or it has seen the word
sequences it is judged on.

Writes runs/trial15_fuse_check.md. Light: numpy over saved arrays, seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ...config import REPO_ROOT, RUNS_DIR
from ...metrics import balanced_topk_accuracy
from ..wordpairs import N_WORDS, adjacent_weight, fit_gap_mixture, fuse_run, load_table, scored_chains, step_transitions

#BAcc@10 is the scored metric
K = 10

#the dump key holding each token's 50 primary scores; trial 13 wrote "retrieval"
SCORE_KEYS = ("primary_logits", "retrieval")

#temperatures tried when fitting: z-scores are O(1), so this spans every
#plausible calibration with room on both sides
TEMPERATURE_GRID = np.geomspace(0.02, 5.0, 400)

#settings compared: temperature as a multiple of the fitted one, and strength
TEMPERATURE_SCALES = (0.5, 1.0, 2.0)
STRENGTHS = (0.5, 1.0)

#the simulated gaps are drawn from this seed, so every setting sees the same
GAP_SEED = 0


def parse_args():
    """Command-line arguments for the local check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="directories holding dumps/")
    parser.add_argument("--table", default="data/trial15_wordpairs_train.npz", help="repo-relative word-pair table")
    parser.add_argument("--protocols", nargs="+", default=["chunked", "fragment"], help="dumps to score")
    parser.add_argument(
        "--fit-protocol",
        default="chunked",
        help="protocol whose validation words fit the temperature and choose the setting; "
        "'reference' for trial 13/15 dumps, which have no chunked protocol",
    )
    parser.add_argument("--csv", default="<CSV>", help="the submission the printed fuse command names")
    parser.add_argument("--out", default=None, help="default runs/trial15_fuse_check.md")
    return parser.parse_args()


def read_dump(run_dir, protocol, partition):
    """One run's dump for one protocol and partition.

    ARGS:
        run_dir (Path): holds dumps/
        protocol (str): "chunked", "fragment", ...
        partition (str): "validation" or "test"

    RETURNS:
        dict: scores [n_tokens, 50], primary [n_tokens], sequence [n_tokens]
    """
    path = Path(run_dir) / "dumps" / f"{protocol}_{partition}.npz"
    with np.load(path) as stored:
        key = next((name for name in SCORE_KEYS if name in stored.files), None)
        if key is None:
            raise SystemExit(f"{path}: none of {SCORE_KEYS} in {stored.files}")
        return {
            "scores": stored[key].astype(np.float64),
            "primary": stored["primary"].astype(np.int64),
            "sequence": stored["sequence"].astype(np.int64),
        }


def zscore_rows(scores):
    """Each row z-scored across its 50 words; a flat row stays flat.

    ARGS:
        scores (np.ndarray): [n, 50]

    RETURNS:
        [n, 50] float64
    """
    #axis=1 is the word axis: each row is ranked on its own
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def ensemble_dumps(dumps):
    """The runs' ensemble, z-scored as cli.fuse reads an ensemble CSV.

    ARGS:
        dumps (list of dict): from `read_dump`, the same tokens in the same order

    RETURNS:
        standardised : [n_tokens, 50]
        primary : [n_tokens]
        sequence : [n_tokens]
    """
    #⚠️ the same tokens in the same order, or the average mixes words
    first = dumps[0]
    for dump in dumps[1:]:
        if not (np.array_equal(dump["primary"], first["primary"]) and np.array_equal(dump["sequence"], first["sequence"])):
            raise SystemExit("The runs' dumps do not list the same tokens in the same order.")

    #axis 0 is the run axis; z-scored again, as standardise_rows does to log softmax
    average = np.mean([zscore_rows(dump["scores"]) for dump in dumps], axis=0)
    return zscore_rows(average), first["primary"], first["sequence"]


def fit_temperature(standardised, targets):
    """The temperature that best calibrates softmax(z / temperature).

    ARGS:
        standardised (np.ndarray): [n, 50] z-scores of scored tokens
        targets (np.ndarray): [n] true columns

    RETURNS:
        float
    """
    best, best_loss = None, np.inf
    for temperature in TEMPERATURE_GRID:
        #mean negative log-probability of the true word; axis=1 is the words
        scaled = standardised / temperature
        log_norm = np.log(np.exp(scaled - scaled.max(axis=1, keepdims=True)).sum(axis=1)) + scaled.max(axis=1)
        loss = float(np.mean(log_norm - scaled[np.arange(len(targets)), targets]))
        if loss < best_loss:
            best, best_loss = float(temperature), loss
    return best


def prepare_chains(primary, sequence, table, max_skipped, seed=GAP_SEED):
    """The chains to fuse, each pair with a simulated gap and its weights.

    ARGS:
        primary (np.ndarray): [n_tokens], -1 when unscored
        sequence (np.ndarray): [n_tokens]
        table (dict): from `load_table`
        max_skipped (int): 0 for adjacent runs, 1 to allow one unscored word
        seed (int): for the simulated gaps

    RETURNS:
        chains : list of dict: tokens, gap_weights, oracle_weights
        mixture : dict: shift, share fitted on the simulated gaps, and the
            true adjacent share
    """
    chains = [(tokens, skips) for tokens, skips in scored_chains(sequence, primary, max_skipped) if len(tokens) >= 2]
    generator = np.random.default_rng(seed)

    #each pair's gap from the log-normal of its true kind, as the holdout's
    #onsets would show it
    gaps = []
    for _, skips in chains:
        kinds = [table["gap_models"]["adjacent" if skip == 0 else "skip"] for skip in skips]
        gaps.append(np.exp(np.array([generator.normal(mean, spread) for mean, spread in kinds])))
    all_gaps = np.concatenate(gaps) if gaps else np.zeros(0)
    all_skips = np.concatenate([skips for _, skips in chains]) if chains else np.zeros(0)

    #the mixture fitted on the gaps alone, as cli.fuse fits it on the holdout's
    shift, share = fit_gap_mixture(all_gaps, table["gap_models"]) if len(all_gaps) else (0.0, 1.0)
    prepared = [
        {
            "tokens": tokens,
            "gap_weights": adjacent_weight(chain_gaps, table["gap_models"], shift, share),
            "oracle_weights": (skips == 0).astype(np.float64),
        }
        for (tokens, skips), chain_gaps in zip(chains, gaps)
    ]
    mixture = {"shift": shift, "share": share, "true_share": float(np.mean(all_skips == 0)) if len(all_skips) else 1.0}
    return prepared, mixture


def fuse_chains(standardised, chains, table, temperature, strength, method):
    """Every token's scores after fusing every chain one way.

    ARGS:
        standardised (np.ndarray): [n_tokens, 50]
        chains (list of dict): from `prepare_chains`
        table (dict): from `load_table`
        temperature (float)
        strength (float)
        method (str): "adjacent" (every step adjacent), "gap" or "oracle"

    RETURNS:
        fused : [n_tokens, 50]
        in_context : [n_tokens] bool, tokens of a chain of two or more
    """
    fused = standardised / temperature
    in_context = np.zeros(len(standardised), dtype=bool)
    for chain in chains:
        tokens = chain["tokens"]

        #one matrix for every step, or one per step mixed by its weight
        steps = table["transitions"]
        if method != "adjacent":
            steps = step_transitions(chain[f"{method}_weights"], table["transitions"], table["skip_transitions"])
        fused[tokens] = fuse_run(standardised[tokens], steps, table["stationary"], temperature, strength)
        in_context[tokens] = True
    return fused, in_context


def bacc(scores, targets):
    """BAcc@10 over the 50 words, in percent.

    ARGS:
        scores (np.ndarray): [n, 50]
        targets (np.ndarray): [n]

    RETURNS:
        float
    """
    if len(targets) == 0:
        return float("nan")
    return 100.0 * balanced_topk_accuracy(torch.from_numpy(scores), torch.from_numpy(targets), K, N_WORDS)


def score_method(standardised, primary, chains, table, temperature, strength, method):
    """BAcc@10 before and after one fusion, on all scored words and on context words.

    ARGS:
        as `fuse_chains`, plus primary [n_tokens]

    RETURNS:
        dict: all_before, all_after, context_before, context_after, n_scored, n_context
    """
    fused, in_context = fuse_chains(standardised, chains, table, temperature, strength, method)
    scored = primary >= 0
    context = in_context & scored
    return {
        "all_before": bacc(standardised[scored], primary[scored]),
        "all_after": bacc(fused[scored], primary[scored]),
        "context_before": bacc(standardised[context], primary[context]),
        "context_after": bacc(fused[context], primary[context]),
        "n_scored": int(scored.sum()),
        "n_context": int(context.sum()),
    }


def choose_setting(context_after):
    """The setting to submit, from validation context words alone.

    The rule, fixed before any test score is read: the highest gap-aware
    context-word BAcc@10 on the fitting protocol's validation words, section
    B; exact Bayes (scale 1, strength 1) unless another setting beats it by
    more than 0.5 points.

    ARGS:
        context_after (dict): (scale, strength) -> fused context BAcc@10

    RETURNS:
        (scale, strength)
    """
    default = (1.0, 1.0)
    best = max(context_after, key=lambda setting: context_after[setting])
    return best if context_after[best] - context_after[default] > 0.5 else default


def main():
    """Rebuild the ensemble, fit, fuse both ways, score, and write the report."""
    args = parse_args()
    table = load_table(REPO_ROOT / args.table)
    if set(table["partitions"]) & {"validation", "test"}:
        print(f"[fuse_check] WARNING: the table counted {table['partitions']}; scores below are optimistic")

    #the fitting protocol is always scored, and scored first
    protocols = [args.fit_protocol] + [protocol for protocol in args.protocols if protocol != args.fit_protocol]

    #every protocol and partition, ensembled once, with both kinds of chain
    data = {}
    for protocol in protocols:
        for partition in ("validation", "test"):
            standardised, primary, sequence = ensemble_dumps([read_dump(run, protocol, partition) for run in args.runs])
            data[(protocol, partition)] = {
                "standardised": standardised,
                "primary": primary,
                "adjacent": prepare_chains(primary, sequence, table, max_skipped=0),
                "skip": prepare_chains(primary, sequence, table, max_skipped=1),
            }

    #the temperature from the fitting protocol's validation words, every scored one
    fit = data[(args.fit_protocol, "validation")]
    scored = fit["primary"] >= 0
    temperature = fit_temperature(fit["standardised"][scored], fit["primary"][scored])
    print(f"[fuse_check] fitted temperature {temperature:.4f} on {int(scored.sum())} {args.fit_protocol} validation words")

    #a fit at either end of the grid means the scores carry ~no information
    #(top) or are perfect (bottom); either way the numbers below mean little
    at_edge = temperature in (float(TEMPERATURE_GRID[0]), float(TEMPERATURE_GRID[-1]))
    if at_edge:
        print("[fuse_check] WARNING: the temperature sits at the edge of its grid; the scores look uninformative")

    #section B first: it chooses the setting
    settings = [(scale, strength) for scale in TEMPERATURE_SCALES for strength in STRENGTHS]
    section_b = {}
    for key, entry in data.items():
        chains, _ = entry["skip"]
        for scale, strength in settings:
            section_b[key + (scale, strength)] = {
                method: score_method(entry["standardised"], entry["primary"], chains, table, temperature * scale, strength, method)
                for method in ("adjacent", "gap", "oracle")
            }
    chosen = choose_setting({setting: section_b[(args.fit_protocol, "validation") + setting]["gap"]["context_after"]
                             for setting in settings})

    lines = [
        "# Trial 15 -- the word-pair fusion, measured on per-word dumps",
        "",
        f"Ensemble of {len(args.runs)} runs: " + ", ".join(Path(run).name for run in args.runs) + ".",
        f"Table: `{args.table}` (counted on {', '.join(table['partitions'])}; smoothing "
        f"{table['smoothing']:g} adjacent, {table['skip_smoothing']:g} skip-one).",
        f"Fitted temperature {temperature:.4f}. BAcc@10 in percent, chance 20; 'context' is the words",
        "with at least one scored neighbour in their run, the only ones a fusion can move.",
        "",
        *(["**WARNING: the temperature sits at the edge of its grid; the scores look uninformative.**", ""] if at_edge else []),
        "## B. Runs that may skip one unscored word -- the holdout's apparent structure",
        "",
        "Context-word BAcc@10. `adjacent-only` treats every pair as adjacent (as first submitted);",
        "`gap-aware` weighs each pair by its simulated onset gap; `oracle` is told each pair's kind.",
        "",
    ]
    for key, entry in data.items():
        _, mixture = entry["skip"]
        lines.append(f"- {key[0]} {key[1]}: gap mixture fitted x{np.exp(mixture['shift']):.2f}, "
                     f"adjacent share {mixture['share']:.2f} (true {mixture['true_share']:.2f})")
    lines += ["", "| protocol | partition | temp x | strength | before | adjacent-only | gap-aware | oracle | n context / scored |",
              "|---|---|---|---|---|---|---|---|---|"]
    for (protocol, partition, scale, strength), methods in section_b.items():
        mark = " **(chosen)**" if (scale, strength) == chosen else ""
        gap = methods["gap"]
        lines.append(
            f"| {protocol} | {partition} | {scale:g} | {strength:g}{mark} | {gap['context_before']:.2f} | "
            f"{methods['adjacent']['context_after']:.2f} | {gap['context_after']:.2f} | {methods['oracle']['context_after']:.2f} | "
            f"{gap['n_context']} / {gap['n_scored']} |"
        )

    #section A: directly adjacent runs only, the first fusion's own setting
    lines += ["", "## A. Directly adjacent runs only -- what the first fusion assumed", "",
              "| protocol | partition | temp x | strength | all before | all after | context before | context after | n context / scored |",
              "|---|---|---|---|---|---|---|---|---|"]
    for key, entry in data.items():
        chains, _ = entry["adjacent"]
        for scale, strength in settings:
            row = score_method(entry["standardised"], entry["primary"], chains, table, temperature * scale, strength, "adjacent")
            lines.append(
                f"| {key[0]} | {key[1]} | {scale:g} | {strength:g} | {row['all_before']:.2f} | {row['all_after']:.2f} | "
                f"{row['context_before']:.2f} | {row['context_after']:.2f} | {row['n_context']} / {row['n_scored']} |"
            )

    #each run alone, section B's chains, gap-aware at the chosen setting
    lines += ["", f"## Each run alone ({args.fit_protocol} test, section B, gap-aware, own temperature)", "",
              "| run | temperature | before | after | gain |", "|---|---|---|---|---|"]
    for run in args.runs:
        validation = read_dump(run, args.fit_protocol, "validation")
        test = read_dump(run, args.fit_protocol, "test")
        own_validation = zscore_rows(validation["scores"])
        own_scored = validation["primary"] >= 0
        own_temperature = fit_temperature(own_validation[own_scored], validation["primary"][own_scored])
        chains, _ = prepare_chains(test["primary"], test["sequence"], table, max_skipped=1)
        row = score_method(zscore_rows(test["scores"]), test["primary"], chains, table,
                           own_temperature * chosen[0], chosen[1], "gap")
        lines.append(f"| {Path(run).name} | {own_temperature:.4f} | {row['context_before']:.2f} | "
                     f"{row['context_after']:.2f} | {row['context_after'] - row['context_before']:+.2f} |")

    #the command for the chosen setting
    command = (f"uv run python -m src.pnpl2026.trial15.cli.fuse --csv {args.csv} "
               f"--temperature {temperature * chosen[0]:.4f} --strength {chosen[1]:g} --transitions gap "
               f"--table {args.table} --holdout-dir data/holdout")
    lines += ["", f"## Chosen on {args.fit_protocol} validation, section B, gap-aware", "",
              f"Temperature x{chosen[0]:g}, strength {chosen[1]:g}:", "", "```bash", command, "```", ""]

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial15_fuse_check.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[fuse_check] wrote {out_path}")


if __name__ == "__main__":
    main()
