"""Measure the word-pair fusion on per-word dumps before spending a submission.

    PYTHONPATH=src python -m pnpl2026.trial15.cli.fuse_check \\
        --runs runs/trial14_calibration/trial13_g1_s0 runs/trial14_calibration/trial13_g1_s1 ... \\
        [--table data/trial15_wordpairs_train.npz]

Each run directory holds `dumps/<protocol>_<partition>.npz` from an
`evaluate --dump` (trial 13's, 14's or 15's): every token of every sequence in
loader order, scored or not, so consecutive scored tokens are directly
adjacent words. The runs' ensemble is rebuilt exactly as `cli.ensemble`
averages CSVs and `cli.fuse` reads one: each run's rows z-scored, averaged,
z-scored again. Then:

    1. the temperature is fitted on the VALIDATION words of `--fit-protocol`
       (chunked by default; trial 13/15 dumps have no chunked protocol, so
       pass reference for them): the one that makes softmax(z / temperature)
       the best-calibrated probability of the true word
    2. every run of consecutive scored words is fused, and BAcc@10 is scored
       before and after on validation and test, under the chunked protocol
       (every word, ~2,700 test words) and the fragment protocol (holdout-like
       runs of 4+ scored words, 128 test words). "context" is the words with at
       least one scored neighbour, the only ones the fusion can move
    3. a setting is chosen on the same protocol's VALIDATION context words
       alone, by the rule in `choose_setting`, and the matching `cli.fuse`
       command printed
    4. each run alone, fused at its own fitted temperature: is the gain the
       same in every model, or carried by a few?

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
from ..wordpairs import N_WORDS, fuse_run, load_table, scored_runs

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


def fused_scores(standardised, primary, sequence, transitions, stationary, temperature, strength):
    """Every token's scores after fusing its run.

    ARGS:
        standardised (np.ndarray): [n_tokens, 50]
        primary (np.ndarray): [n_tokens], -1 when unscored
        sequence (np.ndarray): [n_tokens]
        transitions (np.ndarray): [50, 50]
        stationary (np.ndarray): [50]
        temperature (float)
        strength (float)

    RETURNS:
        fused : [n_tokens, 50]
        in_context : [n_tokens] bool, scored tokens with a scored neighbour
    """
    fused = standardised / temperature
    in_context = np.zeros(len(primary), dtype=bool)

    #only runs of two or more scored words carry any context
    for run in scored_runs(sequence, primary):
        if len(run) < 2:
            continue
        fused[run] = fuse_run(standardised[run], transitions, stationary, temperature, strength)
        in_context[run] = True
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


def score_setting(standardised, primary, sequence, transitions, stationary, temperature, strength):
    """BAcc@10 before and after fusion, on all scored words and on context words.

    ARGS:
        as `fused_scores`

    RETURNS:
        dict: all_before, all_after, context_before, context_after, n_scored, n_context
    """
    fused, in_context = fused_scores(standardised, primary, sequence, transitions, stationary, temperature, strength)
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


def choose_setting(validation_results):
    """The setting to submit, from validation context words alone.

    The rule, fixed before any test score is read: the highest fused
    context-word BAcc@10 on the fitting protocol's validation words; exact
    Bayes (scale 1, strength 1) unless another setting beats it by more than
    0.5 points.

    ARGS:
        validation_results (dict): (scale, strength) -> score_setting dict

    RETURNS:
        (scale, strength)
    """
    default = (1.0, 1.0)
    best = max(validation_results, key=lambda setting: validation_results[setting]["context_after"])
    margin = validation_results[best]["context_after"] - validation_results[default]["context_after"]
    return best if margin > 0.5 else default


def main():
    """Rebuild the ensemble, fit, fuse, score, and write the report."""
    args = parse_args()
    transitions, stationary, info = load_table(REPO_ROOT / args.table)
    if set(info["partitions"]) & {"validation", "test"}:
        print(f"[fuse_check] WARNING: the table counted {info['partitions']}; scores below are optimistic")

    #the fitting protocol is always scored, and scored first
    protocols = [args.fit_protocol] + [protocol for protocol in args.protocols if protocol != args.fit_protocol]

    #every protocol and partition, ensembled once
    data = {}
    for protocol in protocols:
        for partition in ("validation", "test"):
            dumps = [read_dump(run, protocol, partition) for run in args.runs]
            data[(protocol, partition)] = ensemble_dumps(dumps)

    #the temperature from the fitting protocol's validation words, every scored one
    standardised, primary, sequence = data[(args.fit_protocol, "validation")]
    scored = primary >= 0
    temperature = fit_temperature(standardised[scored], primary[scored])
    print(f"[fuse_check] fitted temperature {temperature:.4f} on {int(scored.sum())} {args.fit_protocol} validation words")

    #a fit at either end of the grid means the scores carry ~no information
    #(top) or are perfect (bottom); either way the numbers below mean little
    at_edge = temperature in (float(TEMPERATURE_GRID[0]), float(TEMPERATURE_GRID[-1]))
    if at_edge:
        print("[fuse_check] WARNING: the temperature sits at the edge of its grid; the scores look uninformative")

    #every setting on every protocol and partition
    results = {}
    for key, (standardised, primary, sequence) in data.items():
        for scale in TEMPERATURE_SCALES:
            for strength in STRENGTHS:
                results[key + (scale, strength)] = score_setting(
                    standardised, primary, sequence, transitions, stationary, temperature * scale, strength
                )
    chosen = choose_setting({(scale, strength): results[(args.fit_protocol, "validation", scale, strength)]
                             for scale in TEMPERATURE_SCALES for strength in STRENGTHS})

    lines = [
        "# Trial 15 -- the word-pair fusion, measured on per-word dumps",
        "",
        f"Ensemble of {len(args.runs)} runs: " + ", ".join(Path(run).name for run in args.runs) + ".",
        f"Table: `{args.table}` (counted on {', '.join(info['partitions'])}; smoothing {info['smoothing']:g}).",
        f"Fitted temperature {temperature:.4f}. BAcc@10 in percent, chance 20; 'context' is the words",
        "with at least one scored neighbour, the only ones the fusion can move.",
        "",
        *(["**WARNING: the temperature sits at the edge of its grid; the scores look uninformative.**", ""] if at_edge else []),
        "| protocol | partition | temp x | strength | all before | all after | context before | context after | n context / scored |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (protocol, partition, scale, strength), row in results.items():
        mark = " **(chosen)**" if (scale, strength) == chosen else ""
        lines.append(
            f"| {protocol} | {partition} | {scale:g} | {strength:g}{mark} | {row['all_before']:.2f} | {row['all_after']:.2f} | "
            f"{row['context_before']:.2f} | {row['context_after']:.2f} | {row['n_context']} / {row['n_scored']} |"
        )

    #each run alone, the fitting protocol's test words, its own temperature
    lines += ["", f"## Each run alone ({args.fit_protocol} test, context words, own temperature)", "",
              "| run | temperature | before | after | gain |", "|---|---|---|---|---|"]
    for run in args.runs:
        validation = read_dump(run, args.fit_protocol, "validation")
        test = read_dump(run, args.fit_protocol, "test")
        own_validation = zscore_rows(validation["scores"])
        own_scored = validation["primary"] >= 0
        own_temperature = fit_temperature(own_validation[own_scored], validation["primary"][own_scored])
        row = score_setting(zscore_rows(test["scores"]), test["primary"], test["sequence"], transitions,
                            stationary, own_temperature * chosen[0], chosen[1])
        lines.append(f"| {Path(run).name} | {own_temperature:.4f} | {row['context_before']:.2f} | "
                     f"{row['context_after']:.2f} | {row['context_after'] - row['context_before']:+.2f} |")

    #the command for the chosen setting
    command = (f"uv run python -m src.pnpl2026.trial15.cli.fuse --csv {args.csv} "
               f"--temperature {temperature * chosen[0]:.4f} --strength {chosen[1]:g} "
               f"--table {args.table} --holdout-dir data/holdout")
    lines += ["", f"## Chosen on {args.fit_protocol} validation", "",
              f"Temperature x{chosen[0]:g}, strength {chosen[1]:g}:", "", "```bash", command, "```", ""]

    out_path = Path(args.out) if args.out else RUNS_DIR / "trial15_fuse_check.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[fuse_check] wrote {out_path}")


if __name__ == "__main__":
    main()
