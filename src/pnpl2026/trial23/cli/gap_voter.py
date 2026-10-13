"""Fit the onset-gap voter on training speech, re-fit it on the holdout, write it as a CSV.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.gap_voter \\
        --mix submissions/ens_t22_tc1b75_ts_hubh.csv --out submissions/gap_voter_ts.csv \\
        --holdout-dir data/holdout

Change 6 of the Route to 73 (`gaps.py` has the model). Steps:

    1   training: every subject-0 TRAIN recording's events table, at 250 Hz
        (`timeds`, exact), chained as the holdout lists words; one Gaussian
        of the log gap ratio per word. Saved to data/trial23_gap_model.npz
        and reused; `--refit-training` rebuilds it
    2   the standalone check: gap-only BAcc@10 on the validation and test
        recordings' chains, with labels (chance 20)
    3   the holdout: every sentence row's log gap ratio; EM with the mix's
        soft labels (temperature by the confidence rule) re-fits the
        Gaussians; each row scored log N(x | word)
    4   the label-free check: does the gap track the typical gap of the
        mix's top-1 word? A weak link means the mix is not using the cue

The CSV's sentence rows with a gap carry the voter; every other row is flat
and so adds nothing when mixed. Join it to a mix at a low weight:

    cli.ensemble --group REST=<mix>.csv --group GAP=gap_voter_ts.csv --weights REST=23,GAP=1

Needs the events tables on disk (data/<task>/derivatives/events/), as
`cli.prepare` downloaded them. Minutes on a login node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ...config import DATA_DIR, REPO_ROOT, RUNS_DIR
from ..data_processing.build_reference import discover_recordings, load_word_table, pnpl_partitions
from ..gaps import GapModel, holdout_features, table_features
from ..local_checks import bacc
from ..transduction import fit_temperature_confidence, softmax_rows
from .ensemble import read_submission, standardise_rows

#the events tables' own timeline, exact on the 250 Hz grid
EVENTS_RATE = 250.0


def parse_args():
    """Command-line arguments for the gap voter."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mix", required=True, help="the best mix, hub-corrected: its soft labels drive the EM")
    parser.add_argument("--out", required=True, help="the voter CSV")
    parser.add_argument("--model", default="data/trial23_gap_model.npz", help="repo-relative fitted training model")
    parser.add_argument("--refit-training", action="store_true", help="rebuild the training model from the events")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root holding the task folders")
    parser.add_argument("--confidence", type=float, default=0.3, help="the mix's mean top-1 probability after tempering")
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--track", default="deep")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def partition_features(data_dir):
    """The chained log gap ratios of every subject-0 recording, by pnpl partition.

    ARGS:
        data_dir (str)

    RETURNS:
        dict: partition -> (features [m], labels [m])
    """
    partitions = pnpl_partitions()
    collected = {}
    recordings = discover_recordings(Path(data_dir))
    if not recordings:
        raise SystemExit(f"No recordings under {data_dir}: the events tables are needed.")
    for record in recordings:
        partition = partitions.get((record["session"], record["task"], record["run"]))
        if partition is None:
            continue
        #the onsets at 250 Hz: timeds, the downsampled timeline, exact on the grid
        table = load_word_table(record["events_path"], sfreq=EVENTS_RATE)
        features, labels = table_features(table, EVENTS_RATE)
        collected.setdefault(partition, ([], []))
        collected[partition][0].append(features)
        collected[partition][1].append(labels)
    return {name: (np.concatenate(parts[0]), np.concatenate(parts[1])) for name, parts in collected.items()}


def training_model(args, lines):
    """The training Gaussians, fitted once and saved, with the standalone check.

    ARGS:
        args (argparse.Namespace)
        lines (list): report lines, appended to

    RETURNS:
        GapModel
    """
    path = REPO_ROOT / args.model
    if path.exists() and not args.refit_training:
        with np.load(path) as stored:
            lines.append(f"Training model read from `{args.model}` ({int(stored['counts'].sum())} tokens).")
            return GapModel(stored["means"], stored["spreads"], stored["counts"])

    by_partition = partition_features(args.data_dir)
    features, labels = by_partition["train"]
    model = GapModel.fit(features, labels)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, means=model.means, spreads=model.spreads, counts=model.counts)
    lines.append(f"Training model fitted on {len(features):,} chained train tokens; saved to `{args.model}`.")

    #the standalone check: the gap alone, scored with labels (chance 20)
    for partition in ("validation", "test"):
        if partition in by_partition and len(by_partition[partition][0]):
            held_x, held_y = by_partition[partition]
            lines.append(f"- gap alone on {partition} chains ({len(held_x):,} tokens): BAcc@10 "
                         f"{bacc(model.log_likelihood(held_x), held_y):.2f} (chance 20)")
    return model


def main():
    """Fit or read the training model, re-fit on the holdout, write the voter and the report."""
    args = parse_args()
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary, write_submission

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    lines = ["# Trial 22 -- the onset-gap voter", ""]
    model = training_model(args, lines)
    vocabulary = list(load_vocabulary("primary"))
    order = np.argsort(model.means)
    lines.append("Shortest typical gap: " + ", ".join(vocabulary[word] for word in order[:6])
                 + "; longest: " + ", ".join(vocabulary[word] for word in order[-6:]) + ".")

    #the holdout rows that have a gap, and the mix's soft labels for them
    metadata = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir).metadata
    features = holdout_features(metadata)
    voting = np.isfinite(features)
    index, primary, _ = read_submission(Path(args.mix), load_vocabulary("primary"), load_vocabulary("moses"))
    if not np.array_equal(index, np.arange(len(metadata))):
        raise SystemExit(f"{args.mix}: its rows are not the holdout's 0..{len(metadata) - 1}.")
    mix = standardise_rows(primary)
    temperature = fit_temperature_confidence(mix[voting], args.confidence)
    soft = softmax_rows(mix[voting] / temperature)

    #EM on the holdout's own rows, anchored to the training fit
    holdout_model, history = model.refit(features[voting], soft)
    scores = holdout_model.log_likelihood(features[voting])
    lines += ["", f"Holdout: {int(voting.sum())} of {len(metadata)} rows have a gap (not a sentence's last word, "
              f"sentences of 3+ words). EM log-likelihood {history[0]:.1f} -> {history[-1]:.1f}; mean shift of the "
              f"word means {np.abs(holdout_model.means - model.means).mean():.3f}."]

    #the label-free check: the gap against the typical gap of the mix's top-1 word
    top = mix[voting].argmax(axis=1)
    rho = spearmanr(features[voting], model.means[top]).correlation
    lines.append(f"Label-free: Spearman of each row's gap with its top-1 word's typical gap {rho:+.3f} "
                 "(near 0: the mix is not using the cue).")

    #the CSV: voting rows the gap's distribution, every other row flat
    table = np.full((len(metadata), len(vocabulary)), 1.0 / len(vocabulary))
    table[voting] = softmax_rows(scores)
    write_submission(str(out_path), indices=np.arange(len(metadata)), primary_probs=table, secondary_probs=np.full_like(table, 1.0 / len(vocabulary)))
    report = RUNS_DIR / f"trial23_gap_voter_{out_path.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[gap_voter] wrote {out_path} and {report}")


if __name__ == "__main__":
    main()
