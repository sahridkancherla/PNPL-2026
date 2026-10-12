"""A virtual subject 0: the other subjects' recordings of its sentences, mapped into its sensor space.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.align --holdout-dir data/holdout \\
        [--subjects 1-39] [--mode mcca|raw] [--components auto] [--name mcca]

Change 1 of the Route to 73, route 2 (`crosssubject.py` has the method).

    mcca   multiway CCA over subject 0 and the chosen subjects, fitted on
           the matched holdout sentences at 50 Hz; the number of shared
           components C chosen label-free by K-fold over sentences: how well
           the others' mapped average predicts subject 0's held-out MEG.
           Each subject is weighted by its own held-out correlation
    raw    the others' raw epochs averaged as they are, no mapping: the
           baseline the alignment has to beat

Writes data/trial22_virtual/<name>/COMPETITION_HOLDOUT/, a holdout shaped
exactly as subject 0's -- its metadata copied, so every row keeps its index
-- whose matched sentences hold the virtual MEG and whose unmatched ones
(the two one-word sentences) and lone words are subject 0's own; plus
matched.json and runs/trial22_align_<name>.md. Every member then predicts it
with `cli.predict --holdout-dir data/trial22_virtual/<name>`, unchanged.

Heavy for a login node: every subject's sentence file is read twice (39 x
~200 MB). Run it as `pbs/trial22_align.pbs`, about 15-30 minutes on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR
from ..crosssubject import (
    MODEL_RATE,
    SubjectMap,
    Whitener,
    channel_correlation,
    common_lengths,
    fit_forward_map,
    fit_mcca,
    holdout_file,
    load_matches,
    read_sentences,
    sentence_file,
    stacked_epochs,
    write_virtual,
)
from .repeat_check import parse_subjects


def parse_args():
    """Command-line arguments for the alignment."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--holdout-dir", default="data/holdout", help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--matches", default="data/trial22_matches.json", help="repo-relative, from cli.repeat_check")
    parser.add_argument("--subjects", default="1-39", help="the other subjects mapped in")
    parser.add_argument("--mode", choices=("mcca", "raw"), default="mcca")
    parser.add_argument("--pcs", type=int, default=60, help="per-subject PCA components, below the SSS rank")
    parser.add_argument("--grid", default="5,10,20,30,40,60,80", help="shared-component counts tried")
    parser.add_argument("--components", default="auto", help="'auto' (K-fold choice) or a count")
    parser.add_argument("--folds", type=int, default=5, help="sentence folds for the choice")
    parser.add_argument("--name", default=None, help="default the mode's name")
    parser.add_argument("--out-root", default="data/trial22_virtual", help="repo-relative")
    return parser.parse_args()


def load_stacks(holdout_dir, subjects, matches, epoch_order, lengths):
    """Every subject's matched epochs at 50 Hz, scaled, in subject 0's order.

    ARGS:
        holdout_dir (str)
        subjects (list): 0 first, then the others
        matches (dict): from `load_matches`
        epoch_order (list): subject 0's epochs every subject matched
        lengths (dict): from `common_lengths`

    RETURNS:
        stacks : dict subject -> [306, T] scaled
        scalers : dict subject -> (median, iqr)
        bounds : [n_epochs + 1] where each sentence starts in a stack
    """
    stacks, scalers, bounds = {}, {}, None
    for subject in subjects:
        #one subject's MEG in memory at a time; its stack is a fifth of the rate
        data = read_sentences(sentence_file(holdout_dir, subject))
        pairs = None if subject == 0 else matches[subject]
        stacked, median, iqr, bounds = stacked_epochs(data, epoch_order, lengths, pairs)
        stacks[subject] = stacked.astype(np.float32)
        scalers[subject] = (median, iqr)
        print(f"[align] subject {subject}: {stacked.shape[1]} samples at {MODEL_RATE} Hz")
        del data
    return stacks, scalers, bounds


def select(stack, bounds, epochs):
    """The columns of a stack that belong to some sentences.

    ARGS:
        stack (np.ndarray): [306, T]
        bounds (np.ndarray): sentence starts, from `load_stacks`
        epochs (np.ndarray): sentence positions in the epoch order

    RETURNS:
        [306, T_selected] float64
    """
    return np.concatenate([stack[:, bounds[index] : bounds[index + 1]] for index in epochs], axis=1).astype(np.float64)


def cross_validate(stacks, bounds, subjects, grid, n_pcs, n_folds):
    """Held-out correlation of the mapped average with subject 0, per component count.

    ARGS:
        stacks (dict): from `load_stacks`
        bounds (np.ndarray)
        subjects (list): 0 first
        grid (list): component counts
        n_pcs (int)
        n_folds (int)

    RETURNS:
        average : dict C -> mean held-out correlation of the others' average
        per_subject : dict C -> dict subject -> mean held-out correlation
        raw : float, the unmapped average's held-out correlation
    """
    n_epochs = len(bounds) - 1
    average = {count: [] for count in grid}
    per_subject = {count: {subject: [] for subject in subjects[1:]} for count in grid}
    raw = []
    for fold in range(n_folds):
        #interleaved folds, so every fold spans the whole file
        test = np.arange(fold, n_epochs, n_folds)
        train = np.setdiff1d(np.arange(n_epochs), test)

        #whiteners and the MCCA fitted on the training sentences only
        whiteners = {subject: Whitener(select(stacks[subject], bounds, train), n_pcs) for subject in subjects}
        train_white = [whiteners[subject](select(stacks[subject], bounds, train)) for subject in subjects]
        blocks, _ = fit_mcca(train_white, max(grid))
        reference_test = select(stacks[0], bounds, test)
        test_white = {subject: whiteners[subject](select(stacks[subject], bounds, test)) for subject in subjects[1:]}

        #the unmapped baseline: the others' scaled sensors averaged as they are
        raw.append(channel_correlation(np.mean([select(stacks[subject], bounds, test) for subject in subjects[1:]], axis=0), reference_test))
        for count in grid:
            #subject 0's map from its own shared-component estimate, training only
            weights, intercept = fit_forward_map(select(stacks[0], bounds, train), blocks[0][:, :count].T @ train_white[0])
            mapped = {}
            for index, subject in enumerate(subjects[1:], start=1):
                shared = blocks[index][:, :count].T @ test_white[subject]
                mapped[subject] = weights @ shared + intercept[:, None]
                per_subject[count][subject].append(channel_correlation(mapped[subject], reference_test))
            average[count].append(channel_correlation(np.mean(list(mapped.values()), axis=0), reference_test))
        print(f"[align] fold {fold + 1}/{n_folds}: " + ", ".join(f"C={count} {average[count][-1]:.3f}" for count in grid))
    return ({count: float(np.mean(values)) for count, values in average.items()},
            {count: {subject: float(np.mean(values)) for subject, values in table.items()} for count, table in per_subject.items()},
            float(np.mean(raw)))


def fit_maps(stacks, scalers, subjects, n_pcs, n_components):
    """Every other subject's raw-to-raw map into subject 0, fitted on all matched sentences.

    ARGS:
        stacks (dict), scalers (dict): from `load_stacks`
        subjects (list): 0 first
        n_pcs (int), n_components (int)

    RETURNS:
        maps : dict subject -> SubjectMap
        eigenvalues : [C]
    """
    whiteners = {subject: Whitener(stacks[subject].astype(np.float64), n_pcs) for subject in subjects}
    whitened = [whiteners[subject](stacks[subject].astype(np.float64)) for subject in subjects]
    blocks, eigenvalues = fit_mcca(whitened, n_components)

    #subject 0's sensors from its own estimate of the shared components
    weights, intercept = fit_forward_map(stacks[0].astype(np.float64), blocks[0].T @ whitened[0])
    reference_median, reference_iqr = scalers[0]
    maps = {}
    for index, subject in enumerate(subjects[1:], start=1):
        median, iqr = scalers[subject]
        maps[subject] = SubjectMap(median, iqr, whiteners[subject], blocks[index], weights, intercept, reference_median, reference_iqr)
    return maps, eigenvalues


def build_virtual(holdout_dir, reference, subjects, matches, epoch_order, maps, weights):
    """The virtual epochs: each matched sentence the weighted mean of the others' mapped recordings.

    ARGS:
        holdout_dir (str)
        reference (dict): subject 0's data, from `read_sentences`
        subjects (list): the others
        matches (dict)
        epoch_order (list): subject 0's matched epochs
        maps (dict or None): subject -> SubjectMap; None averages raw epochs
        weights (dict): subject -> weight, summing to 1

    RETURNS:
        [n, 306, T_max] float32, subject 0's shape; unmatched epochs its own
    """
    virtual = reference["meg"].astype(np.float64).copy()
    matched = set(epoch_order)
    for epoch in matched:
        virtual[epoch] = 0.0
    for subject in subjects:
        data = read_sentences(sentence_file(holdout_dir, subject))
        own = dict(matches[subject])
        for epoch in epoch_order:
            #the other subject's epoch of this sentence, at subject 0's length
            length = int(reference["n_times"][epoch])
            raw = data["meg"][own[epoch], :, : int(data["n_times"][own[epoch]])].astype(np.float64)
            if raw.shape[1] < length:
                raw = np.pad(raw, ((0, 0), (0, length - raw.shape[1])), mode="edge")
            mapped = maps[subject](raw[:, :length]) if maps is not None else raw[:, :length]
            virtual[epoch, :, :length] += weights[subject] * mapped
        print(f"[align] subject {subject} mapped in (weight {weights[subject]:.4f})")
        del data
    return virtual.astype(np.float32)


def main():
    """Fit, choose, map, write the virtual holdout and the report."""
    args = parse_args()
    name = args.name or args.mode
    matches = load_matches(REPO_ROOT / args.matches)
    others = [subject for subject in parse_subjects(args.subjects) if subject != 0 and matches.get(subject)]
    if not others:
        raise SystemExit("No chosen subject has matched sentences: run cli.repeat_check first.")
    subjects = [0] + others

    #the sentences every chosen subject matched, in subject 0's epoch order
    shared = set.intersection(*[{pair[0] for pair in matches[subject]} for subject in others])
    epoch_order = sorted(shared)
    reference_path = sentence_file(args.holdout_dir, 0)
    reference = read_sentences(reference_path)
    others_data = {subject: {"n_times": read_sentences_n_times(args.holdout_dir, subject)} for subject in others}
    lengths = common_lengths(others_data, {subject: matches[subject] for subject in others}, reference)
    print(f"[align] {len(others)} subjects, {len(epoch_order)} sentences matched by all; mode {args.mode}")

    lines = [f"# Trial 22 -- virtual subject 0 `{name}` ({args.mode})", "",
             f"{len(others)} other subjects; {len(epoch_order)} of {len(reference['n_times'])} sentences matched by all.", ""]
    stacks, scalers, bounds = load_stacks(args.holdout_dir, subjects, matches, epoch_order, lengths)
    grid = [int(value) for value in args.grid.split(",")]

    #the label-free choice: held-out sentences, subject 0's own MEG as the target
    averages, per_subject, raw_score = cross_validate(stacks, bounds, subjects, grid, args.pcs, args.folds)
    lines += [f"Held-out correlation with subject 0's MEG ({args.folds} folds over sentences, mean over channels):",
              "", "| shared components | others' mapped average |", "|---|---|"]
    lines += [f"| {count} | {averages[count]:.4f} |" for count in grid]
    lines += [f"| raw average, no mapping | {raw_score:.4f} |", ""]

    if args.mode == "raw":
        maps, weights = None, {subject: 1.0 / len(others) for subject in others}
        lines.append("Mode raw: every subject's raw epochs averaged with equal weight.")
        chosen = None
    else:
        chosen = max(averages, key=averages.get) if args.components == "auto" else int(args.components)

        #each subject by its own held-out agreement with subject 0; none below zero
        agreement = {subject: max(per_subject[chosen][subject], 0.0) for subject in others}
        total = sum(agreement.values())
        weights = {subject: value / total for subject, value in agreement.items()} if total > 0 else \
            {subject: 1.0 / len(others) for subject in others}
        maps, eigenvalues = fit_maps(stacks, scalers, subjects, args.pcs, chosen)
        ranked = sorted(others, key=lambda subject: per_subject[chosen][subject], reverse=True)
        lines += [f"Chosen: **C = {chosen}**. Shared-component eigenvalues (1 = one subject's worth): "
                  + ", ".join(f"{value:.2f}" for value in eigenvalues[:10]) + (" ..." if len(eigenvalues) > 10 else ""),
                  "", "Per-subject held-out correlation at the chosen C (the weights):",
                  ", ".join(f"{subject}: {per_subject[chosen][subject]:.3f}" for subject in ranked)]

    #the virtual holdout, then the record of how it was made
    virtual = build_virtual(args.holdout_dir, reference, others, matches, epoch_order, maps, weights)
    out_dir = REPO_ROOT / args.out_root / name
    word_file = holdout_file(args.holdout_dir, 0, "word")
    target = write_virtual(reference_path, out_dir, virtual, word_file)
    (out_dir / "matched.json").write_text(json.dumps({
        "mode": args.mode, "components": chosen, "pcs": args.pcs, "subjects": others,
        "matched_epochs": epoch_order, "weights": {str(subject): weights[subject] for subject in others},
        "heldout": {str(count): averages[count] for count in grid}, "raw_heldout": raw_score,
    }, indent=1), encoding="utf-8")
    lines += ["", f"Wrote `{target.name}` under `{args.out_root}/{name}` and matched.json. Predict every member with "
              f"`--holdout-dir {args.out_root}/{name}`."]
    report = RUNS_DIR / f"trial22_align_{name}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def read_sentences_n_times(holdout_dir, subject):
    """One subject's sentence lengths, without decompressing its MEG.

    ARGS:
        holdout_dir (str)
        subject (int)

    RETURNS:
        [n] int64 samples at 250 Hz
    """
    with np.load(sentence_file(holdout_dir, subject), allow_pickle=False) as stored:
        return np.asarray(stored["sentence_n_times"]).astype(np.int64)


if __name__ == "__main__":
    main()
