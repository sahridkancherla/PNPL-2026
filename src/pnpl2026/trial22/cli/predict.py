"""Turn a checkpoint of EITHER family into a competition submission.

    PYTHONPATH=src python -m pnpl2026.trial22.cli.predict \\
        --ckpt runs/trial22_b_s0/best.ckpt [--onset-shifts -1 0 1] [--window-only] [--bn-adapt]

The checkpoint's model name says which family it is (`families.py`): a
per-word one (trials 10-18, and trial 21's distilled students of them) goes
through `perword.holdout.run`; a continuous one (trials 19-21) through this
file. Both need only the checkpoint, the holdout files and -- for the
default scaling -- `data/trial22_training_scaler.npz` from
`cli.training_scaler`.

The holdout serves 250 Hz MEG in the h5's units; the model reads MEG at the
cache's rate, robust-scaled per recording. So, per holdout item:

    1. resample 250 Hz -> the model's rate with the builder's resample_poly.
       padtype "line": an epoch has edges the continuous recording never
       had, and a zero-padded edge rings into the first and last ~0.2 s
    2. robust-scale, per channel:
         --scaling training   (default) with the FIXED median and IQR of the
                              training recordings. Files get "_ts" in their
                              name
         --scaling holdout    with a median and IQR POOLED over all of a
                              holdout source's epochs
    3. clamp and the baseline: inside the model, as in training

The continuous family reads each holdout SENTENCE whole: one item is the
epoch, with every valid word's onset; a lone word is a 1 s item with one
onset at 0. That is exactly the form training items take.

    --onset-shifts   score the holdout once per shift of every word's onset,
                     in samples at the model's rate (-1 0 1 is -20, 0 and
                     +20 ms at 50 Hz), and average each row's z-scored
                     log-probabilities. "0" alone is exactly the unshifted
                     prediction; anything else adds "_tta"
    --window-only    every example alone, from its own 1 s window
    --bn-adapt       trial 21: re-estimate every BatchNorm's running
                     statistics on the holdout items about to be scored
                     (AdaBN), under either scaling. The megstate stem has
                     none, so for it this changes nothing; the per-word
                     encoder and the dascoli stem have ten. Adds "_bnadapt"

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. Every prediction is written to its
canonical `index`, never appended, and every row must be filled exactly once
or nothing is written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from fractions import Fraction

import numpy as np
import torch
from scipy.signal import resample_poly

from ...config import REPO_ROOT, SUBMISSIONS_DIR
from ..data_processing.build_reference import IQR_FLOOR
from ..families import load_any
from ..lit import adapt_batch_norm
from ..perword import holdout as perword_holdout
from ..subjects import open_holdout, resolve_subject_layer, subject_suffix
from .ensemble import softmax_rows, standardise_rows

#items per forward pass; a holdout sentence is at most 4.12 s
ITEMS_PER_BATCH = 32

#the holdout's sample rate, which the builder's resampler assumes
HOLDOUT_RATE = 250.0


def parse_args():
    """Command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to best.ckpt, of either family")
    parser.add_argument("--out", default=None, help="submission CSV path")
    parser.add_argument("--track", default="deep", help="competition track; 'deep' is subject 0")
    parser.add_argument(
        "--holdout-dir",
        default=None,
        help="a local folder holding COMPETITION_HOLDOUT/; default downloads to the HF cache",
    )
    parser.add_argument(
        "--scaling",
        choices=("training", "holdout"),
        default="training",
        help="fixed training statistics, or pooled over the holdout",
    )
    parser.add_argument(
        "--scaler-file",
        default="data/trial22_training_scaler.npz",
        help="repo-relative, from cli.training_scaler; read by --scaling training",
    )
    parser.add_argument(
        "--onset-shifts",
        nargs="+",
        type=int,
        default=[0],
        help="onset shifts in samples at the model's rate, averaged per row; in-context mode only",
    )
    parser.add_argument(
        "--window-only",
        action="store_true",
        help="every example from its own 1 s window, no context",
    )
    parser.add_argument(
        "--bn-adapt",
        action="store_true",
        help="re-estimate BatchNorm on the holdout items first (AdaBN); a no-op for a network without BatchNorm",
    )
    parser.add_argument("--device", default=None, help="default cuda when available")
    parser.add_argument("--no-ema", action="store_true", help="the live weights, not the average")
    parser.add_argument(
        "--subject",
        type=int,
        default=None,
        help="trial 22: read this holdout subject's files (0-39) instead of the track's; adds _subjKK",
    )
    parser.add_argument(
        "--subject-layer",
        default="none",
        help="trial 22, per-word all-data runs: 'auto' (from --others-manifest), a layer index, or 'none'",
    )
    parser.add_argument(
        "--others-manifest",
        default="data/trial18_others/manifest.json",
        help="repo-relative; the other subjects' cache manifest, which orders the subject layers",
    )
    return parser.parse_args()


def downsample_epoch(raw, sfreq):
    """One holdout epoch from 250 Hz to the model's rate.

    ARGS:
        raw (np.ndarray): [306, n] float32 at 250 Hz, real samples only
        sfreq (float): the model's rate, 50 or 100

    RETURNS:
        [306, about n * sfreq / 250] float32
    """
    #the builder's filter, with the edges extended linearly instead of by zeros
    ratio = Fraction(int(round(sfreq)), int(HOLDOUT_RATE))
    return resample_poly(raw, ratio.numerator, ratio.denominator, axis=1, padtype="line").astype(np.float32)


class RobustScaler:
    """Per-channel median and IQR, fitted on the segments it is given.

    ATTRS:
        median (np.ndarray): [306]
        iqr (np.ndarray): [306], floored as the builder floors it
    """

    def __init__(self, segments):
        """
        ARGS:
            segments (list): [306, n_i] arrays at 50 Hz, every one real data
        """
        #axis=1 is time, pooled over every segment: one statistic per channel
        pooled = np.concatenate(segments, axis=1).astype(np.float64)
        q25, self.median, q75 = np.percentile(pooled, [25, 50, 75], axis=1)
        self.iqr = np.maximum(q75 - q25, IQR_FLOOR)

    def __call__(self, segment):
        """
        ARGS:
            segment (np.ndarray): [306, n]

        RETURNS:
            [306, n] float32, scaled
        """
        return ((segment - self.median[:, None]) / self.iqr[:, None]).astype(np.float32)


class FixedScaler(RobustScaler):
    """The training recordings' median and IQR, read from a file and frozen.

    ATTRS:
        median (np.ndarray): [306]
        iqr (np.ndarray): [306]
    """

    def __init__(self, path):
        """
        ARGS:
            path (Path): written by cli.training_scaler
        """
        with np.load(path) as stored:
            self.median = stored["median"].astype(np.float64)
            self.iqr = np.maximum(stored["iqr"].astype(np.float64), IQR_FLOOR)


def scaler_factory(scaling, scaler_file):
    """What scales a group of holdout segments, under the chosen scaling.

    ARGS:
        scaling (str): "training" or "holdout"
        scaler_file (str): repo-relative, for "training"

    RETURNS:
        callable: list of [306, n] segments -> a scaler for them
    """
    if scaling == "holdout":
        return RobustScaler

    #one frozen scaler for everything: the segments are never looked at
    path = REPO_ROOT / scaler_file
    if not path.exists():
        raise SystemExit(f"No {path}. Run `python -m pnpl2026.trial22.cli.training_scaler` first.")
    fixed = FixedScaler(path)
    return lambda segments: fixed


class Predictor:
    """Runs continuous items through a checkpoint."""

    def __init__(self, module, device):
        """
        ARGS:
            module (LitTrialTwentyTwo): on `device`
            device (torch.device)
        """
        self.module = module
        self.device = device

    def make_batch(self, items):
        """A batch dict, laid out exactly as training's `collate_items`.

        ARGS:
            items (list): of (signal [306, n] at the model's rate, scaled;
                onsets [n_words] int samples within the signal, ascending)

        RETURNS:
            dict of tensors on the device: signal, lengths, onset, marker,
            sequence, position
        """
        n_channels = items[0][0].shape[0]
        lengths = [signal.shape[1] for signal, _ in items]
        signal = np.zeros((len(items), n_channels, max(lengths)), dtype=np.float32)
        onset, sequence, position = [], [], []
        for index, (member, onsets) in enumerate(items):
            #zero-padded to the batch's longest, as training items are
            signal[index, :, : member.shape[1]] = member
            for place, sample in enumerate(onsets):
                onset.append(int(sample))
                sequence.append(index)
                position.append(place)

        #every holdout onset is given, so every word gets the marker
        batch = {
            "signal": torch.from_numpy(signal),
            "lengths": torch.tensor(lengths, dtype=torch.int64),
            "onset": torch.tensor(onset, dtype=torch.int64),
            "marker": torch.ones(len(onset), dtype=torch.bool),
            "sequence": torch.tensor(sequence, dtype=torch.int64),
            "position": torch.tensor(position, dtype=torch.int64),
        }
        return {name: value.to(self.device) for name, value in batch.items()}

    @torch.no_grad()
    def probabilities(self, items):
        """Score a batch of items.

        ARGS:
            items (list): as `make_batch`

        RETURNS:
            primary : [total words, 50] numpy, item by item
            moses : [total words, 50] numpy, the same rows
        """
        #every head the run has, combined as in training's metrics
        primary, moses = self.module.candidate_scores(self.module(self.make_batch(items)))

        #dim=1 is the candidate axis: one distribution per word
        return primary.softmax(dim=1).cpu().numpy(), moses.softmax(dim=1).cpu().numpy()


def fill(rows, scores, primary, moses):
    """Write scores to their canonical rows, refusing to write any row twice.

    ARGS:
        rows (list): canonical indices, in the order `scores` come
        scores (tuple): (primary, moses) from Predictor.probabilities
        primary (np.ndarray): [N, 50], NaN where unwritten
        moses (np.ndarray): [N, 50]
    """
    for row, primary_row, moses_row in zip(rows, *scores):
        #⚠️ a doubled row means two words were mapped to one index
        if not np.isnan(primary[row]).all():
            raise SystemExit(f"Row {row} was predicted twice; the alignment is broken.")
        primary[row] = primary_row
        moses[row] = moses_row


def gather_window_only(holdout, scaler_for, sfreq):
    """Every example alone, from its canonical 1 s window, as batches.

    ARGS:
        holdout (LibriBrainCompetitionHoldout)
        scaler_for (callable): segments -> scaler, from `scaler_factory`
        sfreq (float): the model's rate

    RETURNS:
        list of (rows, items): canonical rows, in the order the items' words
        come out of the model
    """
    #⚠️ iterated exactly once, in the order the loader yields
    windows, metas = [], []
    for raw, batch_metas in holdout.iter_windows(batch_size=64):
        windows += [downsample_epoch(window, sfreq) for window in raw]
        metas += batch_metas

    #one scaler per source: pooled over its windows, or the fixed one
    scalers = {}
    for source in sorted({meta["source"] for meta in metas}):
        scalers[source] = scaler_for([w for w, m in zip(windows, metas) if m["source"] == source])

    batches = []
    for start in range(0, len(metas), ITEMS_PER_BATCH):
        chosen = range(start, min(start + ITEMS_PER_BATCH, len(metas)))
        items = [(scalers[metas[i]["source"]](windows[i]), [0]) for i in chosen]
        batches.append(([metas[i]["index"] for i in chosen], items))
    return batches


def gather_in_context(holdout, scaler_for, sfreq, shift=0):
    """Sentences whole, lone words alone -- the model's own setting -- as batches.

    ARGS:
        holdout (LibriBrainCompetitionHoldout)
        scaler_for (callable): segments -> scaler, from `scaler_factory`
        sfreq (float): the model's rate
        shift (int): every onset moved by this many samples, kept inside
            its item

    RETURNS:
        list of (rows, items), as `gather_window_only`; every item's rows
        are consecutive in `rows`, item after item
    """
    #the loader's metadata decides which words are scored and where each goes
    sentences = defaultdict(list)
    isolated = []
    for meta in holdout.metadata:
        if meta["source"] == "sentence":
            sentences[(meta["subject"], meta["epoch"])].append(meta)
        else:
            isolated.append(meta)

    batches = []
    for subject in holdout.subjects:
        #⚠️ private API of the pinned pnpl: the public loader serves 1 s
        #windows only, and an item is the whole sentence epoch
        with np.load(holdout._ensure_file(subject, "sentence"), allow_pickle=True) as stored:
            meg = np.asarray(stored["meg"], dtype=np.float32)
            n_times = np.asarray(stored["sentence_n_times"]).astype(np.int64)

        #every sentence at the model's rate, then the scaler: fixed, or pooled over them all
        epochs = [downsample_epoch(meg[epoch, :, : n_times[epoch]], sfreq) for epoch in range(len(meg))]
        scaler = scaler_for(epochs)

        keys = sorted(key for key in sentences if key[0] == subject)
        for start in range(0, len(keys), ITEMS_PER_BATCH):
            items, rows = [], []
            for _, epoch in keys[start : start + ITEMS_PER_BATCH]:
                scaled = scaler(epochs[epoch])

                #word-slot order within the sentence, as the loader lists them;
                #each onset on the model's grid, rounded as the builder rounds,
                #then shifted and kept inside the epoch
                onsets = []
                for meta in sorted(sentences[(subject, epoch)], key=lambda meta: meta["word"]):
                    sample = int(round(meta["onset_s"] * sfreq)) + shift
                    onsets.append(min(max(sample, 0), scaled.shape[1] - 1))
                    rows.append(meta["index"])
                items.append((scaled, onsets))

            #rows were collected in the sentence-then-word order scores come in
            batches.append((rows, items))

        #lone words: 1 s items with one onset, scaled as above
        mine = [meta for meta in isolated if meta["subject"] == subject]
        if not mine:
            continue
        with np.load(holdout._ensure_file(subject, "word"), allow_pickle=True) as stored:
            word_meg = np.asarray(stored["meg"], dtype=np.float32)
        word_epochs = [downsample_epoch(epoch, sfreq) for epoch in word_meg]
        word_scaler = scaler_for(word_epochs)

        #a lone word's epoch starts at its onset: only a later onset exists
        for start in range(0, len(mine), ITEMS_PER_BATCH):
            chosen = mine[start : start + ITEMS_PER_BATCH]
            items = []
            for meta in chosen:
                scaled = word_scaler(word_epochs[meta["epoch"]])
                items.append((scaled, [min(max(shift, 0), scaled.shape[1] - 1)]))
            batches.append(([meta["index"] for meta in chosen], items))
    return batches


def units_of(batches):
    """Each item of the gathered batches on its own, with its rows.

    ARGS:
        batches (list): of (rows, items) from a gathering

    RETURNS:
        list of (rows, item): one item's canonical rows, in word order, and
        the item itself (signal, onsets)
    """
    units = []
    for rows, items in batches:
        start = 0
        for item in items:
            n_words = len(item[1])
            units.append((list(rows[start : start + n_words]), item))
            start += n_words

        #⚠️ every row of the batch must have found its item
        if start != len(rows):
            raise SystemExit(f"A gathered batch lists {len(rows)} rows for {start} words.")
    return units


def predict_rows(predictor, batches, n_rows, n_classes):
    """Every holdout row's probabilities from one pass over the batches.

    ARGS:
        predictor (Predictor)
        batches (list): of (rows, sequences)
        n_rows (int): holdout rows
        n_classes (int): 50

    RETURNS:
        primary, moses : [n_rows, n_classes] each
    """
    #NaN until written, so a row never filled cannot pass unnoticed
    primary = np.full((n_rows, n_classes), np.nan)
    moses = np.full((n_rows, n_classes), np.nan)
    for rows, sequences in batches:
        fill(rows, predictor.probabilities(sequences), primary, moses)

    #⚠️ DO NOT REMOVE. A missing row is the only symptom of a broken alignment
    #that surfaces before the leaderboard does
    if np.isnan(primary).any() or np.isnan(moses).any():
        missing = int(np.isnan(primary).any(axis=1).sum())
        raise SystemExit(f"{missing} of {n_rows} holdout rows were never predicted.")
    return primary, moses


def describe(primary):
    """One line on how the predictions spread over the 50 words.

    ARGS:
        primary (np.ndarray): [N, 50]

    RETURNS:
        str
    """
    #a collapsed model puts every example's top choice on one or two words
    top = primary.argmax(axis=1)
    counts = np.bincount(top, minlength=primary.shape[1])
    return (
        f"top-1 covers {int((counts > 0).sum())} of {primary.shape[1]} words; the commonest takes "
        f"{100 * counts.max() / len(top):.1f}%; mean top probability {primary.max(axis=1).mean():.3f}"
    )


def run_continuous(args, module, cfg, info, device):
    """Score every holdout example with a continuous module, check the rows, write the CSV.

    ARGS:
        args (argparse.Namespace): from `parse_args`
        module (LitTrialTwentyTwo): in eval mode, on `device`
        cfg (Trial22Config): the run's config, from the checkpoint
        info (dict): the checkpoint's epoch and global_step
        device (torch.device)
    """
    from pnpl.competition import write_submission

    shifts = list(dict.fromkeys(args.onset_shifts))
    if args.window_only and shifts != [0]:
        raise SystemExit("--onset-shifts needs the in-context mode: a canonical 1 s window has nothing to shift into.")

    sfreq = float(cfg.sfreq)

    #the continuous runs have one subject layer: another subject is read through subject 0's
    if resolve_subject_layer(args, 1) not in (None, 0):
        raise SystemExit("--subject-layer applies to per-word all-data runs; this continuous run has one subject.")
    print(f"[predict] {cfg.run_name} (continuous family, {module.model.stem_kind} stem), epoch {info['epoch']}, "
          f"on {device}; scaling: {args.scaling}; {sfreq:g} Hz")
    predictor = Predictor(module, device)
    scaler_for = scaler_factory(args.scaling, args.scaler_file)

    holdout = open_holdout(args)
    print(f"[predict] {len(holdout)} holdout windows of subject(s) {list(holdout.subjects)}: {holdout.counts()}")
    if args.window_only:
        print("[predict] --window-only: every example from its own 1 s window, no context")
    else:
        print("[predict] in-context mode: each sentence read whole as one item, lone words as 1 s items")

    #one full pass per onset shift; a single pass is written exactly as scored
    passes = []
    for shift in shifts:
        if args.window_only:
            batches = gather_window_only(holdout, scaler_for, sfreq)
        else:
            batches = gather_in_context(holdout, scaler_for, sfreq, shift)

        #AdaBN on exactly the items about to be scored, if asked
        if args.bn_adapt:
            count = adapt_batch_norm(module, (predictor.make_batch(items) for _, items in batches), device)
            print(f"[predict] --bn-adapt: {count} BatchNorm layers re-estimated on the holdout's own items")
        passes.append(predict_rows(predictor, batches, len(holdout), cfg.n_classes))
        if len(shifts) > 1:
            print(f"[predict] onset shift {shift:+d} samples scored")

    #several shifts: each row's z-scored log-probabilities averaged, as cli.ensemble does
    if len(passes) == 1:
        primary, moses = passes[0]
    else:
        primary = softmax_rows(np.mean([standardise_rows(scores) for scores, _ in passes], axis=0))
        moses = softmax_rows(np.mean([standardise_rows(scores) for _, scores in passes], axis=0))

    #⚠️ DO NOT REMOVE: the count check against the loader's own indices
    if len(primary) != len(holdout.indices):
        raise SystemExit(f"{len(primary)} predictions for {len(holdout.indices)} holdout rows.")
    for name, table in (("primary", primary), ("moses", moses)):
        sums = table.sum(axis=1)
        if not np.allclose(sums, 1.0, atol=1e-4) or not np.isfinite(table).all():
            raise SystemExit(f"{name} rows are not distributions (sums {sums.min():.5f}..{sums.max():.5f}).")
    print(f"[predict] {describe(primary)}")

    #the name says how the file was made, so files are never mistaken
    mode = ("window_only" if args.window_only else "in_context") + ("_bnadapt" if args.bn_adapt else "")
    mode += ("_ts" if args.scaling == "training" else "") + ("_tta" if shifts != [0] else "")
    out_path = args.out or str(SUBMISSIONS_DIR / f"{cfg.run_name}{subject_suffix(args)}_{mode}.csv")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    #both vocabularies by retrieval, each in its own column order
    write_submission(out_path, indices=holdout.indices, primary_probs=primary, secondary_probs=moses)
    print(f"[predict] wrote {out_path} ({len(primary)} rows)")
    print("[predict] submission is MANUAL, through the competition web UI.")


def main():
    """Load the checkpoint, find its family, and predict with that family's code."""
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module, cfg, info, family = load_any(args.ckpt, device, use_ema=not args.no_ema)
    if family == "perword":
        perword_holdout.run(args, module, cfg, info, device)
    else:
        run_continuous(args, module, cfg, info, device)


if __name__ == "__main__":
    main()
