"""Turn a trial 10, 12, 13 or 15 checkpoint into a competition submission.

    PYTHONPATH=src python -m pnpl2026.trial15.cli.predict \\
        --ckpt runs/trial15_i1_s0/best.ckpt [--onset-shifts -1 0 1] [--window-only]

Needs only the checkpoint, the holdout files and -- for the default scaling --
`data/trial15_training_scaler.npz` from `cli.training_scaler`: the run's
config, the spatial layer and the 50 + 50 candidate embeddings are all inside
the checkpoint.

The holdout serves 250 Hz MEG in the h5's units; the model reads 50 Hz MEG
robust-scaled per recording. So, per holdout item:

    1. resample 250 -> 50 Hz with the builder's resample_poly. padtype
       "line": an epoch has edges the continuous recording never had, and a
       zero-padded edge rings into the first and last ~0.2 s
    2. robust-scale, per channel:
         --scaling training   (default) with the FIXED median and IQR of the
                              training recordings -- the holdout principle
                              (CLAUDE.md): nothing computed across the
                              holdout as a whole. Files get "_ts" in their name
         --scaling holdout    with a median and IQR POOLED over all of a
                              holdout source's epochs, as every submission
                              before 2026-10-07 did. Kept only to reproduce
                              those files; never for a final pick
    3. clamp and baseline (and I1/I2's per-window standardisation): inside the
       model, exactly as in training

Two modes, as trial 8:

    default          each holdout SENTENCE whole: a scored word's window runs
                     from its onset to the earlier of 3 s later and the
                     sentence's end, and the sentence transformer sees every
                     scored word. Isolated words go in alone, with their 1 s
    --window-only    every example alone, from its own 1 s window

    --onset-shifts   in-context mode only. Score the holdout once per shift of
                     every word's onset, in 50 Hz samples (e.g. -1 0 1 is
                     -20, 0 and +20 ms), and average each row's z-scored
                     log-probabilities -- each item on its own, so within the
                     principle. A lone word has no MEG before its onset, so a
                     negative shift leaves it unshifted. "0" alone is exactly
                     the unshifted prediction; anything else adds "_tta"
    --bn-adapt       re-estimate BatchNorm on the holdout windows about to be
                     scored. It pools all holdout rows, so it breaks the
                     principle and is refused with --scaling training

⚠️ RULES. The default mode uses holdout MEG OUTSIDE each example's 1 s window,
within the same item. The organisers' guidance (2026-10-07): unsupervised use
of holdout MEG is not prohibited; we keep to one item at a time.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. Every prediction is written to its
canonical `index`, never appended, and every row must be filled exactly once
or nothing is written.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch
from scipy.signal import resample_poly

from ...config import REPO_ROOT, SUBMISSIONS_DIR
from ..data_processing.build_reference import DOWNSAMPLE, IQR_FLOOR, SAMPLE_RATE_OUT
from ..lit import adapt_batch_norm, load_trained
from .ensemble import softmax_rows, standardise_rows

#sequences per forward pass; a sentence holds at most 9 scored words
SEQUENCES_PER_BATCH = 32

#the holdout's sample rate, which the builder's resampler assumes
HOLDOUT_RATE = 250.0


def parse_args():
    """Command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to best.ckpt")
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
        help="fixed training statistics (the principle), or pooled over the holdout (old files only)",
    )
    parser.add_argument(
        "--scaler-file",
        default="data/trial15_training_scaler.npz",
        help="repo-relative, from cli.training_scaler; read by --scaling training",
    )
    parser.add_argument(
        "--onset-shifts",
        nargs="+",
        type=int,
        default=[0],
        help="onset shifts in 50 Hz samples, averaged per row; in-context mode only",
    )
    parser.add_argument(
        "--window-only",
        action="store_true",
        help="every example from its own 1 s window, no context",
    )
    parser.add_argument("--device", default=None, help="default cuda when available")
    parser.add_argument(
        "--bn-adapt",
        action="store_true",
        help="re-estimate BatchNorm on the holdout windows first (pools all rows; --scaling holdout only)",
    )
    parser.add_argument("--no-ema", action="store_true", help="the live weights, not the average")
    return parser.parse_args()


def downsample_epoch(raw):
    """One holdout epoch from 250 Hz to the model's 50 Hz.

    ARGS:
        raw (np.ndarray): [306, n] float32 at 250 Hz, real samples only

    RETURNS:
        [306, ceil(n / 5)] float32 at 50 Hz; input sample 5k is output sample k
    """
    #the builder's filter, with the edges extended linearly instead of by zeros
    return resample_poly(raw, 1, DOWNSAMPLE, axis=1, padtype="line").astype(np.float32)


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
        raise SystemExit(f"No {path}. Run `python -m pnpl2026.trial15.cli.training_scaler` first.")
    fixed = FixedScaler(path)
    return lambda segments: fixed


class Predictor:
    """Runs sequences of scaled 50 Hz windows through a checkpoint."""

    def __init__(self, module, window_samples, device):
        """
        ARGS:
            module (LitTrialFifteen): in eval mode, on `device`
            window_samples (int): the padded window length, 150
            device (torch.device)
        """
        self.module = module
        self.window_samples = window_samples
        self.device = device

    def make_batch(self, sequences):
        """A batch dict, laid out exactly as training's `collate_tokens`.

        ARGS:
            sequences (list): one list per sequence of scaled [306, n] windows
                at 50 Hz, n <= window_samples, in word order

        RETURNS:
            dict of tensors on the device: windows, lengths, sequence, position
        """
        windows, lengths, sequence, position = [], [], [], []
        for index, members in enumerate(sequences):
            for place, window in enumerate(members):
                #zero-padded to the full length, as training windows are
                padded = np.zeros((window.shape[0], self.window_samples), dtype=np.float32)
                padded[:, : window.shape[1]] = window[:, : self.window_samples]
                windows.append(padded)
                lengths.append(min(window.shape[1], self.window_samples))
                sequence.append(index)
                position.append(place)

        batch = {
            "windows": torch.from_numpy(np.stack(windows)),
            "lengths": torch.tensor(lengths, dtype=torch.int64),
            "sequence": torch.tensor(sequence, dtype=torch.int64),
            "position": torch.tensor(position, dtype=torch.int64),
        }
        return {name: value.to(self.device) for name, value in batch.items()}

    @torch.no_grad()
    def probabilities(self, sequences):
        """Score a batch of sequences.

        ARGS:
            sequences (list): as `make_batch`

        RETURNS:
            primary : [total words, 50] numpy, sequence by sequence
            moses : [total words, 50] numpy, the same rows
        """
        primary, moses = self.module.candidate_logits(self.module(self.make_batch(sequences))["embedding"])

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


def gather_window_only(holdout, scaler_for):
    """Every example alone, from its canonical 1 s window, as batches.

    ARGS:
        holdout (LibriBrainCompetitionHoldout)
        scaler_for (callable): segments -> scaler, from `scaler_factory`

    RETURNS:
        list of (rows, sequences): canonical rows, in the order the sequences'
        words come out of the model
    """
    #⚠️ iterated exactly once, in the order the loader yields
    windows, metas = [], []
    for raw, batch_metas in holdout.iter_windows(batch_size=64):
        windows += [downsample_epoch(window) for window in raw]
        metas += batch_metas

    #one scaler per source: pooled over its windows, or the fixed one
    scalers = {}
    for source in sorted({meta["source"] for meta in metas}):
        scalers[source] = scaler_for([w for w, m in zip(windows, metas) if m["source"] == source])

    batches = []
    for start in range(0, len(metas), SEQUENCES_PER_BATCH):
        chosen = range(start, min(start + SEQUENCES_PER_BATCH, len(metas)))
        sequences = [[scalers[metas[i]["source"]](windows[i])] for i in chosen]
        batches.append(([metas[i]["index"] for i in chosen], sequences))
    return batches


def gather_in_context(holdout, window_samples, scaler_for, shift=0):
    """Sentences whole, isolated words alone -- the model's own setting -- as batches.

    ARGS:
        holdout (LibriBrainCompetitionHoldout)
        window_samples (int): the longest window after onset, at 50 Hz
        scaler_for (callable): segments -> scaler, from `scaler_factory`
        shift (int): every onset moved by this many 50 Hz samples, kept inside
            its epoch

    RETURNS:
        list of (rows, sequences), as `gather_window_only`
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
        #windows only, and context needs the whole sentence epoch
        with np.load(holdout._ensure_file(subject, "sentence"), allow_pickle=True) as stored:
            meg = np.asarray(stored["meg"], dtype=np.float32)
            n_times = np.asarray(stored["sentence_n_times"]).astype(np.int64)

        #every sentence at 50 Hz, then the scaler: fixed, or pooled over them all
        epochs = [downsample_epoch(meg[epoch, :, : n_times[epoch]]) for epoch in range(len(meg))]
        scaler = scaler_for(epochs)

        keys = sorted(key for key in sentences if key[0] == subject)
        for start in range(0, len(keys), SEQUENCES_PER_BATCH):
            sequences, rows = [], []
            for _, epoch in keys[start : start + SEQUENCES_PER_BATCH]:
                scaled = scaler(epochs[epoch])

                #word-slot order within the sentence, as the loader lists them
                members = []
                for meta in sorted(sentences[(subject, epoch)], key=lambda meta: meta["word"]):
                    #the onset on the 50 Hz grid, rounded as the builder rounds,
                    #then shifted and kept inside the epoch
                    first = int(round(meta["onset_s"] * SAMPLE_RATE_OUT)) + shift
                    first = min(max(first, 0), scaled.shape[1] - 1)
                    members.append(scaled[:, first : min(first + window_samples, scaled.shape[1])])
                    rows.append(meta["index"])
                sequences.append(members)

            #rows were collected in the sentence-then-word order scores come in
            batches.append((rows, sequences))

        #isolated words: one-word sequences, their 1 s epoch, scaled as above
        mine = [meta for meta in isolated if meta["subject"] == subject]
        if not mine:
            continue
        with np.load(holdout._ensure_file(subject, "word"), allow_pickle=True) as stored:
            word_meg = np.asarray(stored["meg"], dtype=np.float32)
        word_epochs = [downsample_epoch(epoch) for epoch in word_meg]
        word_scaler = scaler_for(word_epochs)

        #a lone word's epoch starts at its onset: only a later start exists
        start_sample = max(shift, 0)
        for start in range(0, len(mine), SEQUENCES_PER_BATCH):
            chosen = mine[start : start + SEQUENCES_PER_BATCH]
            sequences = [[word_scaler(word_epochs[meta["epoch"]])[:, start_sample:]] for meta in chosen]
            batches.append(([meta["index"] for meta in chosen], sequences))
    return batches


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


def main():
    """Score every holdout example, check the rows, write the CSV."""
    args = parse_args()
    from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

    #the options that break the principle, or make no sense together, stop here
    shifts = list(dict.fromkeys(args.onset_shifts))
    if args.bn_adapt and args.scaling == "training":
        raise SystemExit("--bn-adapt pools all holdout rows (against the holdout principle); "
                         "it is allowed only with --scaling holdout, to reproduce old files.")
    if args.window_only and shifts != [0]:
        raise SystemExit("--onset-shifts needs the in-context mode: a canonical 1 s window has nothing to shift into.")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module, cfg, info = load_trained(args.ckpt, device, use_ema=not args.no_ema)
    print(f"[predict] {cfg.run_name}, epoch {info['epoch']}, on {device}; scaling: {args.scaling}")

    #⚠️ the model's rate, checked: the resampler below produces 50 Hz only
    if float(cfg.sfreq) != SAMPLE_RATE_OUT or HOLDOUT_RATE / DOWNSAMPLE != SAMPLE_RATE_OUT:
        raise SystemExit(f"The run was trained at {cfg.sfreq} Hz; this predictor serves {SAMPLE_RATE_OUT}.")
    window_samples = cfg.samples(cfg.window_seconds)
    predictor = Predictor(module, window_samples, device)
    scaler_for = scaler_factory(args.scaling, args.scaler_file)

    holdout = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir)
    print(f"[predict] {len(holdout)} holdout windows: {holdout.counts()}")
    if args.window_only:
        print("[predict] --window-only: every example from its own 1 s window, no context")
    else:
        print("[predict] in-context mode: each run read whole, one item at a time")

    #one full pass per onset shift; a single pass is written exactly as scored
    passes = []
    for shift in shifts:
        if args.window_only:
            batches = gather_window_only(holdout, scaler_for)
        else:
            batches = gather_in_context(holdout, window_samples, scaler_for, shift)

        #AdaBN on exactly the windows about to be scored, if asked
        if args.bn_adapt:
            count = adapt_batch_norm(module, (predictor.make_batch(sequences) for _, sequences in batches), device)
            print(f"[predict] --bn-adapt: {count} BatchNorm layers re-estimated on the holdout's own windows")
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

    #the name says how the file was made, so a non-compliant one is never mistaken
    mode = ("window_only" if args.window_only else "in_context") + ("_bnadapt" if args.bn_adapt else "")
    mode += ("_ts" if args.scaling == "training" else "") + ("_tta" if shifts != [0] else "")
    out_path = args.out or str(SUBMISSIONS_DIR / f"{cfg.run_name}_{mode}.csv")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    #both vocabularies by retrieval, each in its own column order
    write_submission(out_path, indices=holdout.indices, primary_probs=primary, secondary_probs=moses)
    print(f"[predict] wrote {out_path} ({len(primary)} rows)")
    print("[predict] submission is MANUAL, through the competition web UI.")


if __name__ == "__main__":
    main()
