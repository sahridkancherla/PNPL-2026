"""The per-word family's view of the holdout: windows per word, sentences as sequences.

Trial 18's `cli/predict.py` without its command line: the resampler, the two
scalers, the batch maker and the two gatherings, plus `run`, the whole
prediction for a loaded per-word module. `cli.predict` calls `run` when a
checkpoint is a per-word one; `cli.distil` uses the gatherings and the batch
maker to serve holdout items to a student.

Per holdout item:

    1. resample 250 -> 50 Hz with the builder's resample_poly, padtype
       "line": an epoch has edges the continuous recording never had
    2. robust-scale per channel: the FIXED training median and IQR
       (`--scaling training`), or one pooled over the holdout source's epochs
    3. clamp and baseline: inside the model, exactly as in training

In-context mode, each holdout SENTENCE whole: a scored word's window runs
from its onset to the earlier of 3 s later and the sentence's end, and the
sentence transformer sees every scored word; a lone word goes in alone with
its 1 s. `--window-only` gives every example its own 1 s window and nothing
else.

Trial 21 lifts the old refusal of `--bn-adapt` under the training scaler:
the holdout principle was withdrawn on 2026-10-08, and re-estimating
BatchNorm on the holdout's own windows (AdaBN) is the gate for the session
adaptation this trial is about.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. Every prediction is written to its
canonical `index`, never appended, and every row must be filled exactly once
or nothing is written.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.signal import resample_poly

from ...config import REPO_ROOT, SUBMISSIONS_DIR
from ..cli.ensemble import softmax_rows, standardise_rows
from ..data_processing.build_reference import DOWNSAMPLE, IQR_FLOOR, SAMPLE_RATE_OUT
from .lit import adapt_batch_norm

#sequences per forward pass; a sentence holds at most 9 scored words
SEQUENCES_PER_BATCH = 32

#the holdout's sample rate, which the builder's resampler assumes
HOLDOUT_RATE = 250.0


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
        raise SystemExit(f"No {path}. Run `python -m pnpl2026.trial22.cli.training_scaler` first.")
    fixed = FixedScaler(path)
    return lambda segments: fixed


class Predictor:
    """Runs sequences of scaled 50 Hz windows through a per-word checkpoint."""

    def __init__(self, module, window_samples, device, subject_layer=None):
        """
        ARGS:
            module (LitTrialEighteen): on `device`
            window_samples (int): the padded window length, 150
            device (torch.device)
            subject_layer (int or None): trial 22: every window through this
                subject layer; None is subject 0's map, as in training
        """
        self.module = module
        self.window_samples = window_samples
        self.device = device
        self.subject_layer = subject_layer

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

        #another subject's layer for every window, when one is named
        if self.subject_layer is not None:
            batch["subject"] = torch.full((len(lengths),), int(self.subject_layer), dtype=torch.int64)
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
        #one head: its scaled cosines; Variant K: both heads combined
        primary, moses = self.module.candidate_scores(self.module(self.make_batch(sequences)))

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
        list of (rows, sequences), as `gather_window_only`; every sequence's
        rows are consecutive in `rows`, sequence after sequence
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


def units_of(batches):
    """Each sequence of the gathered batches on its own, with its rows.

    `cli.distil` draws holdout items one sequence at a time; the gatherings
    above pack them 32 to a batch.

    ARGS:
        batches (list): of (rows, sequences) from a gathering

    RETURNS:
        list of (rows, members): one sequence's canonical rows, in word
        order, and its scaled windows
    """
    units = []
    for rows, sequences in batches:
        start = 0
        for members in sequences:
            units.append((list(rows[start : start + len(members)]), members))
            start += len(members)

        #⚠️ every row of the batch must have found its sequence
        if start != len(rows):
            raise SystemExit(f"A gathered batch lists {len(rows)} rows for {start} windows.")
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


def run(args, module, cfg, info, device):
    """Score every holdout example with a per-word module, check the rows, write the CSV.

    ARGS:
        args (argparse.Namespace): `cli.predict`'s arguments
        module (LitTrialEighteen): in eval mode, on `device`
        cfg (Trial18Config): the run's config, from the checkpoint
        info (dict): the checkpoint's epoch and global_step
        device (torch.device)
    """
    from pnpl.competition import write_submission

    shifts = list(dict.fromkeys(args.onset_shifts))
    if args.window_only and shifts != [0]:
        raise SystemExit("--onset-shifts needs the in-context mode: a canonical 1 s window has nothing to shift into.")
    print(f"[predict] {cfg.run_name} (per-word family), epoch {info['epoch']}, on {device}; scaling: {args.scaling}")

    #⚠️ the model's rate, checked: the resampler here produces 50 Hz only
    if float(cfg.sfreq) != SAMPLE_RATE_OUT or HOLDOUT_RATE / DOWNSAMPLE != SAMPLE_RATE_OUT:
        raise SystemExit(f"The run was trained at {cfg.sfreq} Hz; this predictor serves {SAMPLE_RATE_OUT}.")
    window_samples = cfg.samples(cfg.window_seconds)

    #trial 22: another subject's own layer, for a run trained with several
    from ..subjects import open_holdout, resolve_subject_layer, subject_suffix

    subject_layer = resolve_subject_layer(args, int(getattr(cfg, "n_subjects", 1)))
    if subject_layer is not None:
        print(f"[predict] every window through subject layer {subject_layer}")
    predictor = Predictor(module, window_samples, device, subject_layer)
    scaler_for = scaler_factory(args.scaling, args.scaler_file)

    holdout = open_holdout(args)
    print(f"[predict] {len(holdout)} holdout windows of subject(s) {list(holdout.subjects)}: {holdout.counts()}")
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

    #the name says how the file was made, so files are never mistaken
    mode = ("window_only" if args.window_only else "in_context") + ("_bnadapt" if args.bn_adapt else "")
    mode += ("_ts" if args.scaling == "training" else "") + ("_tta" if shifts != [0] else "")
    out_path = args.out or str(SUBMISSIONS_DIR / f"{cfg.run_name}{subject_suffix(args)}_{mode}.csv")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    #both vocabularies by retrieval, each in its own column order
    write_submission(out_path, indices=holdout.indices, primary_probs=primary, secondary_probs=moses)
    print(f"[predict] wrote {out_path} ({len(primary)} rows)")
    print("[predict] submission is MANUAL, through the competition web UI.")
