"""Turn a trial-9 checkpoint into a competition submission.

    PYTHONPATH=src python -m pnpl2026.trial9.cli.predict \\
        --ckpt runs/trial9_h1_s0/best.ckpt \\
        --config configs/base.yaml configs/trial9/trial9.yaml \\
                 configs/trial9/h1_highpass_05.yaml

Two modes:

    default          each holdout SENTENCE goes through the model whole. A
                     scored word's window runs from its onset to the earlier of
                     3 s later and the sentence's end, and the sentence
                     transformer sees every scored word. The isolated-word
                     source goes in as one-word sequences with 1 s windows

    --window-only    every example alone, from its own 1 s window, with no
                     context at all -- the canonical `iter_windows()` input.
                     Training sees 9.6% of its words exactly this way

⚠️ RULES. The default mode uses holdout MEG OUTSIDE each example's 1 s window:
the rest of its sentence, and the other words' windows. Nothing published says
whether that is allowed. Until the organisers answer, --window-only is the
only submission known to be within the rules.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. Sentences are rebuilt from the loader's
own metadata, and every prediction is written back to its canonical `index`,
never appended. Every row must be filled exactly once, or this refuses to
write anything.

⚠️ Pass the run's own configs, WITHOUT pbs/trial9.pbs's staged-cache layer:
the transforms are read from the continuous cache on RDS.
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from ...config import SUBMISSIONS_DIR
from ..config import load_trial9_config
from ..data import load_manifest
from ..lit import LitTrialNine
from ..losses import semantic_logits

#what the continuous cache's builder copied trial 1's transforms into
TRANSFORMS_NAME = "transforms.npz"

#sentences per forward pass in the default mode; at most 9 words each
SENTENCES_PER_BATCH = 32


def parse_args():
    """Command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to best.ckpt")
    parser.add_argument("--config", nargs="+", required=True, help="the run's own configs")
    parser.add_argument("--out", default=None, help="submission CSV path")
    parser.add_argument("--track", default="deep", help="competition track; 'deep' is subject 0")
    parser.add_argument(
        "--holdout-dir",
        default=None,
        help="a local folder holding COMPETITION_HOLDOUT/; default downloads to the HF cache",
    )
    parser.add_argument(
        "--window-only",
        action="store_true",
        help="every example from its own 1 s window, no context: the rules-safe mode",
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="predict with the live weights. WARNING: the run was SELECTED on the averaged ones",
    )
    return parser.parse_args()


def load_transforms(cache_dir):
    """Read the standardiser and whitener the continuous cache was built with.

    ARGS:
        cache_dir (Path): the continuous cache

    RETURNS:
        centre : [n_sensors], scale : [n_sensors], matrix : [n_components, n_sensors]
    """
    path = cache_dir / TRANSFORMS_NAME
    if not path.exists():
        raise SystemExit(f"No {TRANSFORMS_NAME} in {cache_dir} -- rebuild the continuous cache.")
    stored = np.load(path)
    return tuple(np.asarray(stored[name], dtype=np.float64) for name in ("centre", "scale", "matrix"))


def to_components(raw, centre, scale, matrix, clip_sigma):
    """Put one raw sensor window into the model's input space.

    The same arithmetic the continuous cache was built with.

    ARGS:
        raw (np.ndarray): [n_sensors, n_samples] float32, as served
        centre (np.ndarray): [n_sensors]
        scale (np.ndarray): [n_sensors]
        matrix (np.ndarray): [n_components, n_sensors]
        clip_sigma (float): the cache's clip

    RETURNS:
        components : [n_components, n_samples] float32
    """
    #float64 for the affine and the matmul, as the builder does
    standardised = (raw.astype(np.float64) - centre[:, None]) / scale[:, None]
    return np.clip(matrix @ standardised, -clip_sigma, clip_sigma).astype(np.float32)


class Predictor:
    """Runs batches of holdout words through a checkpoint, scores them."""

    def __init__(self, model, transforms, window_samples, clip_sigma):
        """
        ARGS:
            model (LitTrialNine): in eval mode
            transforms (tuple): centre, scale, matrix
            window_samples (int): the padded window length
            clip_sigma (float): the cache's clip
        """
        self.model = model
        self.transforms = transforms
        self.window_samples = window_samples
        self.clip_sigma = clip_sigma

    @torch.no_grad()
    def probabilities(self, sequences):
        """Score a batch of sequences of raw windows.

        ARGS:
            sequences (list): one list per sequence of raw [306, n] windows,
                in word order

        RETURNS:
            [total words, n_classes] float numpy array, sequence by sequence
        """
        windows, lengths, sequence, position = [], [], [], []
        for index, raws in enumerate(sequences):
            for place, raw in enumerate(raws):
                components = to_components(raw, *self.transforms, self.clip_sigma)

                #zero-padded to the full length, exactly as training windows are
                padded = np.zeros((components.shape[0], self.window_samples), dtype=np.float32)
                padded[:, : components.shape[1]] = components
                windows.append(padded)
                lengths.append(components.shape[1])
                sequence.append(index)
                position.append(place)

        batch = {
            "windows": torch.from_numpy(np.stack(windows)),
            "lengths": torch.tensor(lengths, dtype=torch.int64),
            "sequence": torch.tensor(sequence, dtype=torch.int64),
            "position": torch.tensor(position, dtype=torch.int64),
        }
        out = self.model(batch)
        logit_scale = self.model.log_scale.exp()
        semantic = (
            semantic_logits(out["embedding"], self.model.target_embeddings, logit_scale)
            if self.model.uses_semantic
            else torch.zeros_like(out["logits"])
        )

        #the same blend validation was scored on
        blended = self.model._blend(out["logits"], semantic)
        return blended.softmax(dim=1).numpy()


def predict_window_only(holdout, predictor, probabilities):
    """Every example alone, from its canonical 1 s window.

    ARGS:
        holdout (LibriBrainCompetitionHoldout)
        predictor (Predictor)
        probabilities (np.ndarray): [N, n_classes], filled in place
    """
    #⚠️ iterated exactly once, in the order the loader yields
    for raw, metas in holdout.iter_windows(batch_size=64):
        scores = predictor.probabilities([[window] for window in raw])
        for meta, row in zip(metas, scores):
            probabilities[meta["index"]] = row


def predict_in_context(holdout, predictor, probabilities, window_samples):
    """Sentences whole, isolated words alone -- the model's own setting.

    ARGS:
        holdout (LibriBrainCompetitionHoldout)
        predictor (Predictor)
        probabilities (np.ndarray): [N, n_classes], filled in place
        window_samples (int): the longest window after onset
    """
    #the loader's metadata is the source of truth for which words are scored
    #and where each one's row goes; group it back into sentences
    sentences = defaultdict(list)
    isolated = []
    for meta in holdout.metadata:
        if meta["source"] == "sentence":
            sentences[(meta["subject"], meta["epoch"])].append(meta)
        else:
            isolated.append(meta)

    for subject in holdout.subjects:
        #⚠️ private API of the pinned pnpl: the public loader serves 1 s
        #windows only, and context needs the whole sentence epoch
        with np.load(holdout._ensure_file(subject, "sentence"), allow_pickle=True) as stored:
            meg = np.asarray(stored["meg"], dtype=np.float32)
            n_times = np.asarray(stored["sentence_n_times"]).astype(np.int64)

        keys = sorted(key for key in sentences if key[0] == subject)
        for start in range(0, len(keys), SENTENCES_PER_BATCH):
            chosen = keys[start : start + SENTENCES_PER_BATCH]
            batch, rows = [], []
            for _, epoch in chosen:
                #word-slot order within the sentence, as the loader lists them
                metas = sorted(sentences[(subject, epoch)], key=lambda meta: meta["word"])
                raws = []
                for meta in metas:
                    first = meta["start_sample"]
                    stop = min(first + window_samples, int(n_times[epoch]))
                    raws.append(meg[epoch, :, first:stop])
                    rows.append(meta["index"])
                batch.append(raws)

            #rows were collected in the same sentence-then-word order the
            #predictor returns its scores in
            for row, scores in zip(rows, predictor.probabilities(batch)):
                probabilities[row] = scores

        #isolated words: one-word sequences, their full 1 s epoch
        with np.load(holdout._ensure_file(subject, "word"), allow_pickle=True) as stored:
            word_meg = np.asarray(stored["meg"], dtype=np.float32)
        mine = [meta for meta in isolated if meta["subject"] == subject]
        for start in range(0, len(mine), 64):
            chosen = mine[start : start + 64]
            scores = predictor.probabilities([[word_meg[meta["epoch"]]] for meta in chosen])
            for meta, row in zip(chosen, scores):
                probabilities[meta["index"]] = row


def main():
    """Score every holdout example, check the rows, write the CSV."""
    args = parse_args()
    cfg = load_trial9_config(*args.config)

    from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

    model = LitTrialNine.load_from_checkpoint(args.ckpt, cfg=cfg, map_location="cpu")
    if not args.no_ema and model.apply_ema_weights():
        print("[predict] using the averaged (EMA) weights")
    model.eval()

    cache_dir = cfg.resolved_continuous_dir()
    window_samples = int(round(cfg.window_seconds * cfg.sfreq))
    #the clip is the one the cache was built with, read rather than assumed
    clip_sigma = float(load_manifest(cache_dir)["clip_sigma"])
    predictor = Predictor(model, load_transforms(cache_dir), window_samples, clip_sigma)

    holdout = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir)
    print(f"[predict] {len(holdout)} holdout windows: {holdout.counts()}")

    #NaN until written, so a row never filled cannot pass unnoticed
    probabilities = np.full((len(holdout), cfg.n_classes), np.nan)
    if args.window_only:
        print("[predict] --window-only: every example from its own 1 s window, no context")
        predict_window_only(holdout, predictor, probabilities)
    else:
        print(
            "[predict] WARNING: in-context mode uses holdout MEG outside each example's "
            "1 s window. Confirm with the organisers before submitting it."
        )
        predict_in_context(holdout, predictor, probabilities, window_samples)

    #⚠️ DO NOT REMOVE. A missing or doubled row is the only symptom of a broken
    #alignment that surfaces before the leaderboard does
    if len(probabilities) != len(holdout.indices) or np.isnan(probabilities).any():
        missing = int(np.isnan(probabilities).any(axis=1).sum())
        raise SystemExit(f"{missing} of {len(probabilities)} holdout rows were never predicted.")

    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4) or not np.isfinite(probabilities).all():
        raise SystemExit(f"Rows are not distributions (sums {sums.min():.5f}..{sums.max():.5f}).")

    mode = "window_only" if args.window_only else "in_context"
    out_path = args.out or str(SUBMISSIONS_DIR / f"{cfg.run_name}_{mode}.csv")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    #primary vocabulary only, so the Moses columns get the uniform distribution
    secondary = np.full_like(probabilities, 1.0 / probabilities.shape[1])
    write_submission(
        out_path,
        indices=holdout.indices,
        primary_probs=probabilities,
        secondary_probs=secondary,
    )
    print(f"[predict] wrote {out_path} ({len(probabilities)} rows)")
    print("[predict] submission is MANUAL, through the competition web UI.")


if __name__ == "__main__":
    main()
