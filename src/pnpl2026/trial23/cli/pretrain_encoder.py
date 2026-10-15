"""Self-supervised pretraining of the per-word encoder: masked-span reconstruction.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.pretrain_encoder \\
        --config configs/base.yaml configs/trial18/trial18.yaml configs/trial18/hpc.yaml \\
        --run-name trial23_ssl --holdout-dir data/holdout [--hours 1.75]

The final night's SSL member (`info/final-night-14oct.md`). G2's word encoder
(merger, 1x1, subject layer, the ten dilated convs, the final 1x1 pair -- built
by the same `build_model` the fine-tune uses) learns to fill in masked stretches
of MEG, with no label anywhere:

    windows   3 s crops of every recording of subject 0's cache (train,
              validation and test: no label is read) and, for --holdout-share
              of each batch, crops of the holdout epochs -- subject 0's, scaled
              by the fixed training scaler as `cli.predict` scales them, and
              --holdout-subjects' others, each scaled by its own pooled median
              and IQR (the organisers allow any use of the holdout MEG)
    masking   0.2 s spans zeroed after the clamp and baseline, until about
              --mask-share of each window's real samples are hidden
    target    the clamped, baselined input on the hidden samples (MSE), read
              by a 1x1 decoder from the conv stack's per-step output

Writes `runs/<run-name>/encoder.pt` (the encoder's state dict without the
attention pool, which the fine-tune trains afresh) and `pretrain.json` (the
settings, the windows seen, the loss as a fraction of the hidden samples'
variance). Trial 18's `init_encoder_from` loads it (configs/trial18/ssl_g2.yaml).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

from ...config import REPO_ROOT, RUNS_DIR
from ..perword.config import load_trial18_config
from ..perword.data.dataset import ReferenceRecordings
from ..perword.lit import spatial_inputs
from ..perword.models import SubjectLayer, build_model
from .predict import FixedScaler, RobustScaler, downsample_epoch

#every partition of subject 0's cache: no label is read, so none is held back
PARTITIONS = ("train", "validation", "test")


def parse_args():
    """Command-line arguments for the pretraining."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", required=True, help="trial 18 config layers that define G2's encoder")
    parser.add_argument("--run-name", default="trial23_ssl")
    parser.add_argument("--cache-dir", default=None, help="override the config's cache_dir (a staged copy)")
    parser.add_argument("--holdout-dir", default=None, help="a folder holding COMPETITION_HOLDOUT/; '' skips the holdout")
    parser.add_argument("--holdout-subjects", default="all", help="'all' (0-39), or a comma list such as 0,1")
    parser.add_argument("--holdout-share", type=float, default=0.25, help="share of each batch cut from holdout epochs")
    parser.add_argument("--scaler-file", default="data/trial23_training_scaler.npz", help="repo-relative, for subject 0")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--hours", type=float, default=1.75, help="stop after this long, whatever the step count")
    parser.add_argument("--batch-size", type=int, default=256, help="windows per step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--mask-share", type=float, default=0.4)
    parser.add_argument("--span-seconds", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default=None, help="default cuda when available")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="overwrite an existing encoder.pt")
    return parser.parse_args()


class TrainingCrops(IterableDataset):
    """Endless random crops of subject 0's recordings, longer recordings drawn more often."""

    def __init__(self, recordings, window_samples, seed):
        """
        ARGS:
            recordings (ReferenceRecordings)
            window_samples (int): the crop length
            seed (int): each worker adds its own id
        """
        self.recordings = recordings
        self.window_samples = window_samples
        self.seed = seed

        #a recording's chance is its share of the samples a crop can start at
        starts = np.maximum(np.array(recordings.n_times) - window_samples, 0).astype(np.float64)
        self.weights = starts / starts.sum()

    def __iter__(self):
        """Yield (window [306, window_samples] float32, length) forever."""
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(self.seed * 1000 + (info.id if info is not None else 0))
        while True:
            recording = int(rng.choice(len(self.weights), p=self.weights))
            start = int(rng.integers(0, max(self.recordings.n_times[recording] - self.window_samples, 1)))
            #the cache stores [n_times, 306]; the encoder reads [306, n_times]
            crop = np.asarray(self.recordings.meg(recording)[start : start + self.window_samples], dtype=np.float32)
            yield crop.T.copy(), self.window_samples


def holdout_epochs(holdout_dir, subjects, scaler_file, sample_rate):
    """Every holdout epoch of the chosen subjects, at the model's rate, scaled.

    ARGS:
        holdout_dir (str): a folder holding COMPETITION_HOLDOUT/
        subjects (list): subject ids
        scaler_file (str): repo-relative, subject 0's fixed scaler
        sample_rate (float): the model's rate

    RETURNS:
        list of [306, n] float16 arrays, and a dict of epochs per subject
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    epochs, counts = [], {}
    for subject in subjects:
        try:
            holdout = LibriBrainCompetitionHoldout(subjects=[subject], data_path=holdout_dir)
            mine = []
            with np.load(holdout._ensure_file(subject, "sentence"), allow_pickle=True) as stored:
                meg = np.asarray(stored["meg"], dtype=np.float32)
                n_times = np.asarray(stored["sentence_n_times"]).astype(np.int64)
            mine += [downsample_epoch(meg[epoch, :, : n_times[epoch]], sample_rate) for epoch in range(len(meg))]
            with np.load(holdout._ensure_file(subject, "word"), allow_pickle=True) as stored:
                mine += [downsample_epoch(epoch, sample_rate) for epoch in np.asarray(stored["meg"], dtype=np.float32)]
        except Exception as error:
            #a subject whose files are not on disk is skipped, not fatal
            print(f"[ssl] holdout subject {subject} skipped: {type(error).__name__}: {error}")
            continue

        #subject 0 as cli.predict scales it; the others by their own pooled statistics
        scaler = FixedScaler(REPO_ROOT / scaler_file) if subject == 0 else RobustScaler(mine)
        epochs += [scaler(epoch).astype(np.float16) for epoch in mine]
        counts[int(subject)] = len(mine)
    return epochs, counts


def holdout_batch(epochs, n_windows, window_samples, rng):
    """Random crops of random holdout epochs, zero-padded to the window.

    ARGS:
        epochs (list): from `holdout_epochs`
        n_windows (int)
        window_samples (int)
        rng (np.random.Generator)

    RETURNS:
        windows : [n_windows, 306, window_samples] float32
        lengths : [n_windows] int64
    """
    n_channels = epochs[0].shape[0]
    windows = np.zeros((n_windows, n_channels, window_samples), dtype=np.float32)
    lengths = np.zeros(n_windows, dtype=np.int64)
    for row in range(n_windows):
        epoch = epochs[int(rng.integers(len(epochs)))]
        length = min(epoch.shape[1], window_samples)
        start = int(rng.integers(0, epoch.shape[1] - length + 1))
        windows[row, :, :length] = epoch[:, start : start + length]
        lengths[row] = length
    return windows, lengths


def span_mask(lengths, n_times, share, span, generator):
    """Which samples are hidden: spans of `span` samples, about `share` of each window's real ones.

    ARGS:
        lengths (torch.Tensor): [N] real samples per window
        n_times (int): padded length
        share (float): the hidden share aimed at
        span (int): samples per hidden span
        generator (torch.Generator)

    RETURNS:
        [N, n_times] bool, True where hidden
    """
    device = lengths.device
    n_spans = torch.clamp(torch.round(share * lengths.float() / span), min=1).long()
    most = int(n_spans.max())

    #span starts uniform over each window's real part; dim 1 is the span axis
    room = torch.clamp(lengths - span, min=0) + 1
    starts = (torch.rand(len(lengths), most, generator=generator, device=device) * room.unsqueeze(1)).long()
    active = torch.arange(most, device=device).unsqueeze(0) < n_spans.unsqueeze(1)
    steps = torch.arange(n_times, device=device).view(1, 1, -1)
    inside = (steps >= starts.unsqueeze(2)) & (steps < (starts + span).unsqueeze(2)) & active.unsqueeze(2)
    return inside.any(dim=1) & (steps.view(1, -1) < lengths.unsqueeze(1))


def encode_steps(encoder, windows, lengths, hidden):
    """The encoder's per-step output with the hidden samples zeroed at its input.

    The word encoder's own path up to its final 1x1 pair, as `features` runs
    it -- the clamp, the baseline, the spatial layer, the 1x1, the subject
    layer and the conv stack -- without the attention pool.

    ARGS:
        encoder (WordEncoder)
        windows (torch.Tensor): [N, 306, T]
        lengths (torch.Tensor): [N]
        hidden (torch.Tensor): [N, T] bool

    RETURNS:
        steps : [N, d_model, T]
        target : [N, 306, T] - the clamped, baselined input
        valid : [N, T] bool
    """
    x = windows.float().clamp(-encoder.clamp, encoder.clamp)
    valid = encoder.frame_valid(lengths, x.shape[2])
    mask = valid.unsqueeze(1).to(x.dtype)
    if encoder.input_norm == "instance":
        x = encoder.standardise_window(x, mask)

    #per channel, the mean of the window's first 0.5 s; dim 2 is time
    if encoder.baseline_samples > 0:
        baseline = x[:, :, : encoder.baseline_samples].mean(dim=2, keepdim=True)
        x = (x - baseline) * mask
    target = x

    #the hidden spans are zero at the input: only their context is seen
    seen = x * (~hidden).unsqueeze(1).to(x.dtype)
    h = encoder.spatial(seen)
    #the per-channel FIR filters of the temporal-filter front-end, when the encoder has them
    if getattr(encoder, "temporal_filter", None) is not None:
        h = encoder.temporal_filter(h * mask) * mask
    h = encoder.initial_linear(h)
    h = (encoder.subject(h, None) if isinstance(encoder.subject, SubjectLayer) else encoder.subject(h)) * mask
    h, _ = encoder.convs.forward_with_taps(h, mask, ())
    return encoder.final(h), target, valid


def collate_crops(items):
    """Training crops into a batch.

    ARGS:
        items (list): of (window [306, T] float32, length)

    RETURNS:
        windows [N, 306, T] float32, lengths [N] int64
    """
    windows = torch.from_numpy(np.stack([window for window, _ in items], axis=0))
    lengths = torch.tensor([length for _, length in items], dtype=torch.int64)
    return windows, lengths


def main():
    """Pretrain the encoder, then write encoder.pt and pretrain.json."""
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = load_trial18_config(*args.config)
    if args.cache_dir:
        cfg = dataclasses.replace(cfg, cache_dir=args.cache_dir)
    run_dir = RUNS_DIR / args.run_name
    if (run_dir / "encoder.pt").exists() and not args.force:
        raise SystemExit(f"{run_dir / 'encoder.pt'} exists; pass --force to overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    #G2's encoder exactly as the fine-tune builds it; only the encoder is kept
    spatial = spatial_inputs(cfg)
    network = build_model(cfg.model, n_channels=cfg.n_channels, embed_dim=1024, sample_rate=float(cfg.sfreq), clamp=cfg.clamp,
                          baseline_seconds=cfg.baseline_seconds, positions=spatial.get("positions"), pca=spatial.get("pca"),
                          second_head=False, refine=None, n_subjects=1, **cfg.model_kwargs)
    encoder = network.encoder.to(device)
    d_model = encoder.final[-1].out_channels
    decoder = nn.Conv1d(d_model, cfg.n_channels, 1).to(device)
    window_samples = cfg.samples(cfg.window_seconds)
    span = max(cfg.samples(args.span_seconds), 1)

    #the data: subject 0's every recording, and the holdout epochs
    recordings = ReferenceRecordings(cfg.resolved_cache_dir(), list(PARTITIONS))
    crops = DataLoader(TrainingCrops(recordings, window_samples, args.seed), batch_size=args.batch_size,
                       num_workers=args.num_workers, collate_fn=collate_crops, pin_memory=device.type == "cuda")
    subjects = list(range(40)) if args.holdout_subjects == "all" else [int(s) for s in args.holdout_subjects.split(",")]
    epochs, counts = holdout_epochs(args.holdout_dir, subjects, args.scaler_file, float(cfg.sfreq)) if args.holdout_dir else ([], {})
    n_holdout = int(round(args.batch_size * args.holdout_share)) if epochs else 0
    print(f"[ssl] {len(recordings.records)} recordings ({sum(recordings.n_times) / cfg.sfreq / 3600:.1f} h), "
          f"{len(epochs)} holdout epochs from {len(counts)} subjects; {args.batch_size - n_holdout} + {n_holdout} windows "
          f"per step of {window_samples} samples; spans of {span}, {100 * args.mask_share:.0f}% hidden; on {device}")

    #AdamW, linear warmup into a cosine to 0 over the planned steps
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    def rate(step):
        """Multiplier on the peak rate at `step`."""
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * min((step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1), 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, rate)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    encoder.train()
    decoder.train()
    history, started, step, seen_windows = [], time.time(), 0, 0
    for windows, lengths in crops:
        #the holdout's share of the batch replaces as many training crops
        if n_holdout:
            extra, extra_lengths = holdout_batch(epochs, n_holdout, window_samples, rng)
            windows = torch.cat([windows[: args.batch_size - n_holdout], torch.from_numpy(extra)], dim=0)
            lengths = torch.cat([lengths[: args.batch_size - n_holdout], torch.from_numpy(extra_lengths)], dim=0)
        windows, lengths = windows.to(device, non_blocking=True), lengths.to(device)
        hidden = span_mask(lengths, windows.shape[2], args.mask_share, span, generator)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            steps, target, valid = encode_steps(encoder, windows, lengths, hidden)
            prediction = decoder(steps)
        #the loss in float32, on hidden real samples only; dim 1 is the sensor axis
        weight = (hidden & valid).unsqueeze(1).float()
        error = ((prediction.float() - target) ** 2 * weight).sum() / (weight.sum() * target.shape[1]).clamp_min(1.0)
        variance = ((target ** 2) * weight).sum() / (weight.sum() * target.shape[1]).clamp_min(1.0)

        optimiser.zero_grad(set_to_none=True)
        scaler.scale(error).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        step += 1
        seen_windows += len(lengths)

        #the error as a share of the hidden samples' own variance: 1 is predicting zero
        if step % args.log_every == 0 or step == 1:
            fraction = float(error.detach()) / max(float(variance.detach()), 1e-12)
            hidden_share = float((hidden & valid).sum()) / max(float(valid.sum()), 1.0)
            history.append({"step": step, "loss": float(error.detach()), "unexplained": fraction,
                            "hidden_share": hidden_share, "minutes": (time.time() - started) / 60})
            print(f"[ssl] step {step}: loss {float(error.detach()):.4f}, {100 * fraction:.1f}% of the hidden variance "
                  f"unexplained, {100 * hidden_share:.0f}% hidden, {(time.time() - started) / 60:.1f} min")
        if step >= args.steps or time.time() - started > args.hours * 3600:
            break

    #the encoder without its pool: the fine-tune trains the pool afresh
    state = {name: value.detach().cpu() for name, value in encoder.state_dict().items() if not name.startswith("pool.")}
    torch.save(state, run_dir / "encoder.pt")
    record = {"settings": {name: value for name, value in vars(args).items()}, "steps": step, "windows_seen": seen_windows,
              "holdout_epochs": counts, "window_samples": window_samples, "span_samples": span,
              "minutes": (time.time() - started) / 60, "history": history}
    (run_dir / "pretrain.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"[ssl] wrote {run_dir / 'encoder.pt'} ({len(state)} tensors) after {step} steps, {seen_windows:,} windows")


if __name__ == "__main__":
    main()
