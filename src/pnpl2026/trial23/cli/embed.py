"""Dump a member's embeddings: the holdout's rows, and holdout-shaped chains of its own partitions.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.embed --ckpt runs/trial13_g2_s0/best.ckpt \\
        --holdout-dir data/holdout [--partitions train validation test] [--check-csv <the member's CSV>]

Change 2 of the Route to 73, second half: the CSVs hold 50 word scores; the
doc's transduction prototypes and kNN memory need the networks' own
embeddings. Either family (`families.load_any`). Writes, under
runs/<run>/embeddings/:

    holdout.npz            every canonical row, gathered exactly as
                           `cli.predict` gathers it (in context, the fixed
                           training scaler): each head's L2-normalised
                           embedding (t5; second for Variant K; tower for
                           the continuous family), the primary logits, and
                           each row's source and epoch
    chains_<partition>.npz the CHAINS protocol (`data/sequences.py`): only
                           vocabulary words as tokens, chained as the holdout
                           lists them, MEG cut 1.13-1.68 s after the last.
                           Per token: the heads, the logits, the label,
                           task, sequence and position. The training
                           partition is capped per word (`--train-cap`)

`--check-csv` compares the holdout logits' softmax with the member's own
submission CSV: the same rows must rank the same words, or the dump is not
the member the mix used.

Trial 23 adds two protocols and a TTA:

    chains38_<partition>.npz   the chains cut to 3-8 words, as the research
                               note's testbed and P6/P5 checks read them;
                               each token also carries its onset, recording
                               key and row in the recording's table
    isolated_<partition>.npz   every scored word alone, 1 s from onset
    --lone-tta                 P7, isolated protocol only: onset shifts of
                               -1/+1 samples and 8 sensor-dropout passes
                               (tta.py), z-scored and averaged with the plain
                               pass, written beside the run as
                               <run>_tta/dumps/isolated_<partition>.npz --
                               the layout cli.lone_check reads
    --recentre <reference>     P8b: every recording, and each holdout source
                               group, re-centred on the training covariance
                               (recentre.py); every dump's name gains _rc

GPU, minutes per member: `pbs/trial23_embed.pbs`, an array over
configs/trial23/members.txt.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...config import REPO_ROOT, RUNS_DIR
from ..families import load_any
from ..recentre import load_reference, recentre_dataset, recentring_scaler_factory
from ..tta import sensor_dropout
from .ensemble import read_submission

#the heads a network may have, by the output key that holds each
HEAD_KEYS = {"t5": "embedding", "second": "embedding_second", "tower": "brain_tower"}

#the chains are scored on identical sequences by every run: drawn with one seed
TRAIN_CAP_SEED = 0

#P7's sensor-dropout passes per lone-word TTA
LONE_TTA_DROPOUT_PASSES = 8

#P8b's transforms, by the recordings they were computed for
RECENTRE_CACHE = {}


def zscore_logits(logits):
    """Each row's logits z-scored, as cli.ensemble reads a CSV.

    ARGS:
        logits (np.ndarray): [n, 50]

    RETURNS:
        [n, 50] float64
    """
    #axis=1 is the word axis
    centred = logits.astype(np.float64) - logits.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


class NoTargets:
    """Stands in for the t5 table: the dump needs no training targets."""

    def rows(self, words):
        """No word has a training target here.

        ARGS:
            words (np.ndarray): [n] words

        RETURNS:
            [n] int64 array of -1
        """
        return np.full(len(words), -1, dtype=np.int64)


def parse_args():
    """Command-line arguments for the embedding dump."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="best.ckpt of either family")
    parser.add_argument("--out-dir", default=None, help="default runs/<the checkpoint's run>/embeddings")
    parser.add_argument("--partitions", nargs="*", default=["train", "validation", "test"], help="chains dumped; none skips")
    parser.add_argument("--protocol", default="chains", help="trial 23: chains, chains38 (the testbed's 3-8 words) or isolated")
    parser.add_argument("--lone-tta", action="store_true",
                        help="trial 23's P7, isolated protocol: onset shifts -1/0/+1 and 8 sensor-dropout passes averaged, "
                             "written beside the run as <run>_tta/dumps/isolated_<partition>.npz for cli.lone_check")
    parser.add_argument("--recentre", default=None,
                        help="trial 23's P8b: data/trial23_recentre_reference.npz from cli.recentre -- every recording (and "
                             "holdout source) re-centred on the training covariance; the dumps end in _rc")
    parser.add_argument("--train-cap", type=int, default=2000, help="most training tokens kept per word")
    parser.add_argument("--cache-dir", default=None, help="override the run's cache_dir (a staged copy)")
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/; empty string skips the holdout")
    parser.add_argument("--track", default="deep")
    parser.add_argument("--scaling", choices=("training", "holdout"), default="training")
    parser.add_argument("--scaler-file", default="data/trial23_training_scaler.npz")
    parser.add_argument("--check-csv", default=None, help="the member's submission CSV, to confirm the dump is that member")
    parser.add_argument("--batch-size", type=int, default=32, help="sequences per batch")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def head_outputs(out):
    """Every head's embeddings in one forward pass, L2-normalised.

    ARGS:
        out (dict): a module's forward output

    RETURNS:
        dict: head name -> [N, D] float32 numpy
    """
    heads = {}
    for name, key in HEAD_KEYS.items():
        if key in out and out[key] is not None:
            #cosine is what every head is compared by, so unit length is kept
            vectors = out[key].float()
            heads[name] = torch.nn.functional.normalize(vectors, dim=1).cpu().numpy()
    return heads


def holdout_scaler(scaler_factory, args):
    """The holdout's scaler factory, re-centring each source group when asked (P8b).

    ARGS:
        scaler_factory (callable): the family's own
        args (argparse.Namespace)

    RETURNS:
        callable: segments -> scaler
    """
    scaler_for = scaler_factory(args.scaling, args.scaler_file)
    if not args.recentre:
        return scaler_for
    reference, n_directions = load_reference(REPO_ROOT / args.recentre)
    return recentring_scaler_factory(scaler_for, reference, n_directions)


def recentre_recordings(dataset, args):
    """Re-centre every recording of a chain dataset (P8b); each transform computed once.

    ARGS:
        dataset: either family's dataset, with `recordings`
        args (argparse.Namespace)
    """
    if not args.recentre:
        return
    #one transform per recording key, kept across TTA's rebuilt datasets
    keys = tuple(record["key"] for record in dataset.recordings.records)
    if keys not in RECENTRE_CACHE:
        reference, n_directions = load_reference(REPO_ROOT / args.recentre)
        recentre_dataset(dataset.recordings, reference, n_directions)
        RECENTRE_CACHE[keys] = dataset.recordings.transforms
        print(f"[embed] {len(keys)} recordings re-centred on {args.recentre} ({n_directions} directions)")
    dataset.recordings.transforms = RECENTRE_CACHE[keys]


@torch.no_grad()
def dump_holdout(module, cfg, family, args, device):
    """The holdout's rows, gathered as `cli.predict` gathers them.

    ARGS:
        module: in eval mode, on `device`
        cfg: the run's config
        family (str): "perword" or "continuous"
        args (argparse.Namespace)
        device (torch.device)

    RETURNS:
        dict of numpy arrays, one row per canonical index
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    holdout = LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir)

    #each family's own gathering and batch layout, unchanged
    if family == "perword":
        from ..perword.holdout import Predictor, gather_in_context, scaler_factory

        predictor = Predictor(module, cfg.samples(cfg.window_seconds), device)
        batches = gather_in_context(holdout, cfg.samples(cfg.window_seconds), holdout_scaler(scaler_factory, args))
    else:
        from .predict import Predictor, gather_in_context, scaler_factory

        predictor = Predictor(module, device)
        batches = gather_in_context(holdout, holdout_scaler(scaler_factory, args), float(cfg.sfreq))

    n_rows = len(holdout)
    arrays = {"logits": np.full((n_rows, cfg.n_classes), np.nan, dtype=np.float32)}
    for rows, items in batches:
        out = module(predictor.make_batch(items))
        logits = module.candidate_scores(out)[0].float().cpu().numpy()

        #⚠️ every row written once, at its canonical index
        if not np.isnan(arrays["logits"][rows]).all():
            raise SystemExit("A holdout row was gathered twice; the alignment is broken.")
        arrays["logits"][rows] = logits
        for name, vectors in head_outputs(out).items():
            arrays.setdefault(name, np.full((n_rows, vectors.shape[1]), np.nan, dtype=np.float32))
            arrays[name][rows] = vectors
    if np.isnan(arrays["logits"]).any():
        raise SystemExit(f"{int(np.isnan(arrays['logits']).any(axis=1).sum())} holdout rows were never scored.")
    arrays["source"] = np.array([meta["source"] for meta in holdout.metadata])
    arrays["epoch"] = np.array([meta["epoch"] for meta in holdout.metadata], dtype=np.int64)
    return arrays


def check_against_csv(logits, csv_path):
    """Does the dump rank the words as the member's own CSV does?

    ARGS:
        logits (np.ndarray): [N, 50] from `dump_holdout`
        csv_path (str)

    RETURNS:
        str: one line for the log
    """
    from pnpl.competition import load_vocabulary

    index, primary, _ = read_submission(Path(csv_path), load_vocabulary("primary"), load_vocabulary("moses"))
    if len(index) != len(logits):
        raise SystemExit(f"{csv_path} has {len(index)} rows; the dump has {len(logits)}.")

    #the same top word, and closely the same ten, row by row
    same_top = float(np.mean(logits.argmax(axis=1) == primary.argmax(axis=1)))
    top_dump = np.argpartition(-logits, 9, axis=1)[:, :10]
    top_csv = np.argpartition(-primary, 9, axis=1)[:, :10]
    overlap = float(np.mean([len(set(a) & set(b)) / 10 for a, b in zip(top_dump, top_csv)]))
    if same_top < 0.98:
        raise SystemExit(f"The dump ranks differently from {csv_path}: top-1 equal on {100 * same_top:.1f}% of rows.")
    return f"[embed] matches {Path(csv_path).name}: top-1 equal on {100 * same_top:.2f}% of rows, top-10 overlap {100 * overlap:.2f}%"


def chain_dataset(cfg, family, partition, cache_dir, protocol="chains"):
    """One partition's dataset under a dump protocol, cut with this trial's protocol settings.

    ARGS:
        cfg: the run's config
        family (str)
        partition (str)
        cache_dir (str or None): an override, e.g. a staged copy
        protocol (str): "chains", "chains38" or "isolated"

    RETURNS:
        dataset, collate function
    """
    #the protocol fields reset, so every member's chains are the same tokens
    if family == "perword":
        from ..perword.config import with_protocol_defaults
        from ..perword.data import build_partition
        from ..perword.data.dataset import collate_tokens as collate
    else:
        from ..config import with_protocol_defaults
        from ..data import build_partition
        from ..data.dataset import collate_items as collate
    cfg = with_protocol_defaults(cfg)
    if cache_dir:
        cfg = replace(cfg, cache_dir=cache_dir)
    return build_partition(cfg, NoTargets(), partition, protocol), collate


@torch.no_grad()
def dump_chains(module, cfg, family, partition, args, device, shift=0):
    """Every chain token's heads, logits and labels on one partition.

    ARGS:
        module, cfg, family: as `dump_holdout`
        partition (str)
        args (argparse.Namespace)
        device (torch.device)
        shift (int): trial 23's P7: every onset moved by this many samples

    RETURNS:
        dict of numpy arrays, one row per kept token
    """
    dataset, collate = chain_dataset(cfg, family, partition, args.cache_dir, args.protocol)
    recentre_recordings(dataset, args)
    if shift:
        #the same items, each onset moved: a lone word's window starts earlier or later
        dataset.sequences = [replace(sequence, onsets=sequence.onsets + shift) for sequence in dataset.sequences]

    #⚠️ never shuffled: token metadata is rebuilt from the dataset's own order
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate)
    collected = {"logits": [], "primary": [], "isolated": []}
    for batch in loader:
        batch = {name: value.to(device) for name, value in batch.items()}
        out = module(batch)
        collected["logits"].append(module.candidate_scores(out)[0].float().cpu().numpy())
        collected["primary"].append(batch["primary"].cpu().numpy())
        collected["isolated"].append(batch["isolated"].cpu().numpy())
        for name, vectors in head_outputs(out).items():
            collected.setdefault(name, []).append(vectors)
    arrays = {name: np.concatenate(values) for name, values in collected.items()}

    #each token's sequence, place in it, recording and task, in loader order
    arrays["sequence"] = np.concatenate([np.full(len(seq.onsets), index) for index, seq in enumerate(dataset.sequences)])
    arrays["position"] = np.concatenate([np.arange(len(seq.onsets)) for seq in dataset.sequences])
    recording = np.concatenate([np.full(len(seq.onsets), seq.recording) for seq in dataset.sequences])
    arrays["task"] = np.array([record["task"] for record in dataset.recordings.records])[recording]
    arrays["recording"] = recording

    #trial 23: each token's onset (cache samples), recording key and word-table row
    arrays["onset"] = np.concatenate([np.asarray(seq.onsets, dtype=np.int64) for seq in dataset.sequences])
    arrays["key"] = np.array([record["key"] for record in dataset.recordings.records])[recording]
    if all(getattr(seq, "table_rows", None) is not None for seq in dataset.sequences):
        arrays["table_index"] = np.concatenate([seq.table_rows for seq in dataset.sequences])
    if len(arrays["primary"]) != len(arrays["sequence"]):
        raise SystemExit(f"{partition}: {len(arrays['primary'])} scored tokens for {len(arrays['sequence'])} listed.")

    #the training memory capped per word, the same draw for every member
    if partition == "train" and args.train_cap > 0:
        generator = np.random.default_rng(TRAIN_CAP_SEED)
        keep = np.zeros(len(arrays["primary"]), dtype=bool)
        for word in np.unique(arrays["primary"][arrays["primary"] >= 0]):
            rows = np.flatnonzero(arrays["primary"] == word)
            chosen = rows if len(rows) <= args.train_cap else generator.choice(rows, args.train_cap, replace=False)
            keep[chosen] = True
        arrays = {name: value[keep] for name, value in arrays.items()}
    return arrays


def save(path, arrays):
    """Write a dump, embeddings as float16 to halve the disk.

    ARGS:
        path (Path)
        arrays (dict)
    """
    stored = {name: value.astype(np.float16) if name in HEAD_KEYS else value for name, value in arrays.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **stored)


def main():
    """Load the member, dump the holdout and each partition's chains."""
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    module, cfg, info, family = load_any(args.ckpt, device, use_ema=not args.no_ema)
    run = Path(args.ckpt).parent.name
    out_dir = Path(args.out_dir) if args.out_dir else RUNS_DIR / run / "embeddings"

    #P7's TTA is for lone words: their protocol alone
    if args.lone_tta and args.protocol != "isolated":
        raise SystemExit(f"--lone-tta scores lone words: it needs --protocol isolated, not {args.protocol!r}.")
    print(f"[embed] {run} ({family} family), epoch {info['epoch']}, on {device} -> {out_dir}")

    #P8b's dumps are kept apart from the plain ones by name
    suffix = "_rc" if args.recentre else ""

    #the holdout, exactly as predicted, and the check that it is this member
    if args.holdout_dir != "":
        arrays = dump_holdout(module, cfg, family, args, device)
        if args.check_csv:
            print(check_against_csv(arrays["logits"], args.check_csv))
        save(out_dir / f"holdout{suffix}.npz", arrays)
        print(f"[embed] holdout: {len(arrays['logits'])} rows; heads {[name for name in HEAD_KEYS if name in arrays]}")

    #the chains of every partition asked for
    for partition in args.partitions:
        arrays = dump_chains(module, cfg, family, partition, args, device)
        save(out_dir / f"{args.protocol}_{partition}{suffix}.npz", arrays)

        #P7: the same items under onset shifts and sensor-dropout passes, averaged
        if args.lone_tta:
            passes = [zscore_logits(arrays["logits"])]
            for shift in (-1, 1):
                passes.append(zscore_logits(dump_chains(module, cfg, family, partition, args, device, shift)["logits"]))
            for draw in range(LONE_TTA_DROPOUT_PASSES):
                with sensor_dropout(module, seed=1000 + draw):
                    passes.append(zscore_logits(dump_chains(module, cfg, family, partition, args, device)["logits"]))
            #beside the run's own directory, so cli.lone_check reads `<run>_tta` as a run
            tta_dir = Path(args.ckpt).parent.parent / f"{run}_tta{suffix}" / "dumps"
            tta_dir.mkdir(parents=True, exist_ok=True)
            np.savez(tta_dir / f"{args.protocol}_{partition}.npz", primary_logits=np.mean(passes, axis=0),
                     primary=arrays["primary"], isolated=arrays["isolated"], task=arrays["task"], sequence=arrays["sequence"])
            print(f"[embed] {args.protocol} {partition} with TTA ({len(passes)} passes) -> {tta_dir}")
        scored = arrays["primary"] >= 0
        print(f"[embed] {args.protocol} {partition}: {len(arrays['primary'])} tokens, {int(scored.sum())} scored, "
              f"{len(np.unique(arrays['sequence']))} chains, {len(np.unique(arrays['primary'][scored]))} words")


if __name__ == "__main__":
    main()
