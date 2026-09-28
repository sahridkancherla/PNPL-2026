"""Turn a trial-5 checkpoint into a competition submission.

    PYTHONPATH=src python -m pnpl2026.trial5.cli.predict \\
        --ckpt runs/trial5_laptop/best.ckpt --out submissions/trial5.csv

⚠️ ROW ORDERING IS THE WHOLE CONTRACT. `iter_windows()` yields windows in the
canonical order and the host builds its label CSV by iterating the SAME loader.
Indices align by construction and only by construction. Nothing here sorts,
shuffles, filters or reorders anything, and the count check at the end is not
optional -- reordering breaks alignment with no error and no warning, and the
only symptom is a chance-level leaderboard score against a validation number
that looked fine.

⚠️ `iter_windows()` yields RAW 306-channel windows. The model was trained on 73
whitened components, so the standardiser and whitener that built the cache have
to be re-applied here. They are stored in the cache's `transforms.npz`. Applying
them per window is equivalent to applying them to the continuous recording
before cutting, because both are purely per-channel affine and a linear mix --
neither touches the time axis.

⚠️ No Kaggle credentials exist on this machine. This writes a CSV; submitting it
is manual, through the web UI. Nothing here submits anything.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from ...config import SUBMISSIONS_DIR
from ..config import load_trial5_config
from ..lit import LitTrialFive
from ..losses import semantic_logits

#what the cache builder wrote its fitted transforms into
TRANSFORMS_NAME = "transforms.npz"


def parse_args():
    """Command-line arguments for prediction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to best.ckpt")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="the same configs the run was trained with",
    )
    parser.add_argument("--out", default=None, help="submission CSV path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--track", default="deep", help="competition track; 'deep' is subject 0"
    )
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="predict with the live weights instead of the averaged ones. ⚠️ The "
        "run was SELECTED on the averaged ones, so this is an ablation, not a "
        "default.",
    )
    return parser.parse_args()


def load_transforms(cache_dir):
    """Read the standardiser and whitener the cache was built with.

    ARGS:
        cache_dir (Path): the cache directory, which holds transforms.npz

    RETURNS:
        centre : [n_sensors] per-channel offset
        scale : [n_sensors] per-channel divisor
        matrix : [n_components, n_sensors] the whitening matrix
    """
    path = cache_dir / TRANSFORMS_NAME
    if not path.exists():
        raise SystemExit(
            f"No {TRANSFORMS_NAME} in {cache_dir}. Without the fitted "
            f"standardiser and whitener the holdout cannot be put into the same "
            f"space the model was trained in."
        )

    stored = np.load(path)
    return (
        np.asarray(stored["centre"], dtype=np.float64),
        np.asarray(stored["scale"], dtype=np.float64),
        np.asarray(stored["matrix"], dtype=np.float64),
    )


def to_components(raw, centre, scale, matrix, clip_sigma):
    """Put one batch of raw sensor windows into the model's input space.

    ARGS:
        raw (np.ndarray): [batch, n_sensors, n_samples] float32, as served
        centre (np.ndarray): [n_sensors] standardiser offset
        scale (np.ndarray): [n_sensors] standardiser divisor
        matrix (np.ndarray): [n_components, n_sensors] whitening matrix
        clip_sigma (float): the same clip the cache applied before fp16

    RETURNS:
        components : [batch, n_components, n_samples] float32
    """
    #float64 for the affine and the matmul, matching what the builder did; the
    #21x magnetometer/gradiometer scale gap makes float32 here marginal
    standardised = (raw.astype(np.float64) - centre[None, :, None]) / scale[None, :, None]

    #einsum rather than matmul so the batch axis stays where it is; `matrix`
    #mixes SENSORS, which is axis 1
    whitened = np.einsum("cs,bst->bct", matrix, standardised)

    #the model never saw an unclipped window, so it should not see one now
    return np.clip(whitened, -clip_sigma, clip_sigma).astype(np.float32)


@torch.no_grad()
def main():
    """Score every holdout window in canonical order and write the CSV."""
    args = parse_args()
    cfg = load_trial5_config(*args.config)

    from pnpl.competition import LibriBrainCompetitionHoldout, write_submission

    model = LitTrialFive.load_from_checkpoint(
        args.ckpt, cfg=cfg, map_location="cpu", n_sessions=0, strict=False
    )
    #⚠️ the run was selected on the EMA weights, so the submission has to use
    #them too or it is a different model from the one that scored
    if not args.no_ema:
        if model.apply_ema_weights():
            print("[predict] using the averaged (EMA) weights")
        else:
            print("[predict] no EMA in this checkpoint; using the live weights")
    else:
        print("[predict] --no-ema: using the live weights")

    model.eval()

    #CPU on purpose: this is one pass over the holdout, and MPS numerics
    #differing from CPU on a submission is not a trade worth making
    centre, scale, matrix = load_transforms(cfg.resolved_cache_dir())

    #the whitener's output width must be what the model was built for
    if matrix.shape[0] != cfg.n_components:
        raise SystemExit(
            f"transforms.npz whitener produces {matrix.shape[0]} components but "
            f"the config says n_components={cfg.n_components}."
        )

    holdout = LibriBrainCompetitionHoldout(track=args.track)
    print(f"[predict] {len(holdout)} holdout windows: {holdout.counts()}")

    primary = []
    logit_scale = model.log_scale.exp()

    #⚠️ iterated exactly once, in the order the loader yields. Never sorted
    for raw, _meta in holdout.iter_windows(batch_size=args.batch_size):
        windows = torch.from_numpy(
            to_components(raw, centre, scale, matrix, cfg.clip_sigma)
        )
        out = model(windows)

        semantic = (
            semantic_logits(out["embedding"], model.target_embeddings, logit_scale)
            if model.uses_semantic
            else torch.zeros_like(out["logits"])
        )

        #the same blend the validation metric was computed on, so the submission
        #scores what the run was selected on
        blended = model._blend(out["logits"], semantic)
        primary.append(blended.softmax(dim=1).numpy())

    probabilities = np.concatenate(primary)

    #⚠️ DO NOT REMOVE. A count mismatch means the iteration went wrong, and it
    #is the only symptom that surfaces before the leaderboard does
    if len(probabilities) != len(holdout.indices):
        raise SystemExit(
            f"Produced {len(probabilities)} predictions for "
            f"{len(holdout.indices)} holdout windows. The row alignment is "
            f"broken and the submission would score at chance."
        )

    #every row must be a distribution, or the scorer reads garbage silently
    sums = probabilities.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-4):
        raise SystemExit(
            f"Row probabilities do not sum to 1 (min {sums.min():.5f}, "
            f"max {sums.max():.5f})."
        )
    if not np.isfinite(probabilities).all():
        raise SystemExit("Non-finite probabilities in the submission.")

    out_path = args.out or str(SUBMISSIONS_DIR / f"{cfg.run_name}.csv")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    #trial 5 trains on the primary vocabulary only, so the Moses columns get the
    #uniform distribution rather than a number nothing supports
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
