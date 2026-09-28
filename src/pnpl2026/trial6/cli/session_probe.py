"""C9: does the frozen encoder carry session identity? A linear probe.

    PYTHONPATH=src python -m pnpl2026.trial6.cli.session_probe \\
        --ckpt runs/trial6_c0/best.ckpt \\
        --config configs/base.yaml configs/trial6/trial6.yaml \\
                 configs/trial6/c0_reference.yaml

Variant C keeps the session adversary only if this finds something for it to
remove. Trial 5's adversary loss sat at chance even with the adversary off,
which means either the features carry no session identity or the adversary's
small head failed to find it. A linear probe on frozen features separates the
two, and costs a few minutes instead of a training run.

The probe reads `pooled` -- what every head reads -- from the EMA weights the
run was selected on, over TRAINING sessions, the adversary's own label space.

⚠️ Each recording is split by POSITION, never at random: its first 80% of
windows fit the probe and its last 20% test it. Neighbouring windows are near
duplicates, so a random split would read memorised windows as session identity.

A shuffled-label probe is fitted on the same features as the chance line. Its
score, not 1/n_sessions alone, is what the real probe has to clear.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from ..config import load_trial6_config
from ..data import Trial6DataModule
from ..lit import LitTrialSix


def parse_args():
    """Command-line arguments for the session probe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="path to best.ckpt")
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="the same configs the run was trained with",
    )
    parser.add_argument(
        "--per-recording",
        type=int,
        default=400,
        help="windows taken from each recording, evenly spaced through it",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="the LAST this fraction of each recording's windows is held out",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="probe the live weights instead of the averaged ones",
    )
    return parser.parse_args()


def split_by_position(recordings, per_recording, test_fraction):
    """Pick evenly spaced windows per recording, then split each by time.

    ARGS:
        recordings (list): np.ndarray of window positions per recording,
            each in recorded order
        per_recording (int): windows to take from each recording at most
        test_fraction (float): share of each recording's picks held out, taken
            from its END

    RETURNS:
        fit_positions : np.ndarray of window positions for fitting the probe
        test_positions : np.ndarray of window positions for scoring it
    """
    fit_positions, test_positions = [], []
    for positions in recordings:
        #evenly spaced, so the picks span the whole recording, not its start
        n_picked = min(per_recording, len(positions))
        picked = positions[np.linspace(0, len(positions) - 1, n_picked).round().astype(int)]
        picked = np.unique(picked)

        #the boundary falls in recorded order: fit early, test late
        n_test = max(1, int(round(len(picked) * test_fraction)))
        fit_positions.append(picked[:-n_test])
        test_positions.append(picked[-n_test:])

    return np.concatenate(fit_positions), np.concatenate(test_positions)


@torch.no_grad()
def pooled_features(model, dataset, positions, batch_size, device):
    """Run windows through the frozen encoder and keep what the heads read.

    ARGS:
        model (LitTrialSix): in eval mode, on `device`
        dataset (CachedWindows): the un-jittered training split
        positions (np.ndarray): which windows
        batch_size (int): windows per forward pass
        device (torch.device): where the model lives

    RETURNS:
        features : [n_windows, d_model] float32 numpy array
    """
    features = []
    for start in range(0, len(positions), batch_size):
        #index 0 of each item is the window; the session id is read separately
        windows = torch.stack(
            [dataset[int(position)][0] for position in positions[start : start + batch_size]]
        )
        features.append(model(windows.to(device))["pooled"].cpu().numpy())
    return np.concatenate(features)


def fit_and_score(fit_x, fit_y, test_x, test_y):
    """Standardise, fit a multinomial logistic regression, score it.

    ARGS:
        fit_x (np.ndarray): [n_fit, d_model]
        fit_y (np.ndarray): [n_fit] session indices
        test_x (np.ndarray): [n_test, d_model]
        test_y (np.ndarray): [n_test] session indices

    RETURNS:
        float: balanced accuracy on the test windows
    """
    #fitted on the fit windows only, so the test windows stay unseen
    scaler = StandardScaler().fit(fit_x)
    probe = LogisticRegression(max_iter=2000)
    probe.fit(scaler.transform(fit_x), fit_y)
    return float(balanced_accuracy_score(test_y, probe.predict(scaler.transform(test_x))))


def main():
    """Fit the probe on frozen features and report it against chance."""
    args = parse_args()
    cfg = load_trial6_config(*args.config)

    datamodule = Trial6DataModule(cfg)
    datamodule.setup()
    dataset = datamodule.fit_set

    #n_sessions must match the checkpoint's adversary, or its weights will not
    #load; strict=False then tolerates a run trained without one
    model = LitTrialSix.load_from_checkpoint(
        args.ckpt,
        cfg=cfg,
        map_location="cpu",
        n_sessions=datamodule.n_sessions,
        strict=False,
    )

    #⚠️ the run was SELECTED on the averaged weights, so those are what get probed
    if not args.no_ema and model.apply_ema_weights():
        print("[probe] using the averaged (EMA) weights")
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    fit_positions, test_positions = split_by_position(
        dataset.recordings(), args.per_recording, args.test_fraction
    )
    fit_y = dataset.session_ids[fit_positions]
    test_y = dataset.session_ids[test_positions]
    print(
        f"[probe] {len(fit_positions)} fit / {len(test_positions)} test windows over "
        f"{len(np.unique(fit_y))} training sessions, on {device}"
    )

    fit_x = pooled_features(model, dataset, fit_positions, args.batch_size, device)
    test_x = pooled_features(model, dataset, test_positions, args.batch_size, device)

    real = fit_and_score(fit_x, fit_y, test_x, test_y)

    #the same probe on permuted session labels: what "nothing to find" scores
    generator = np.random.default_rng(cfg.seed)
    shuffled = fit_and_score(fit_x, generator.permutation(fit_y), test_x, test_y)

    #null SE of a balanced accuracy: each class's recall is a proportion at
    #the chance rate, and the mean over K classes divides the variance by K^2
    classes, counts = np.unique(test_y, return_counts=True)
    chance = 1.0 / len(classes)
    null_se = float(np.sqrt(np.sum(chance * (1 - chance) / counts)) / len(classes))

    print(f"[probe] session balanced accuracy   {real:.3f}")
    print(f"[probe] shuffled-label probe        {shuffled:.3f}")
    print(f"[probe] chance 1/{len(classes)} = {chance:.3f}, null SE ~{null_se:.3f}")

    #three SEs above the shuffled probe: a margin, not a significance test
    if real > shuffled + 3 * null_se:
        print(
            "[probe] session identity IS linearly decodable -- the adversary has "
            "something to remove. Keep it."
        )
    else:
        print(
            "[probe] no session identity beyond chance -- the adversary has "
            "nothing to push against. Drop it (session_adv_weight: 0.0)."
        )


if __name__ == "__main__":
    main()
