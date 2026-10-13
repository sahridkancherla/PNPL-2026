"""P4 of the 13 Oct research note: MEG-XL as a frozen feature extractor, residualized like P1/P3.

MEG-XL (Jayalath et al., ICML 2026; MIT, github.com/neural-processing-lab/MEG-XL,
weights pnpl/MEG-XL) reads 150 s at 50 Hz -- in its LibriBrain training form,
50 word windows of 3 s each (-0.5 to +2.5 s around an onset), each window
baseline-corrected, robust-scaled per channel and clipped to +/-5 on its own
(its `preprocess_segment_with_subsegments`) -- with every sensor's normalised
position, orientation and type. Its BioCodec tokenizer turns 12 samples into
one token, so 3 s is 12.5 tokens and 150 s is 625.

Here every holdout or testbed row becomes one such word window, rows in one
seeded shuffled order -- the note warns that MEG-XL's long context re-votes
the timing prior, and a shuffled order keeps sentence structure out of the
context -- and a row's features are the 3 tokens covering its own 0.5 s,
averaged over channels: [3, latent] per row, residualized on the timing
design afterwards like every P1/P3 feature.

Sensors: LibriBrain's fine-calibration file (metadata/neo/sss_cal.dat on the
pnpl/LibriBrain dataset) holds every channel's MNE `loc` -- position, then
the coil's three axes -- and MEG-XL's own loader reads exactly those: the
position, and the axis loc[9:12] for a magnetometer (coil 3024, type 1) or
loc[3:6] for a planar gradiometer (coil 3012, type 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

#samples per word window at 50 Hz, and before its onset (MEG-XL's -0.5 to +2.5 s)
WINDOW_SAMPLES = 150
ONSET_IN_WINDOW = 25

#word windows per 150 s segment
WINDOWS_PER_SEGMENT = 50

#BioCodec's downsampling: samples per time token
SAMPLES_PER_TOKEN = 12

#a row's feature window starts 2 samples before its onset (crossfeatures' -0.04 s)
FEATURE_START = -2

#tokens kept per row: 3 tokens cover the 0.5 s feature window exactly
TOKENS_PER_ROW = 3

#MEG-XL's sensor type codes
MAGNETOMETER, GRADIOMETER = 1, 0


def sensor_layout(fine_calibration_path, channel_names):
    """Every channel's position, orientation and type, in the data's channel order.

    ARGS:
        fine_calibration_path (Path): LibriBrain's sss_cal.dat
        channel_names (list): the data's 306 channels, e.g. "MEG0113"

    RETURNS:
        xyzdir [306, 6] float64 (position, then MEG-XL's orientation axis) and
        types [306] int64 (1 magnetometer, 0 gradiometer), unnormalised
    """
    import mne

    calibration = mne.preprocessing.read_fine_calibration(str(fine_calibration_path))
    locs = {name.replace(" ", ""): np.asarray(loc, dtype=np.float64) for name, loc in zip(calibration["ch_names"], calibration["locs"])}
    xyzdir, types = [], []
    for name in channel_names:
        key = name.replace(" ", "")
        if key not in locs:
            raise SystemExit(f"{fine_calibration_path}: no calibration entry for {name}.")
        loc = locs[key]

        #Elekta naming: MEGxxx1 is the magnetometer, xxx2 and xxx3 the planar pair
        is_magnetometer = key.endswith("1")
        axis = loc[9:12] if is_magnetometer else loc[3:6]
        xyzdir.append(np.concatenate([loc[:3], axis]))
        types.append(MAGNETOMETER if is_magnetometer else GRADIOMETER)
    return np.stack(xyzdir), np.asarray(types, dtype=np.int64)


def normalise_positions(xyzdir):
    """MEG-XL's `norm_sensor_positions`: positions centred, scaled by sqrt(3 x mean squared radius).

    ARGS:
        xyzdir (np.ndarray): [306, 6]

    RETURNS:
        [306, 6] float64, a copy; the orientations untouched
    """
    normalised = np.array(xyzdir, dtype=np.float64)
    #axis=0 is the sensor axis: one centre for the array
    normalised[:, :3] -= normalised[:, :3].mean(axis=0, keepdims=True)
    normalised[:, :3] /= np.sqrt(3.0 * np.mean(np.sum(normalised[:, :3] ** 2, axis=1)))
    return normalised


def word_window(piece, onset):
    """One row's 3 s window of a piece: 0.5 s before its onset to 2.5 s after, edges repeated.

    ARGS:
        piece (np.ndarray): [306, n] at 50 Hz
        onset (int): the row's onset sample in the piece

    RETURNS:
        [306, 150] float64
    """
    start = int(onset) - ONSET_IN_WINDOW
    padded = np.pad(np.asarray(piece, dtype=np.float64), ((0, 0), (WINDOW_SAMPLES, WINDOW_SAMPLES)), mode="edge")
    return padded[:, start + WINDOW_SAMPLES : start + 2 * WINDOW_SAMPLES]


def segment_order(n_rows, seed=0, shuffled=True, per_segment=WINDOWS_PER_SEGMENT):
    """The rows in segments of 50 windows; the last segment filled with repeats.

    ARGS:
        n_rows (int)
        seed (int)
        shuffled (bool): a seeded random order (default) or the rows' own
        per_segment (int): windows per segment; 50 is MEG-XL's 150 s

    RETURNS:
        list of [per_segment] int arrays of row indices; a repeat past the
        rows' end is marked -1 - row, so its features are dropped
    """
    order = np.random.default_rng(seed).permutation(n_rows) if shuffled else np.arange(n_rows)
    segments = []
    for start in range(0, n_rows, per_segment):
        chosen = order[start : start + per_segment]
        if len(chosen) < per_segment:
            #filler windows: real rows repeated so every segment is full length, then ignored
            filler = np.resize(order, per_segment - len(chosen))
            chosen = np.concatenate([chosen, -1 - filler])
        segments.append(chosen)
    return segments


def row_tokens(position):
    """The 3 tokens covering a window's 0.5 s feature window, by its place in the segment.

    ARGS:
        position (int): 0..49

    RETURNS:
        [3] int64 token indices
    """
    first = (position * WINDOW_SAMPLES + ONSET_IN_WINDOW + FEATURE_START) // SAMPLES_PER_TOKEN
    return np.arange(first, first + TOKENS_PER_ROW, dtype=np.int64)


def load_megxl(repo, checkpoint_path, tokenizer_path, device):
    """MEG-XL frozen, as its README loads it, and the repo's own preprocessing.

    ARGS:
        repo (Path): a clone of github.com/neural-processing-lab/MEG-XL
        checkpoint_path (Path): meg-xl-med-v2.ckpt (pnpl/MEG-XL)
        tokenizer_path (Path): the repo's brainstorm/neuro_tokenizers/biocodec_ckpt.pt
        device (torch.device)

    RETURNS:
        model (eval mode), preprocess (callable: [306, T], types -> [306, T]),
        and a dict of load diagnostics
    """
    import inspect

    import torch

    sys.path.insert(0, str(repo))
    from brainstorm.data.preprocessing import preprocess_segment_with_subsegments
    from brainstorm.models.criss_cross_transformer import CrissCrossTransformerModule
    from brainstorm.neuro_tokenizers.biocodec.model import BioCodecModel

    #the tokenizer: its factory, then the checkpoint with torch.compile's prefix removed
    tokenizer = BioCodecModel._get_optimized_model()
    tokenizer_state = torch.load(tokenizer_path, map_location=device, weights_only=False)["model_state_dict"]
    tokenizer.load_state_dict({key.replace("_orig_mod.", ""): value for key, value in tokenizer_state.items()})
    tokenizer.eval()

    #the model from its saved hyperparameters; RoPE tables are rebuilt, not loaded
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    accepted = set(inspect.signature(CrissCrossTransformerModule.__init__).parameters)
    hyperparameters = {key: value for key, value in dict(checkpoint["hyper_parameters"]).items() if key in accepted}
    model = CrissCrossTransformerModule(tokenizer=tokenizer, **hyperparameters).to(device)
    state = {key: value for key, value in checkpoint["state_dict"].items() if "rope_embedding_layer.rotate" not in key}
    result = model.load_state_dict(state, strict=False)
    model.eval()
    diagnostics = {"hyperparameters": {key: str(value) for key, value in hyperparameters.items()},
                   "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
                   "loaded_keys": len(state)}

    def preprocess(segment, types):
        """MEG-XL's own per-window baseline, robust scaling and clipping."""
        return preprocess_segment_with_subsegments(segment, types, 50.0, subsegment_duration=3.0, baseline_duration=0.5, clip_range=(-5, 5))

    return model, preprocess, diagnostics


def subject_features(model, preprocess, pieces, piece_of_row, onsets, layout, device, seed=0, shuffled=True,
                     per_segment=WINDOWS_PER_SEGMENT, bf16=False):
    """One subject's MEG-XL features of every row: [n_rows, 3, latent].

    ARGS:
        model: from `load_megxl`
        preprocess (callable): from `load_megxl`
        pieces (list): [306, n_i] at 50 Hz
        piece_of_row (np.ndarray): [n_rows]
        onsets (np.ndarray): [n_rows] onset sample within the row's piece
        layout (tuple): normalised xyzdir [306, 6], types [306]
        device (torch.device)
        seed (int), shuffled (bool): the segments' row order
        per_segment (int): windows per forward pass; 50 is MEG-XL's 150 s
        bf16 (bool): autocast on a GPU; off by default, as the README runs it --
            the tokenizer's codes are a nearest-codebook choice, so precision
            can flip them

    RETURNS:
        [n_rows, 3, latent] float32
    """
    import torch

    xyzdir, types = layout
    n_channels = len(types)
    sensor_xyz = torch.as_tensor(xyzdir[None, :, :3], dtype=torch.float32, device=device)
    sensor_abc = torch.as_tensor(xyzdir[None, :, 3:], dtype=torch.float32, device=device)
    sensor_types = torch.as_tensor(types[None], dtype=torch.long, device=device)
    sensor_mask = torch.ones((1, n_channels), dtype=torch.float32, device=device)
    features = None
    for segment in segment_order(len(piece_of_row), seed, shuffled, per_segment):
        #50 word windows, each processed on its own by MEG-XL's preprocessing
        rows = np.where(segment >= 0, segment, -1 - segment)
        windows = np.concatenate([word_window(pieces[piece_of_row[row]], onsets[row]) for row in rows], axis=1)
        meg = torch.as_tensor(preprocess(windows, types)[None], dtype=torch.float32, device=device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            output = model(meg, sensor_xyz, sensor_abc, sensor_types, sensor_mask, apply_mask=False)

        #[1, C, T', D] -> [T', D]: averaged over the channels
        pooled = output["features"][0].float().mean(dim=0).cpu().numpy()
        if features is None:
            features = np.zeros((len(piece_of_row), TOKENS_PER_ROW, pooled.shape[1]), dtype=np.float32)
        for position, row in enumerate(segment):
            if row >= 0:
                features[row] = pooled[row_tokens(position)]
    return features
