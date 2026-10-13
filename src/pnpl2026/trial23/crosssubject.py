"""Other subjects' recordings of subject 0's sentences, aligned into subject 0's sensor space.

Change 1 of the Route to 73, route 2. All 39 other subjects heard subject 0's
156 holdout sentences with identical onsets (`cli/repeat_check.py`), so a
matched pair of epochs is the same audio, sample for sample. Multiway CCA
(de Cheveigne et al., "Multiway canonical correlation analysis of brain
signals") finds the components every listener shares:

    1   each subject's epochs, robust-scaled, at 50 Hz, concatenated in
        subject 0's sentence order: X_k [306, T]
    2   per subject, PCA to `n_pcs` components and whitening: z_k [P, T]
    3   the PCA of all subjects' whitened components stacked: its leading
        eigenvectors are the shared components; V_k [P, C] is subject k's
        block, and y_k = V_k' z_k [C, T] is subject k's own estimate of them
    4   subject 0's sensors regressed on its own estimate: A_0 [306, C], so
        A_0 y_k is subject k's recording expressed as subject 0 would show it

Every step is linear, so subject k's whole map from raw MEG to subject 0's
raw MEG is one [306, 306] matrix and an offset (`SubjectMap`), applied at
250 Hz to whole epochs. Fitted on the holdout's own sentences, labels never
seen -- allowed since 12 Oct.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

#the holdout's rate and the models' rate
HOLDOUT_RATE = 250
MODEL_RATE = 50

#a channel this flat after the h5's own scaling is dead, not quiet
IQR_FLOOR = 1e-12

#keys a virtual sentence file copies from subject 0's; never `shuffle_seed`
COPIED_KEYS = ("sentence_n_times", "sentence_sample_mask", "word_onsets_s", "word_mask", "sfreq", "sentence_extra_end_s")


def load_matches(path):
    """The repeat check's match table.

    ARGS:
        path (Path or str): data/trial23_matches.json

    RETURNS:
        dict: subject (int) -> list of (epoch of subject 0, epoch of subject k)
    """
    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(subject): [tuple(pair) for pair in pairs] for subject, pairs in stored["subjects"].items()}


def holdout_file(holdout_dir, subject, source="sentence"):
    """Path of one subject's holdout file, downloaded if missing.

    ARGS:
        holdout_dir (str or None): folder holding COMPETITION_HOLDOUT/
        subject (int)
        source (str): "sentence" or "word"

    RETURNS:
        str
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    #pnpl's own resolver: a local copy if present, else the dataset
    holdout = LibriBrainCompetitionHoldout(subjects=[subject], sources=(source,), data_path=holdout_dir)
    return holdout._ensure_file(subject, source)


def sentence_file(holdout_dir, subject):
    """Path of one subject's sentence file, downloaded if missing.

    ARGS:
        holdout_dir (str or None): folder holding COMPETITION_HOLDOUT/
        subject (int)

    RETURNS:
        str
    """
    return holdout_file(holdout_dir, subject, "sentence")


def read_sentences(path):
    """One sentence file's MEG and metadata.

    ARGS:
        path (str)

    RETURNS:
        dict: meg [n, 306, T_max] float32 at 250 Hz, n_times [n] int64, and
        every key of COPIED_KEYS present in the file
    """
    with np.load(path, allow_pickle=False) as stored:
        loaded = {"meg": np.asarray(stored["meg"], dtype=np.float32)}
        for key in COPIED_KEYS:
            if key in stored.files:
                loaded[key] = np.asarray(stored[key])
    loaded["n_times"] = loaded["sentence_n_times"].astype(np.int64)
    return loaded


def downsample(raw):
    """250 Hz to 50 Hz, as prediction downsamples a holdout epoch.

    ARGS:
        raw (np.ndarray): [306, n] at 250 Hz

    RETURNS:
        [306, about n / 5] float64
    """
    #padtype "line": an epoch's edges were never edges of the recording
    return resample_poly(raw.astype(np.float64), MODEL_RATE, HOLDOUT_RATE, axis=1, padtype="line")


def robust_stats(segments):
    """Per-channel median and IQR, pooled over segments.

    ARGS:
        segments (list): [306, n_i] arrays

    RETURNS:
        median [306], iqr [306] float64
    """
    #axis=1 is time, pooled over every segment
    pooled = np.concatenate(segments, axis=1)
    q25, median, q75 = np.percentile(pooled, [25, 50, 75], axis=1)
    return median, np.maximum(q75 - q25, IQR_FLOOR)


def common_lengths(subjects_data, matches, reference):
    """Each matched sentence's length at 50 Hz, the shortest over all subjects.

    ARGS:
        subjects_data (dict): subject -> data from `read_sentences`
        matches (dict): from `load_matches`, restricted to the subjects used
        reference (dict): subject 0's data

    RETURNS:
        dict: epoch of subject 0 -> samples at 50 Hz every subject has
    """
    lengths = {}
    for subject, pairs in matches.items():
        for reference_epoch, other_epoch in pairs:
            #the shorter of the two epochs bounds the pair
            shortest = min(int(reference["n_times"][reference_epoch]), int(subjects_data[subject]["n_times"][other_epoch]))
            lengths[reference_epoch] = min(lengths.get(reference_epoch, shortest), shortest)
    return {epoch: length // (HOLDOUT_RATE // MODEL_RATE) for epoch, length in lengths.items()}


def stacked_epochs(data, epoch_order, lengths, pairs=None):
    """A subject's matched epochs at 50 Hz, robust-scaled, in subject 0's order.

    ARGS:
        data (dict): from `read_sentences`
        epoch_order (list): subject 0's matched epochs, in order
        lengths (dict): from `common_lengths`
        pairs (list or None): (epoch of subject 0, epoch of this subject);
            None for subject 0 itself

    RETURNS:
        stacked : [306, T] float64, scaled
        median, iqr : [306] - the scaling, fitted on these epochs
        bounds : [n_epochs + 1] int, where each sentence starts in `stacked`
    """
    own = dict(pairs) if pairs is not None else {epoch: epoch for epoch in epoch_order}
    pieces = []
    for epoch in epoch_order:
        #this subject's epoch of the same sentence, cut to the common length
        raw = data["meg"][own[epoch], :, : int(data["n_times"][own[epoch]])]
        pieces.append(downsample(raw)[:, : lengths[epoch]])

    #one scaler per subject, pooled over its matched sentences
    median, iqr = robust_stats(pieces)
    bounds = np.concatenate([[0], np.cumsum([piece.shape[1] for piece in pieces])])
    stacked = (np.concatenate(pieces, axis=1) - median[:, None]) / iqr[:, None]
    return stacked, median, iqr, bounds


class Whitener:
    """A subject's PCA to a few components, each scaled to unit variance.

    ATTRS:
        mean (np.ndarray): [306]
        basis (np.ndarray): [306, P] principal directions
        scale (np.ndarray): [P] their standard deviations
    """

    def __init__(self, scaled, n_pcs):
        """
        ARGS:
            scaled (np.ndarray): [306, T] a subject's scaled data
            n_pcs (int): components kept, below the data's rank
        """
        #axis=1 is time: one mean per channel, then the channel covariance
        self.mean = scaled.mean(axis=1)
        centred = scaled - self.mean[:, None]
        covariance = centred @ centred.T / centred.shape[1]
        values, vectors = np.linalg.eigh(covariance)

        #eigh sorts ascending: the largest n_pcs, largest first
        keep = np.argsort(values)[::-1][:n_pcs]
        self.basis = vectors[:, keep]
        self.scale = np.sqrt(np.maximum(values[keep], 1e-12))

    def __call__(self, scaled):
        """
        ARGS:
            scaled (np.ndarray): [306, n]

        RETURNS:
            [P, n] whitened components
        """
        return (self.basis.T @ (scaled - self.mean[:, None])) / self.scale[:, None]


def fit_mcca(whitened, n_components):
    """Shared components of every subject's whitened data.

    ARGS:
        whitened (list): per subject, [P, T] from its `Whitener`
        n_components (int): shared components kept, C

    RETURNS:
        blocks : list of [P, C], each subject's weights
        eigenvalues : [C] - shared variance; 1 means no more than one subject's
    """
    #every subject's components stacked: [n_subjects * P, T]
    stacked = np.concatenate(whitened, axis=0)
    covariance = stacked @ stacked.T / stacked.shape[1]
    values, vectors = np.linalg.eigh(covariance)

    #the leading eigenvectors, split back into one block per subject
    keep = np.argsort(values)[::-1][:n_components]
    vectors = vectors[:, keep]
    widths = np.cumsum([0] + [block.shape[0] for block in whitened])
    blocks = [vectors[widths[index] : widths[index + 1]] for index in range(len(whitened))]
    return blocks, values[keep]


def fit_forward_map(scaled_reference, shared_reference):
    """Subject 0's sensors as a linear function of its own shared components.

    ARGS:
        scaled_reference (np.ndarray): [306, T] subject 0's scaled data
        shared_reference (np.ndarray): [C, T] its shared-component estimate

    RETURNS:
        weights : [306, C], intercept : [306]
    """
    #least squares with an intercept column: [C + 1, T] regressors
    design = np.concatenate([shared_reference, np.ones((1, shared_reference.shape[1]))], axis=0)
    solution, *_ = np.linalg.lstsq(design.T, scaled_reference.T, rcond=None)
    return solution[:-1].T, solution[-1]


def channel_correlation(predicted, actual):
    """Mean over channels of the correlation over time.

    ARGS:
        predicted (np.ndarray): [306, T]
        actual (np.ndarray): [306, T]

    RETURNS:
        float
    """
    #axis=1 is time: one correlation per channel, then their mean
    predicted = predicted - predicted.mean(axis=1, keepdims=True)
    actual = actual - actual.mean(axis=1, keepdims=True)
    numerator = (predicted * actual).sum(axis=1)
    denominator = np.sqrt((predicted ** 2).sum(axis=1) * (actual ** 2).sum(axis=1)) + 1e-20
    return float(np.mean(numerator / denominator))


class SubjectMap:
    """One subject's raw 250 Hz MEG to subject 0's raw MEG, as one affine map.

    ATTRS:
        matrix (np.ndarray): [306, 306]
        offset (np.ndarray): [306]
    """

    def __init__(self, median, iqr, whitener, block, forward_weights, forward_intercept, reference_median, reference_iqr):
        """Compose scale -> whiten -> shared components -> subject 0's scaled sensors -> raw.

        ARGS:
            median, iqr (np.ndarray): [306] this subject's scaling
            whitener (Whitener): this subject's
            block (np.ndarray): [P, C] its MCCA weights
            forward_weights (np.ndarray): [306, C] subject 0's map, A_0
            forward_intercept (np.ndarray): [306]
            reference_median, reference_iqr (np.ndarray): [306] subject 0's scaling
        """
        #[306, P]: scaled sensors -> whitened components
        to_whitened = whitener.basis / whitener.scale[None, :]

        #[306, 306] in scaled units: A_0 V_k' W_k
        scaled_map = forward_weights @ block.T @ to_whitened.T

        #raw in: divide by this subject's IQR; raw out: multiply by subject 0's
        self.matrix = reference_iqr[:, None] * scaled_map / iqr[None, :]
        centre = scaled_map @ (median / iqr + whitener.mean)
        self.offset = reference_iqr * (forward_intercept - centre) + reference_median

    def __call__(self, raw):
        """
        ARGS:
            raw (np.ndarray): [306, n] at any rate, in the subject's raw units

        RETURNS:
            [306, n] float64 in subject 0's raw units
        """
        return self.matrix @ raw + self.offset[:, None]


def write_virtual(reference_path, out_dir, epochs, word_file):
    """A sentence file shaped exactly as subject 0's, holding other MEG.

    ARGS:
        reference_path (str): subject 0's sentence file, for its metadata
        out_dir (Path): the virtual holdout root; COMPETITION_HOLDOUT/ inside
        epochs (np.ndarray): [n, 306, T_max] float32, subject 0's shape
        word_file (str): subject 0's lone-word file, copied as it is

    RETURNS:
        Path of the sentence file written
    """
    import shutil

    folder = Path(out_dir) / "COMPETITION_HOLDOUT"
    folder.mkdir(parents=True, exist_ok=True)
    with np.load(reference_path, allow_pickle=False) as stored:
        #⚠️ the metadata decides every row's index: copied, never rebuilt
        metadata = {key: np.asarray(stored[key]) for key in COPIED_KEYS if key in stored.files}
        if np.asarray(stored["meg"]).shape != epochs.shape:
            raise SystemExit(f"Virtual epochs {epochs.shape} differ from subject 0's {stored['meg'].shape}.")
    target = folder / "subj00_holdout01_sentence.npz"
    np.savez(target, meg=epochs.astype(np.float32), **metadata)

    #the lone words cannot be matched by timing: subject 0's own
    shutil.copyfile(word_file, folder / "subj00_holdout2_word.npz")
    return target
