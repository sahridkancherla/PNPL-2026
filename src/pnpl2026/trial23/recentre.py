"""P8b of the 13 Oct research note: re-centre a session on the training recordings (Euclidean alignment).

Each session's channel covariance, in the scaled space the models read, is
mapped onto the training recordings' mean covariance:

    M = C_ref^(1/2) C_sess^(-1/2)        x' = M x

both square roots taken on their top 90 eigendirections: SSS leaves the 306
channels nearer rank 96, and a direction past that is numerical noise
(CLAUDE.md, "The input is rank 73 -- or nearer 96"). A session is a holdout
source group -- subject 0's sentence epochs, or its lone words -- or one
cached recording in the local check.

The note: EA helped shared EEG models across subjects (Junqueira et al.,
2024) but fell below the unadapted baseline cross-session, and BatchNorm
re-estimation already hurt here. So it is gated on subject 0's own held-out
sessions (cli/recentre.py check) before any submission.
"""

from __future__ import annotations

import numpy as np

#eigendirections kept on both sides
N_DIRECTIONS = 90

#samples read per recording for a covariance, spread evenly over it
SAMPLES_PER_RECORDING = 50_000

#eigenvalue floor before a negative square root
EIGEN_FLOOR = 1e-12


def covariance(segments):
    """The channel covariance pooled over segments, each channel's mean removed.

    ARGS:
        segments (list): [306, n_i] arrays

    RETURNS:
        [306, 306] float64
    """
    #axis=1 is time, pooled over every segment
    pooled = np.concatenate([np.asarray(segment, dtype=np.float64) for segment in segments], axis=1)
    centred = pooled - pooled.mean(axis=1, keepdims=True)
    return centred @ centred.T / centred.shape[1]


def recording_covariance(meg, max_samples=SAMPLES_PER_RECORDING):
    """One cached recording's covariance from evenly spaced samples.

    ARGS:
        meg (np.ndarray or memmap): [n_times, 306]
        max_samples (int)

    RETURNS:
        [306, 306] float64
    """
    step = max(meg.shape[0] // max_samples, 1)
    return covariance([np.asarray(meg[::step], dtype=np.float64).T])


def top_eigen(matrix, n_directions):
    """The largest eigenvalues and their vectors of a symmetric matrix.

    ARGS:
        matrix (np.ndarray): [306, 306]
        n_directions (int)

    RETURNS:
        values [n_directions] (floored), vectors [306, n_directions]
    """
    values, vectors = np.linalg.eigh(matrix)
    #eigh sorts ascending: the largest first
    keep = np.argsort(values)[::-1][:n_directions]
    return np.maximum(values[keep], EIGEN_FLOOR), vectors[:, keep]


def recentring_transform(session_covariance, reference_covariance, n_directions=N_DIRECTIONS):
    """M = C_ref^(1/2) C_sess^(-1/2), each on its own top eigendirections.

    ARGS:
        session_covariance (np.ndarray): [306, 306]
        reference_covariance (np.ndarray): [306, 306]
        n_directions (int)

    RETURNS:
        [306, 306] float64
    """
    session_values, session_vectors = top_eigen(session_covariance, n_directions)
    reference_values, reference_vectors = top_eigen(reference_covariance, n_directions)

    #whiten the session in its subspace, then colour it as the reference
    whiten = (session_vectors * session_values ** -0.5) @ session_vectors.T
    colour = (reference_vectors * reference_values ** 0.5) @ reference_vectors.T
    return colour @ whiten


class RecentredRecording:
    """A cached recording [n_times, 306], every slice read through a re-centring transform."""

    def __init__(self, meg, transform):
        """
        ARGS:
            meg (np.memmap): [n_times, 306]
            transform (np.ndarray): [306, 306], applied to each sample
        """
        self.meg = meg
        self.shape = meg.shape
        self.transposed = transform.T.astype(np.float32)

    def __getitem__(self, index):
        """
        ARGS:
            index: anything a memmap takes on its time axis

        RETURNS:
            [n, 306] float32, re-centred
        """
        #[n, 306] @ [306, 306]: each sample's channels mixed by the transform
        return np.asarray(self.meg[index], dtype=np.float32) @ self.transposed


def load_reference(path):
    """The training recordings' mean covariance, from `cli.recentre reference`.

    ARGS:
        path (Path)

    RETURNS:
        [306, 306] float64, and the directions it was written for (int)
    """
    with np.load(path) as stored:
        return stored["covariance"].astype(np.float64), int(stored["n_directions"])


def recentre_dataset(recordings, reference_covariance, n_directions=N_DIRECTIONS):
    """Every recording of a dataset re-centred: a transform each, from its own data.

    The transforms are a plain dict on the recordings object, so DataLoader
    workers inherit them; `ReferenceRecordings.meg` applies them.

    ARGS:
        recordings (ReferenceRecordings): either family's
        reference_covariance (np.ndarray): [306, 306]
        n_directions (int)

    RETURNS:
        int: recordings re-centred
    """
    recordings.transforms = {}
    for index in range(len(recordings.records)):
        session = recording_covariance(recordings.meg(index))
        recordings.transforms[index] = recentring_transform(session, reference_covariance, n_directions)
    return len(recordings.transforms)


def recentring_scaler_factory(scaler_for, reference_covariance, n_directions=N_DIRECTIONS):
    """A holdout scaler factory whose scalers also re-centre their group.

    The predictors call the factory once per source group -- a subject's
    sentence epochs, then its lone words -- so each group is one session.

    ARGS:
        scaler_for (callable): segments -> scaler, from `scaler_factory`
        reference_covariance (np.ndarray): [306, 306]
        n_directions (int)

    RETURNS:
        callable: segments -> (segment -> scaled, re-centred float32)
    """
    def factory(segments):
        """The group's scaler, then its re-centring fitted on the scaled group."""
        scaler = scaler_for(segments)
        transform = recentring_transform(covariance([scaler(segment) for segment in segments]), reference_covariance,
                                         n_directions).astype(np.float32)
        return lambda segment: (transform @ scaler(segment)).astype(np.float32)

    return factory
