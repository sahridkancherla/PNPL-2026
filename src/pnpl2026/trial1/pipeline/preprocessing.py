

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass
class ChannelStandardiser:
    """
    Process and normalise all channels

    ATTRS:
        centre (np.ndarray): [n_channels] per-channel median
        scale (np.ndarray): [n_channels] per-channel IQR / NORMAL_IQR
    """

    centre: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(
        cls,
        #iterate through every recording
        recordings: Iterable[np.ndarray],
        scale_factor: int = 50,
        seed: int = 0):
        """
        Fit a centre and scale per channel over the training recordings.

        ARGS:
            recordings: iterable of [n_channel, times] arrays.
            scale_factor (int): keep one timepoint in this many. 1 keeps all
                of them and needs ~16 GB.
            seed (int): seed for the timepoint draw, so a rebuilt cache is
                reproducible.

        RETURNS:
            a fitted ChannelStandardiser
        """
        #preparation
        if scale_factor < 1:
            raise ValueError(f"subsample must be >= 1, got {scale_factor}")
        rng = np.random.default_rng(seed)

        #thin each recording
        thinned = []

        #iterate through every recording
        for recording in recordings:

            #find the number of time points measured
            n_times = recording.shape[1]

            #scale down the number of points we keep (so we don't store as much data)
            n_keep = n_times // scale_factor
            keep = rng.choice(n_times, size=n_keep, replace=False)

            #keep only the thinned data
            thinned.append(recording[:, keep])

        #concatenate all the data we kept along the channel axis.
        pooled = np.concatenate(thinned, axis=1)

        #v1: (x-mean)/std. temporary
        centre = np.mean(pooled, axis=1)
        scale = np.std(pooled, axis=1)

        return cls(centre=centre, scale=scale)

    def apply(self, recording: np.ndarray):
        """
        Standardise one recording.

        ARGS:
            recording (np.ndarray): [n_channels, n_times], continuous

        RETURNS:
            standardised : [n_channels, n_times] - float64, zero-centred,
                each channel with unit variance
        """

        #number of channels
        n_channels = recording.shape[0]

        #check for dimension mismatches
        if n_channels != len(self.centre):
            raise ValueError(
                f"Expected {len(self.centre)} channels, got {n_channels}. ")

        #(x - mean)/std as a temporary formula
        normalised = recording.astype(np.float64, copy=False) - self.centre[:, None]
        return normalised / self.scale[:, None]


@dataclass
class Whitener:
    """
    Whiten and standardise data.

    ATTRS:
        matrix (np.ndarray): [n_components, n_channels] - apply as matrix @ x
        eigenvalues (np.ndarray): [n_channels] - the FULL spectrum, largest
            first. Kept whole rather than truncated: seeing past the cut is
            what justifies where the cut was made.
        shrinkage (float): fraction of the mean eigenvalue added to all of
            them before inverting. Provenance -- not recoverable from matrix.
    """

    matrix: np.ndarray
    eigenvalues: np.ndarray
    shrinkage: float

    @classmethod
    def fit(
        cls,
        recordings: Iterable[np.ndarray],
        standardiser: ChannelStandardiser,
        n_components: int = 73,
        shrinkage: float = 0.01):
        """
        Fit a whitener over the standardised training recordings.

        ARGS:
            recordings: iterable of [n_channels, n_times] arrays, raw. They are
                standardised here, so do not pre-apply the standardiser.
            standardiser: a fitted ChannelStandardiser, applied per recording.
            n_components (int): directions to keep. 73 is the measured rank.
            shrinkage (float): floor added to the eigenvalues before inverting,
                as a fraction of their mean. Bounds the 1/sqrt gain.

        RETURNS:
            a fitted Whitener
        """

        #fetch covariance and number of time points data
        covariance, N = cls.pooled_covariance(recordings, standardiser)

        #perform whitening
        matrix, eigenvalues = cls.whitening_matrix(covariance, n_components, shrinkage)

        #perform a sanity check
        cls.cutoff_sanity_check(eigenvalues, n_components, N)

        #return
        return cls(matrix=matrix, eigenvalues=eigenvalues, shrinkage=shrinkage)

    def apply(self, standardised: np.ndarray) -> np.ndarray:
        """
        Whiten one STANDARDISED recording.

        ARGS:
            standardised (np.ndarray): [n_channels, n_times], already through
                ChannelStandardiser.apply

        RETURNS:
            whitened : [n_components, n_times] - uncorrelated, unit variance
        """
        n_channels = standardised.shape[0]

        #check for dimension mismatches
        if n_channels != self.matrix.shape[1]:
            raise ValueError(
                f"Expected {self.matrix.shape[1]} channels, got {n_channels}. "
                "The whitener was fitted on a different sensor set."
            )

        #one matmul: [n_components, n_channels] @ [n_channels, n_times]
        return self.matrix @ standardised

    @staticmethod
    def pooled_covariance(recordings, standardiser):
        """
        Accumulate one covariance across every recording, streaming.

        ARGS:
            recordings: iterable of [n_channels, n_times] arrays, raw.
            standardiser: a fitted ChannelStandardiser, applied per recording. Its
                pooled centre is what makes the sum accumulable -- re-centring each
                recording would erase the offsets between sessions.

        RETURNS:
            covariance : [n_channels, n_channels]
            n_times_seen (int): total timepoints the covariance was built from
        """
        n_channels = len(standardiser.centre)

        #prepare shapes before finding covariance
        #scatter = will be used for covariance. N = number of time points 
        scatter = np.zeros((n_channels, n_channels))
        N = 0

        #prepare data necessary to find covariance matrix
        for recording in recordings:
            standardised = standardiser.apply(recording)

            #X @ X.T and N for 1/N later in typical covariance matrix formula
            scatter += standardised @ standardised.T
            N += standardised.shape[1]

        #error check
        if N == 0:
            raise ValueError("No recordings were given to fit on.")
        
        #covariance matrix
        covariance = scatter / N

        return covariance, N

    @staticmethod
    def whitening_matrix(covariance, n_components, shrinkage):
        """
        PCA whitening

        ARGS:
            covariance (np.ndarray): [n_channels, n_channels], symmetric
            n_components (int): directions to keep, largest variance first
            shrinkage (float): floor added to every eigenvalue before inverting,
                as a fraction of their mean

        RETURNS:
            matrix : [n_components, n_channels] - apply as matrix @ x
            eigenvalues : [n_channels] - the FULL spectrum, largest first
        """
        #find number of channels
        n_channels = covariance.shape[0]

        #error check
        if not 1 <= n_components <= n_channels:
            raise ValueError(
                f"n_components must be in 1..{n_channels}, got {n_components}"
            )
        if shrinkage < 0:
            raise ValueError(f"shrinkage must be >= 0, got {shrinkage}")

        #eigh returns eigenvalues and eigenvectors in ascending order
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        #sort so the largest eigenvalues are first
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        #keep only the largest eigenvalues (n_components is the index cutoff)
        kept_values = eigenvalues[:n_components]
        kept_vectors = eigenvectors[:, :n_components]

        #whitening divides by sqrt(eigenvalue) so make a floor to avoid explosions
        floor = shrinkage * eigenvalues.mean()

        #apply PCA whitening
        matrix = (kept_vectors).T / np.sqrt(kept_values + floor)
        return matrix, eigenvalues

    @staticmethod
    def cutoff_sanity_check(eigenvalues, n_components, n_times_seen):
        """
        Print what the spectrum looked like and where the cut fell.

        Without this the rank choice is invisible: a cut in the wrong place still
        produces a healthy-looking whitener. The gap across the cut is the evidence
        that the cut is in the right place -- a large one means the dropped
        directions really were empty, and ~1x means you sliced through noise.

        ARGS:
            eigenvalues (np.ndarray): [n_channels] the full spectrum, largest first
            n_components (int): where the cut was made
            n_times_seen (int): timepoints the covariance was built from
        """
        n_channels = len(eigenvalues)
        #normalise the cumulative sum to observe the % of variance kept
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()

        #signal variance kept by the cutoff
        print(f"variance kept {cumulative[n_components - 1]:.4%}")

        if n_components > n_channels:
            raise ValueError(f"n_components expected to be smaller than n_channels")

        print(
            f"last kept eigenvalue {eigenvalues[n_components - 1]:.3e}, "
            f"first dropped eigenvalue {eigenvalues[n_components]:.3e}, "
            f"ratio between them {eigenvalues[n_components - 1] / eigenvalues[n_components]:.1f}x"
            )
