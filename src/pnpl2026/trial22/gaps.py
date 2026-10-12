"""The onset-gap voter: change 6 of the Route to 73.

The gap from a word's onset to the next listed onset approximates the
word's spoken length: "the" is short, "people" long. The holdout lists only
scored words, about a third of neighbouring pairs with one unscored word
between (`holdout-geometry`), so the voter is fitted on training sentences
seen the same way: chains of scored words with at most one unscored word
between neighbours and gaps of 0.8 s or less (`holdout_chains`).

    feature    x = log(gap to the next chain word / the chain's median gap)
               -- dividing by the median removes the speaker's rate
    model      per word, a Gaussian of x, shrunk towards the pooled one
    holdout    re-fitted by EM with the mix's soft labels (the holdout's
               speech may differ from the training narrators'), then each
               row scored by log N(x | word) for every word

The last word of a sentence has no next onset; a sentence of fewer than
three words has no rate to divide by. Their rows get no vote.
"""

from __future__ import annotations

import numpy as np

from .data.sequences import holdout_chains

#the chains the holdout serves: one skipped word at most, gaps under 0.8 s
MAX_SKIPPED = 1
MAX_GAP_SECONDS = 0.8

#a chain needs three words for its median gap to mean anything
MIN_CHAIN_WORDS = 3

#pseudo-tokens of the pooled Gaussian in each word's fit
SHRINK_TOKENS = 20.0

#the least spread a word's Gaussian may have, in log-gap units
MIN_SPREAD = 0.05


def chain_features(onsets_s, labels):
    """Each word's log gap ratio within one chain, every word but the last.

    ARGS:
        onsets_s (np.ndarray): [n] onsets in seconds, ascending
        labels (np.ndarray): [n] scored word of each, or anything

    RETURNS:
        features [n - 1] float64, labels [n - 1] -- empty for short chains
    """
    if len(onsets_s) < MIN_CHAIN_WORDS:
        return np.zeros(0), np.zeros(0, dtype=np.int64)

    #gaps to the next onset, divided by the chain's own median
    gaps = np.diff(np.asarray(onsets_s, dtype=np.float64))
    rate = np.median(gaps)
    features = np.log(np.maximum(gaps, 1e-3) / max(rate, 1e-3))
    return features, np.asarray(labels)[:-1].astype(np.int64)


def table_features(table, sample_rate):
    """Every chain's features in one recording's word table.

    ARGS:
        table (dict): onset_sample [n], sentence [n], primary [n], as
            `load_word_table` returns
        sample_rate (float): the table's onset rate, 250 for the events' own

    RETURNS:
        features [m], labels [m]
    """
    features, labels = [], []
    max_gap = int(round(MAX_GAP_SECONDS * sample_rate))
    sentences = np.asarray(table["sentence"])
    for sentence in np.unique(sentences[sentences >= 0]):
        #one sentence's words, in onset order, chained as the holdout lists them
        rows = np.flatnonzero(sentences == sentence)
        primary = np.asarray(table["primary"])[rows]
        onsets = np.asarray(table["onset_sample"])[rows]
        for chain in holdout_chains(primary, onsets, MAX_SKIPPED, max_gap):
            chain_x, chain_y = chain_features(onsets[chain] / sample_rate, primary[chain])
            features.append(chain_x)
            labels.append(chain_y)
    if not features:
        return np.zeros(0), np.zeros(0, dtype=np.int64)
    return np.concatenate(features), np.concatenate(labels)


class GapModel:
    """One Gaussian of the log gap ratio per word.

    ATTRS:
        means (np.ndarray): [50]
        spreads (np.ndarray): [50] standard deviations
        counts (np.ndarray): [50] training tokens behind each
    """

    def __init__(self, means, spreads, counts):
        """
        ARGS:
            means, spreads, counts (np.ndarray): [50] each
        """
        self.means = np.asarray(means, dtype=np.float64)
        self.spreads = np.maximum(np.asarray(spreads, dtype=np.float64), MIN_SPREAD)
        self.counts = np.asarray(counts, dtype=np.float64)

    @classmethod
    def fit(cls, features, labels, n_words=50, shrink=SHRINK_TOKENS):
        """Per-word Gaussians, each shrunk towards the pooled one.

        ARGS:
            features (np.ndarray): [m]
            labels (np.ndarray): [m] in 0..n_words-1
            n_words (int)
            shrink (float): pooled pseudo-tokens in each word's estimate

        RETURNS:
            GapModel
        """
        pooled_mean, pooled_variance = features.mean(), features.var()
        means, variances, counts = np.empty(n_words), np.empty(n_words), np.zeros(n_words)
        for word in range(n_words):
            own = features[labels == word]
            counts[word] = len(own)

            #the word's own moments, pulled towards the pooled ones
            own_mean = own.mean() if len(own) else pooled_mean
            own_variance = own.var() if len(own) > 1 else pooled_variance
            means[word] = (len(own) * own_mean + shrink * pooled_mean) / (len(own) + shrink)
            variances[word] = (len(own) * own_variance + shrink * pooled_variance) / (len(own) + shrink)
        return cls(means, np.sqrt(variances), counts)

    def log_likelihood(self, features):
        """Every row's log density under every word's Gaussian.

        ARGS:
            features (np.ndarray): [n]

        RETURNS:
            [n, 50] float64
        """
        #[n, 1] against [1, 50]: one density per row and word
        standard = (features[:, None] - self.means[None, :]) / self.spreads[None, :]
        return -0.5 * standard ** 2 - np.log(self.spreads)[None, :]

    def refit(self, features, soft_labels, n_iterations=10, shrink=SHRINK_TOKENS):
        """EM on unlabelled rows: the mix's soft labels as the prior, the Gaussians re-estimated.

        ARGS:
            features (np.ndarray): [n] the holdout rows with a gap
            soft_labels (np.ndarray): [n, 50] the mix's probabilities
            n_iterations (int)
            shrink (float): training pseudo-tokens kept in each word's fit

        RETURNS:
            GapModel, the holdout's; and the log-likelihood per iteration
        """
        model, history = self, []
        log_prior = np.log(np.maximum(soft_labels, 1e-12))
        for _ in range(n_iterations):
            #E: each row's word posterior from the mix and the gap together
            joint = log_prior + model.log_likelihood(features)
            peak = joint.max(axis=1, keepdims=True)
            history.append(float(np.sum(peak[:, 0] + np.log(np.exp(joint - peak).sum(axis=1)))))
            posterior = np.exp(joint - peak)
            posterior /= posterior.sum(axis=1, keepdims=True)

            #M: each word's moments from its soft rows, anchored to training
            mass = posterior.sum(axis=0)
            weighted_mean = (posterior * features[:, None]).sum(axis=0)
            means = (shrink * self.means + weighted_mean) / (shrink + mass)
            second = (posterior * (features[:, None] - means[None, :]) ** 2).sum(axis=0)
            variances = (shrink * self.spreads ** 2 + second) / (shrink + mass)
            model = GapModel(means, np.sqrt(variances), self.counts + mass)
        return model, history


def holdout_features(metadata):
    """The holdout's sentence rows' log gap ratios, from the loader's metadata.

    ARGS:
        metadata (list): `LibriBrainCompetitionHoldout.metadata`

    RETURNS:
        features [N] float64 (NaN where no vote), with every row's index
    """
    features = np.full(len(metadata), np.nan)
    by_sentence = {}
    for meta in metadata:
        if meta["source"] == "sentence":
            by_sentence.setdefault(meta["epoch"], []).append(meta)
    for rows in by_sentence.values():
        #slot order is the words' order; each word's gap to the next listed one
        rows = sorted(rows, key=lambda meta: meta["word"])
        chain_x, _ = chain_features([meta["onset_s"] for meta in rows], np.zeros(len(rows)))
        for meta, value in zip(rows[:-1], chain_x):
            features[meta["index"]] = value
    return features
