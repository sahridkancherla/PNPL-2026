"""Transductive inference over the holdout's rows: change 2 of the Route to 73.

TransCLIP (Zanella et al. 2024, arXiv 2406.01837) boosts a zero-shot
retrieval model by treating the test set as one: a Gaussian mixture over the
rows' features with one cluster per class, a KL pull of each row's soft
assignment towards the model's own zero-shot scores, and a graph term giving
similar rows similar assignments. The doc's version for 868 holdout rows:

    - one cluster per word, started at the word's prototype (its mean
      feature on local data) where one is given
    - the KL pull towards the member's original scores, at strength
      `pull` (the doc submits two strengths)
    - the graph built from each row's nearest rows in OTHER sentences only:
      rows of one sentence look alike because the transformer mixes context
    - word shares estimated from the rows, floored -- not fixed equal:
      transductive methods assuming balance lose on imbalanced sets
      (Veilleux et al. 2021)

The block majorise-minimise updates, per outer round:

    z_i  = softmax( log yhat_i + [log pi + log N(f_i | mu, Sigma) / t + sum_j w_ij z_j] / pull )
    mu_k = sum_i z_ik f_i / sum_i z_ik          Sigma = shared diagonal
    pi_k = max(floor, mean_i z_ik), renormalised

`pull` is the KL term's weight: minimising -z.(terms) + pull * KL(z || yhat)
gives exactly the z above, so pull 1 is TransCLIP's own balance and a large
pull returns yhat. (An exponent on yhat would SHARPEN it instead -- found by
the synthetic test.) `t`, the likelihood temperature, is set without labels
so the first posterior is exactly as confident as yhat: a 50-word Gaussian
likelihood in many dimensions is otherwise sharp enough to override every
model.

The output ranks by log z - log pi: for BALANCED accuracy a row should rank
words by how much more likely the row makes them than their share does --
the estimated shares shape the clustering, not the ranking.

Everything is deterministic: no sampling anywhere.
"""

from __future__ import annotations

import numpy as np

#a probability this small is treated as zero inside logarithms
FLOOR = 1e-12


def softmax_rows(logits):
    """Row-wise softmax.

    ARGS:
        logits (np.ndarray): [N, K]

    RETURNS:
        [N, K] float64, each row a distribution
    """
    #axis=1 is the class axis; shifted by the row maximum for stability
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def zscore_rows(scores):
    """Each row centred and scaled across its classes.

    ARGS:
        scores (np.ndarray): [N, K]

    RETURNS:
        [N, K] float64
    """
    #axis=1 is the class axis; a flat row stays flat
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = centred.std(axis=1, keepdims=True)
    return centred / np.where(spread > 0, spread, 1.0)


def fit_temperature_nll(scores, labels, bounds=(0.05, 20.0), steps=60):
    """The temperature minimising the labelled rows' negative log-likelihood.

    ARGS:
        scores (np.ndarray): [N, K] standardised scores of labelled rows
        labels (np.ndarray): [N] class indices
        bounds (tuple): search range, in log space
        steps (int): golden-section iterations

    RETURNS:
        float
    """
    def loss(log_temperature):
        #mean NLL of the true class at this temperature
        probabilities = softmax_rows(scores / np.exp(log_temperature))
        return -np.mean(np.log(np.maximum(probabilities[np.arange(len(labels)), labels], FLOOR)))

    #golden-section search on a unimodal one-dimensional loss
    low, high = np.log(bounds[0]), np.log(bounds[1])
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    for _ in range(steps):
        left, right = high - ratio * (high - low), low + ratio * (high - low)
        if loss(left) < loss(right):
            high = right
        else:
            low = left
    return float(np.exp(0.5 * (low + high)))


def fit_temperature_confidence(scores, confidence=0.3, bounds=(0.05, 20.0), steps=60):
    """The temperature at which the mean top-1 probability equals `confidence`.

    Label-free: used where no labelled rows exist. 0.3 is about the mix's
    likely top-1 accuracy on holdout rows, as trial 21's distillation chose.

    ARGS:
        scores (np.ndarray): [N, K]
        confidence (float)
        bounds (tuple), steps (int): as `fit_temperature_nll`

    RETURNS:
        float
    """
    low, high = np.log(bounds[0]), np.log(bounds[1])
    for _ in range(steps):
        #the mean top-1 probability falls as the temperature rises
        middle = 0.5 * (low + high)
        if softmax_rows(scores / np.exp(middle)).max(axis=1).mean() > confidence:
            low = middle
        else:
            high = middle
    return float(np.exp(0.5 * (low + high)))


def cross_sentence_graph(features, groups, n_neighbours=3):
    """Affinities to each row's nearest rows in other sentences.

    ARGS:
        features (np.ndarray): [N, D]
        groups (np.ndarray): [N] sentence id of each row
        n_neighbours (int): neighbours kept per row

    RETURNS:
        [N, N] float64, symmetric, non-negative, zero within a sentence
    """
    #cosine similarities between rows; axis=1 normalises each row's feature
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), FLOOR)
    similarity = unit @ unit.T

    #⚠️ same-sentence pairs and a row with itself never count
    same = groups[:, None] == groups[None, :]
    similarity = np.where(same, -np.inf, similarity)
    affinity = np.zeros_like(similarity)
    keep = min(n_neighbours, max(int((~same).sum(axis=1).min()), 0))
    if keep == 0:
        return affinity
    nearest = np.argpartition(-similarity, keep - 1, axis=1)[:, :keep]
    rows = np.repeat(np.arange(len(features)), keep)
    affinity[rows, nearest.ravel()] = np.maximum(similarity[rows, nearest.ravel()], 0.0)

    #symmetrised, as TransCLIP's graph term assumes
    return 0.5 * (affinity + affinity.T)


def gaussian_log_likelihood(features, means, variances):
    """Log density of every row under every class's Gaussian, shared diagonal covariance.

    ARGS:
        features (np.ndarray): [N, D]
        means (np.ndarray): [K, D]
        variances (np.ndarray): [D]

    RETURNS:
        [N, K] float64, up to a constant shared by every class
    """
    #(f - mu)^2 / var summed over dimensions, expanded to avoid an [N, K, D] array
    scaled_features = features / np.sqrt(variances)[None, :]
    scaled_means = means / np.sqrt(variances)[None, :]
    squared = (scaled_features ** 2).sum(axis=1)[:, None] - 2.0 * scaled_features @ scaled_means.T + (scaled_means ** 2).sum(axis=1)[None, :]
    return -0.5 * squared


def likelihood_temperature(log_likelihood, log_prior, target_confidence, bounds=(1e-3, 1e4), steps=60):
    """The divisor that makes the likelihood's posterior as confident as the zero-shot scores.

    ARGS:
        log_likelihood (np.ndarray): [N, K]
        log_prior (np.ndarray): [K]
        target_confidence (float): mean top-1 probability of yhat
        bounds (tuple), steps (int): search range in log space

    RETURNS:
        float
    """
    low, high = np.log(bounds[0]), np.log(bounds[1])
    for _ in range(steps):
        #a larger divisor flattens the posterior
        middle = 0.5 * (low + high)
        posterior = softmax_rows(log_likelihood / np.exp(middle) + log_prior[None, :])
        if posterior.max(axis=1).mean() > target_confidence:
            low = middle
        else:
            high = middle
    return float(np.exp(0.5 * (low + high)))


def transduce(features, zero_shot, groups, pull=1.0, prototypes=None, n_neighbours=3, share_floor=0.005,
              n_outer=10, n_inner=5, variance_floor=1e-3):
    """TransCLIP-style transduction with estimated, floored class shares.

    ARGS:
        features (np.ndarray): [N, D] one row per holdout item
        zero_shot (np.ndarray): [N, K] the member's own probabilities, yhat
        groups (np.ndarray): [N] sentence id per row, for the graph
        pull (float): the KL term's weight; 1 is TransCLIP's, larger stays closer to yhat
        prototypes (np.ndarray or None): [K, D] starting means; NaN rows,
            or None, start from yhat-weighted means of the rows
        n_neighbours (int): graph neighbours per row
        share_floor (float): the least share any class may take
        n_outer (int), n_inner (int): rounds of the updates
        variance_floor (float): per dimension, against collapse

    RETURNS:
        dict: posterior [N, K], shares [K], means [K, D], temperature float
    """
    n_rows, n_classes = zero_shot.shape
    log_yhat = np.log(np.maximum(zero_shot, FLOOR))
    affinity = cross_sentence_graph(features, groups, n_neighbours)

    #starting means: the prototypes where given, else yhat-weighted row means
    weighted = (zero_shot.T @ features) / np.maximum(zero_shot.sum(axis=0), FLOOR)[:, None]
    means = weighted.copy()
    if prototypes is not None:
        given = np.isfinite(prototypes).all(axis=1)
        means[given] = prototypes[given]
    variances = np.maximum(features.var(axis=0), variance_floor)
    shares = np.full(n_classes, 1.0 / n_classes)

    #the likelihood's temperature, fixed once, label-free
    target = zero_shot.max(axis=1).mean()
    temperature = likelihood_temperature(gaussian_log_likelihood(features, means, variances), np.log(shares), target)

    posterior = zero_shot.copy()
    for _ in range(n_outer):
        log_likelihood = gaussian_log_likelihood(features, means, variances) / temperature
        for _ in range(n_inner):
            #z: yhat, then the shares, the likelihood and the graph, divided by the pull
            evidence = np.log(shares)[None, :] + log_likelihood + affinity @ posterior
            posterior = softmax_rows(log_yhat + evidence / pull)

        #the means and the shared diagonal covariance from the soft assignments
        mass = np.maximum(posterior.sum(axis=0), FLOOR)
        means = (posterior.T @ features) / mass[:, None]
        residual = (features ** 2).sum(axis=0) - (mass[:, None] * means ** 2).sum(axis=0)
        variances = np.maximum(residual / n_rows, variance_floor)

        #shares from the rows, floored so a rare word is never buried
        shares = np.maximum(posterior.mean(axis=0), share_floor)
        shares = shares / shares.sum()
    return {"posterior": posterior, "shares": shares, "means": means, "temperature": temperature}


def blend(original, posterior, shares, weight=0.5):
    """The transduced scores mixed back with the original ones.

    ARGS:
        original (np.ndarray): [N, K] standardised scores
        posterior (np.ndarray): [N, K] from `transduce`
        shares (np.ndarray): [K] the estimated shares, divided out
        weight (float): the transduced part's share

    RETURNS:
        [N, K] float64, row-standardised
    """
    #the posterior over the share: balanced accuracy's ranking
    transduced = zscore_rows(np.log(np.maximum(posterior, FLOOR)) - np.log(shares)[None, :])

    #both sides row-standardised, so the weight means what it says
    return zscore_rows((1.0 - weight) * zscore_rows(original) + weight * transduced)


def same_top1_separation(features, scores, groups, max_pairs=200000, seed=0):
    """Do rows with the same top-1 word sit closer than other rows? (cross-sentence pairs only)

    ARGS:
        features (np.ndarray): [N, D]
        scores (np.ndarray): [N, K]
        groups (np.ndarray): [N] sentence ids
        max_pairs (int): pairs sampled when there are more
        seed (int)

    RETURNS:
        same : float, mean cosine of same-top-1 pairs; other : float, of the rest
    """
    unit = features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), FLOOR)
    top = scores.argmax(axis=1)
    first, second = np.triu_indices(len(features), k=1)

    #cross-sentence pairs only: one sentence's rows share context, not words
    keep = groups[first] != groups[second]
    first, second = first[keep], second[keep]
    if len(first) > max_pairs:
        chosen = np.random.default_rng(seed).choice(len(first), max_pairs, replace=False)
        first, second = first[chosen], second[chosen]
    cosine = np.sum(unit[first] * unit[second], axis=1)
    same = top[first] == top[second]
    return float(cosine[same].mean()) if same.any() else float("nan"), float(cosine[~same].mean())
