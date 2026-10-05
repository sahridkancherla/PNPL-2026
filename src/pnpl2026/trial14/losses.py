"""D-SigLIP between predicted word embeddings and a fixed t5 table. Trial 7's.

    L = -1/N sum_i sum_j log sigmoid( z_ij (t cos_ij + b) ),  z = +1 iff i = j

normalised by N, not N^2. `t` and `b` are learnable, initialised 10 and -10.

⚠️ The D is deduplication, d'Ascoli et al.'s: keep one occurrence of each word
in the batch, discard the rest, then plain SigLIP over what is left. Without
it a word's own target appears among its negatives -- `the` is ~6% of every
batch -- and the loss pulls and pushes it at once.

Here the table holds EVERY word's t5 embedding, not just the 50 scored ones,
so a batch's negatives are whatever other words it holds: hundreds of them.

Trial 14 adds the 50-word head's loss and the blend of the two score sets:

    logit_adjusted_cross_entropy   cross-entropy on logits + tau * log(prior).
                                   Training on the adjusted logits makes the
                                   PLAIN logits rank for balanced accuracy
                                   (Menon et al. 2021, "Long-tail learning via
                                   logit adjustment"), with no class weights --
                                   whose 121x span CLAUDE.md warns against
    blend_scores                   both score sets as log-probabilities, mixed
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def one_per_word(labels, shuffle):
    """Positions of one occurrence of each distinct word.

    ARGS:
        labels (torch.Tensor): [N] row ids
        shuffle (bool): pick a random occurrence (training), else the first,
            so evaluation is deterministic

    RETURNS:
        positions : [n_unique] int64 into `labels`
    """
    count = len(labels)
    order = (
        torch.randperm(count, device=labels.device)
        if shuffle
        else torch.arange(count, device=labels.device)
    )

    #the first occurrence of each label in `order`, found as a scatter-min of
    #positions: whichever copy the permutation put earliest survives
    _, inverse = torch.unique(labels[order], return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), count, dtype=torch.int64, device=labels.device)
    first = first.scatter_reduce(0, inverse, torch.arange(count, device=labels.device), reduce="amin")
    return order[first]


def siglip_loss(embeddings, labels, target_table, logit_scale, logit_bias, dedup=True, shuffle=True):
    """SigLIP over a batch of words, deduplicated by default, normalised by N.

    ARGS:
        embeddings (torch.Tensor): [N, embed_dim] predicted, unnormalised
        labels (torch.Tensor): [N] int64 row indices into `target_table`
        target_table (torch.Tensor): [n_words, embed_dim] fixed targets
        logit_scale (torch.Tensor): scalar temperature, already exponentiated
        logit_bias (torch.Tensor): scalar bias
        dedup (bool): keep one occurrence per word
        shuffle (bool): which occurrence survives is random; False in eval

    RETURNS:
        loss : scalar tensor
    """
    if dedup:
        keep = one_per_word(labels, shuffle)
        embeddings, labels = embeddings[keep], labels[keep]

    #unit vectors on both sides, so the product is a cosine
    predicted = F.normalize(embeddings, dim=1)
    targets = F.normalize(target_table[labels], dim=1)

    #[N, N]: row i is prediction i against every surviving word's target
    logits = predicted @ targets.t() * logit_scale + logit_bias

    #+1 on the diagonal, -1 everywhere else
    signs = 2.0 * torch.eye(len(labels), device=logits.device, dtype=logits.dtype) - 1.0

    #dim=1 sums over CANDIDATES, dim=0 averages over the batch: 1/N, not 1/N^2
    return -F.logsigmoid(signs * logits).sum(dim=1).mean(dim=0)


def semantic_logits(embeddings, target_table, logit_scale):
    """Score predicted embeddings against a set of candidate words.

    ARGS:
        embeddings (torch.Tensor): [batch, embed_dim] predicted, unnormalised
        target_table (torch.Tensor): [n_candidates, embed_dim]
        logit_scale (torch.Tensor): scalar temperature, already exponentiated

    RETURNS:
        logits : [batch, n_candidates] - scaled cosines; the ranking is retrieval
    """
    predicted = F.normalize(embeddings, dim=1)
    targets = F.normalize(target_table, dim=1)

    #the bias shifts every candidate equally and cannot change a ranking
    return (predicted @ targets.t()) * logit_scale


def logit_adjusted_cross_entropy(logits, labels, log_prior, tau=1.0, label_smoothing=0.0):
    """Cross-entropy on prior-adjusted logits, for a balanced-accuracy classifier.

    ARGS:
        logits (torch.Tensor): [N, n_classes] the head's plain logits
        labels (torch.Tensor): [N] int64 columns
        log_prior (torch.Tensor): [n_classes] log class frequencies in training
        tau (float): strength of the adjustment; 0 is plain cross-entropy
        label_smoothing (float)

    RETURNS:
        loss : scalar tensor
    """
    #unsqueeze(0) broadcasts the prior over the batch; dim=1 is the class axis
    adjusted = logits + tau * log_prior.unsqueeze(0)
    return F.cross_entropy(adjusted, labels, label_smoothing=label_smoothing)


def blend_scores(retrieval_logits, closed_logits, closed_weight):
    """Mix retrieval and head scores as log-probabilities.

    Both are log-softmaxed first: scaled cosines and logits are on different
    scales, and the louder one would otherwise win regardless of the weight.

    ARGS:
        retrieval_logits (torch.Tensor): [N, 50] scaled cosines
        closed_logits (torch.Tensor or None): [N, 50] the head's plain logits
        closed_weight (float): 0 is retrieval alone, 1 the head alone

    RETURNS:
        [N, 50] scores; only their ranking within a row matters
    """
    if closed_logits is None or closed_weight <= 0:
        return retrieval_logits.log_softmax(dim=1)
    if closed_weight >= 1:
        return closed_logits.log_softmax(dim=1)

    #dim=1 is the candidate axis in both
    return (1 - closed_weight) * retrieval_logits.log_softmax(dim=1) + closed_weight * closed_logits.log_softmax(dim=1)
