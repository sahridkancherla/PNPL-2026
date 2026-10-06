"""D-SigLIP between predicted word embeddings and a fixed t5 table. Trial 7's.

    L = -1/N sum_i sum_j log sigmoid( z_ij (t cos_ij + b) ),  z = +1 iff i = j

normalised by N, not N^2. `t` and `b` are learnable, initialised 10 and -10.

⚠️ The D is deduplication, d'Ascoli et al.'s: keep one occurrence of each word
in the batch, discard the rest, then plain SigLIP over what is left. Without
it a word's own target appears among its negatives -- `the` is ~6% of every
batch -- and the loss pulls and pushes it at once.

Here the table holds EVERY word's t5 embedding, not just the 50 scored ones,
so a batch's negatives are whatever other words it holds: hundreds of them.
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
