"""SigLIP between predicted word embeddings and a fixed t5 table. Variant D's.

    L = -1/N sum_i sum_j log sigmoid( z_ij (t cos_ij + b) ),  z = +1 iff i = j

normalised by N, not N^2 -- trial 5's N^2 left its semantic term near 1% of
the loss. `t` and `b` are learnable, initialised 10 and -10.

⚠️ The D is deduplication, and here it means DROPPING rows, not masking pairs:
keep one occurrence of each word in the batch, discard the rest, then compute
plain SigLIP over what is left. Trials 5 and 6 kept every row and masked
same-word pairs instead. With a closed set of 50 frequent words most of a
batch is repeats -- `the` alone is ~13% of tokens -- so without either, a word's
own target appears among its negatives and the loss pulls and pushes it at
once. D4 runs exactly that, plain SigLIP, to measure what the dedup is worth.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def one_per_word(labels, shuffle):
    """Positions of one occurrence of each distinct word.

    ARGS:
        labels (torch.Tensor): [N] class ids
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
        target_table (torch.Tensor): [n_classes, embed_dim] fixed targets
        logit_scale (torch.Tensor): scalar temperature, already exponentiated
        logit_bias (torch.Tensor): scalar bias
        dedup (bool): keep one occurrence per word. False is D4
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
    """Score a predicted embedding against every word in the vocabulary.

    ARGS:
        embeddings (torch.Tensor): [batch, embed_dim] predicted, unnormalised
        target_table (torch.Tensor): [n_classes, embed_dim] fixed targets
        logit_scale (torch.Tensor): scalar temperature, already exponentiated

    RETURNS:
        logits : [batch, n_classes] - scaled cosine similarities
    """
    predicted = F.normalize(embeddings, dim=1)
    targets = F.normalize(target_table, dim=1)

    #the bias shifts every class equally and cannot change a ranking
    return (predicted @ targets.t()) * logit_scale
