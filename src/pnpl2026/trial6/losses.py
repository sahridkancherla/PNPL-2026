"""D-SigLIP: the contrastive loss the reference model is built on.

SigLIP replaces CLIP's softmax over a batch with an independent sigmoid per
pair, which removes the need for a global normalisation and works at small batch
sizes. The **D** is d'Ascoli et al.'s modification, and on this data it is the
part that matters: *discard the repetitions*.

⚠️ Why the D is not optional here. `the` accounts for 41,143 of the 313,068
in-vocabulary windows, so a batch of 128 contains it about 17 times. Plain
in-batch contrastive treats those 17 windows as negatives *of each other* --
it actively teaches the model that two instances of `the` should be far apart.
Their ablation measured exactly this: CLIP 15.7, SigLIP 14.7, **D-SigLIP 18.2**
top-10 accuracy, p<0.005.

The loss is between a predicted embedding and a FIXED table of target
embeddings, one row per word. Nothing here learns the targets.

⚠️ Trial 5's normalisation was wrong, and Variant C's C7 is the fix. It took the
mean over every kept PAIR, which divides by ~batch^2: at batch 512 the loss
sat near 0.015 against a closed-set term near 1, so the semantic head barely
reached the trunk. SigLIP's own loss sums over each row's candidates and
averages over rows -- 1/|B| sum_i sum_j -- which is ~batch times larger.
`sum_over_candidates` selects it; False keeps trial 5's, for C0.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def d_siglip_loss(
    embeddings,
    labels,
    target_table,
    logit_scale,
    logit_bias,
    sum_over_candidates=False,
):
    """Pairwise sigmoid contrastive loss, with same-word pairs discarded.

    ARGS:
        embeddings (torch.Tensor): [batch, embed_dim] predicted, unnormalised
        labels (torch.Tensor): [batch] int64 row indices into `target_table`
        target_table (torch.Tensor): [n_classes, embed_dim] fixed targets
        logit_scale (torch.Tensor): scalar, the learnable temperature. Held in
            log space by the caller and exponentiated before it arrives here
        logit_bias (torch.Tensor): scalar, the learnable bias
        sum_over_candidates (bool): True is SigLIP's normalisation, sum over
            each row's kept candidates then mean over rows. False is trial
            5's mean over every kept pair

    RETURNS:
        loss : scalar tensor
    """
    #both sides are unit-normalised, so the product below is a cosine and the
    #temperature is the only thing setting the logits' range
    predicted = F.normalize(embeddings, dim=1)
    targets = F.normalize(target_table[labels], dim=1)

    #[batch, batch]: row i is how much prediction i looks like target j
    similarity = predicted @ targets.t()
    logits = similarity * logit_scale + logit_bias

    #+1 where the pair should match, -1 where it should not
    matches = labels.unsqueeze(1) == labels.unsqueeze(0)
    signs = torch.where(matches, 1.0, -1.0).to(logits.dtype)

    #THE D: an off-diagonal pair carrying the same word is neither a positive
    #(it is a different trial) nor a negative (it is the same word), so it is
    #dropped. The diagonal always survives
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    keep = identity | ~matches

    #logsigmoid rather than log(sigmoid(.)): the fused version does not
    #underflow when the argument is a large negative number
    per_pair = -F.logsigmoid(signs * logits) * keep

    #dim=1 is the CANDIDATE axis, dim=0 the batch: every row's evidence is
    #summed, then rows are averaged, so the scale does not shrink with batch
    if sum_over_candidates:
        return per_pair.sum(dim=1).mean(dim=0)

    #trial 5's version: averaged over the pairs that survived
    return per_pair.sum() / keep.sum().clamp(min=1)


def semantic_logits(embeddings, target_table, logit_scale):
    """Score a predicted embedding against every word in the vocabulary.

    This is how the semantic head produces something the metric can read: not a
    classifier layer, but a cosine similarity against the 50 fixed target
    embeddings. It is also exactly what the reference model does at inference.

    ARGS:
        embeddings (torch.Tensor): [batch, embed_dim] predicted, unnormalised
        target_table (torch.Tensor): [n_classes, embed_dim] fixed targets
        logit_scale (torch.Tensor): scalar temperature, already exponentiated

    RETURNS:
        logits : [batch, n_classes] - scaled cosine similarities
    """
    predicted = F.normalize(embeddings, dim=1)
    targets = F.normalize(target_table, dim=1)

    #the bias is deliberately absent: it shifts every class equally and so
    #cannot change a ranking, which is all BAcc@k reads
    return (predicted @ targets.t()) * logit_scale
