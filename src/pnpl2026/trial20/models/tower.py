"""The word tower: a trainable word side for a two-tower word decoder.

Every earlier trial aimed the brain side at one FIXED vector per word (t5's
middle layer), or at several fixed tables through extra heads. Here several
fixed descriptions of a word -- t5 layers, its sound shape, its letters, its
usual length -- go through a small MLP into a shared space, and the brain
side is mapped into the same space (`ContinuousNet.project_tower`). D-SigLIP
pulls matching pairs together as before; both sides learn.

Two reasons (the 10 Oct note): models with a second target space were the
best additions to the mix (Kn +1.7, K2 +1.5), and a tower draws on every
space at once; and hub words -- those nearest the middle of a fixed space --
can be moved apart only if the word side can move.

The descriptions table is `data/word_descriptions.npz`
(`cli.build_word_descriptions`): rows are the t5 table's words in its order,
columns are blocks laid end to end; `block_widths` says where each ends.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordTower(nn.Module):
    """Fixed descriptions of a word -> one unit vector in the shared space.

    ATTRS:
        block_widths (list): widths of the description blocks, in order
        norms (nn.ModuleList): one LayerNorm per block
        hidden (nn.Linear): total -> hidden
        out (nn.Linear): hidden -> out_dim
    """

    def __init__(self, block_widths, hidden=512, out_dim=256):
        """
        ARGS:
            block_widths (sequence): e.g. [1024, 1024, 1024, 1024, 1024, 1024, 1]
            hidden (int)
            out_dim (int): the shared space's width
        """
        super().__init__()
        self.block_widths = [int(width) for width in block_widths]
        if not self.block_widths or min(self.block_widths) < 1:
            raise ValueError(f"block_widths must be positive, got {block_widths}")

        #each block on its own scale before they meet: a 1-wide block is kept as is
        self.norms = nn.ModuleList(
            [nn.LayerNorm(width) if width > 1 else nn.Identity() for width in self.block_widths]
        )
        self.hidden = nn.Linear(sum(self.block_widths), hidden)
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, descriptions):
        """
        ARGS:
            descriptions (torch.Tensor): [n_words, sum(block_widths)]

        RETURNS:
            [n_words, out_dim], unit length
        """
        #the blocks apart, each normalised, back together; dim=1 is the feature axis
        parts = torch.split(descriptions.float(), self.block_widths, dim=1)
        normalised = torch.cat([norm(part) for norm, part in zip(self.norms, parts)], dim=1)
        return F.normalize(self.out(F.gelu(self.hidden(normalised))), dim=1)


def load_descriptions(path, words):
    """The description table, checked against the t5 table's word list.

    ARGS:
        path (Path or str): `data/word_descriptions.npz`
        words (list): the t5 table's normalised words, in order

    RETURNS:
        dict: "table" [n_words, total] float32 tensor, "blocks" list of str,
        "widths" list of int
    """
    import numpy as np

    from ..data.words import normalise_word

    with np.load(path, allow_pickle=False) as stored:
        these = [normalise_word(word) for word in stored["words"]]
        if these != list(words):
            raise ValueError(f"{path} lists other words, or another order, than the t5 table; rebuild it.")
        table = torch.from_numpy(np.asarray(stored["descriptions"], dtype=np.float32))
        blocks = [str(name) for name in stored["blocks"]]
        widths = [int(width) for width in stored["widths"]]
    if table.shape[1] != sum(widths):
        raise ValueError(f"{path}: {table.shape[1]} columns but block widths sum to {sum(widths)}.")
    return {"table": table, "blocks": blocks, "widths": widths}
