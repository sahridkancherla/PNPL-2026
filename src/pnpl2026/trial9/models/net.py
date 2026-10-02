"""The trial-9 network: a per-word encoder, then a transformer across words.

    tokens, flat       [N, 73, 750] + lengths, sequence, position
      -> PerWordEncoder                     [N, 256]
      -> into a [sequences, words] grid     [S, W, 256] + padding mask
      -> word-token masking                 training only, p = 0.15
      -> sentence transformer               depth 4, rotary over word index,
                                            one attention reach per block
      -> back to tokens                     [N, 256]
      -> ClosedSetHead, SemanticHead        [N, 50], [N, 1024]

Every word of a sentence sees the MEG of the others through the sentence
transformer -- the gain Variant D is built for. With `sentence_depth: 0` the
encoder's vectors go straight to the heads: D0 and D2.

Trial 8 added `sentence_reach` (which words each block may read); trial 9
leaves it off. Trial 9's one change is a high-pass inside the encoder -- see
`encoder.py` and `frontend.WindowHighpass`.

⚠️ Two masking modes, and they answer different questions:

    word-token masking   replaces a random 15% of words with a learned token
                         in training. Those words are NOT scored -- the point
                         is that every OTHER word learns not to depend on any
                         one neighbour, as when the holdout's context is short
    mask_target (D8)     every word is predicted with its OWN vector replaced,
                         one copy of its sentence per word. What remains is
                         context -- but the neighbours' windows overlap this
                         word's response, so it is not brain-free
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import PerWordEncoder
from .heads import ClosedSetHead, SemanticHead
from .transformer import FlatTransformer

#t5-large's hidden width, which the semantic head regresses onto
DEFAULT_EMBED_DIM = 1024


class TrialNineNet(nn.Module):
    """Per-word encoder, sentence transformer, two per-word heads.

    ATTRS:
        encoder (PerWordEncoder)
        sentence (FlatTransformer or None): None at sentence_depth 0
        mask_token (nn.Parameter or None): what a masked word is replaced by
        closed_set (ClosedSetHead)
        semantic (SemanticHead)
        word_mask_p (float): training-time masking probability
        causal (bool): D3's causal sentence transformer
    """

    def __init__(
        self,
        n_channels,
        n_classes,
        embed_dim=DEFAULT_EMBED_DIM,
        sentence_depth=4,
        causal=False,
        sentence_reach=None,
        head_dropout=0.1,
        word_mask_p=0.0,
        d_model=256,
        n_heads=8,
        ffn_expansion=4,
        dropout=0.1,
        drop_path=0.1,
        transformer_norm="layer",
        **encoder_kwargs,
    ):
        """Assemble the network.

        ARGS:
            n_channels (int): 73 whitened components
            n_classes (int): 50, the scored vocabulary
            embed_dim (int): target word-embedding width
            sentence_depth (int): blocks across words. 0 is D0 and D2
            causal (bool): each word sees only itself and earlier words (D3)
            sentence_reach (list or None): one entry per sentence block, a
                positive int or None for the whole sentence. None throughout
                is trial 7's full attention
            head_dropout (float): on the word vector, in both heads
            word_mask_p (float): chance a word is masked in training
            d_model (int): word vector width, shared by both transformers
            n_heads (int): attention heads, both transformers
            ffn_expansion (int): FFN hidden multiple, both transformers
            dropout (float): attention and FFN dropout, both transformers
            drop_path (float): maximum stochastic depth, both transformers
            transformer_norm (str): "layer" or "bias"
            **encoder_kwargs: the rest go to PerWordEncoder
        """
        super().__init__()

        #a masked word with no sentence transformer to read around it would
        #reach its head as the bare mask token
        if sentence_depth <= 0 and word_mask_p > 0:
            raise ValueError("word_mask_p needs a sentence transformer (sentence_depth > 0).")

        shared = dict(
            d_model=d_model,
            n_heads=n_heads,
            ffn_expansion=ffn_expansion,
            dropout=dropout,
            drop_path=drop_path,
            transformer_norm=transformer_norm,
        )
        self.encoder = PerWordEncoder(n_channels, **shared, **encoder_kwargs)

        self.sentence = None
        self.mask_token = None
        if sentence_depth > 0:
            self.sentence = FlatTransformer(
                d_model=d_model,
                depth=sentence_depth,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                dropout=dropout,
                drop_path=drop_path,
                norm=transformer_norm,
                reach=sentence_reach,
            )

            #small and random, like a learned embedding, so a masked word is
            #distinguishable from any real one from the first step
            self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)

        self.closed_set = ClosedSetHead(d_model, n_classes, head_dropout)
        self.semantic = SemanticHead(d_model, embed_dim, head_dropout)
        self.word_mask_p = word_mask_p
        self.causal = causal

    def forward(self, windows, lengths, sequence, position, mask_target=False):
        """
        ARGS:
            windows (torch.Tensor): [N, n_channels, n_times], zero-padded
            lengths (torch.Tensor): [N] real samples per window
            sequence (torch.Tensor): [N] which sequence each word belongs to
            position (torch.Tensor): [N] its place in that sequence
            mask_target (bool): D8 -- predict every word with itself masked

        RETURNS:
            dict:
                "words" : [N, d_model] - the encoder's vectors
                "context" : [N, d_model] - what both heads read
                "logits" : [N, n_classes]
                "embedding" : [N, embed_dim] - unnormalised
                "masked" : [N] bool - whose own vector was hidden
        """
        words = self.encoder(windows, lengths)
        masked = torch.zeros(len(words), dtype=torch.bool, device=words.device)

        if self.sentence is None:
            context = words
        elif mask_target:
            context = self._each_masked(words, sequence, position)
            masked = torch.ones_like(masked)
        else:
            #never the only word of a sequence: it would have nothing left
            if self.training and self.word_mask_p > 0:
                sizes = torch.bincount(sequence)
                eligible = sizes[sequence] > 1
                masked = (torch.rand(len(words), device=words.device) < self.word_mask_p) & eligible
            context = self._contextualise(words, sequence, position, masked)

        return {
            "words": words,
            "context": context,
            "logits": self.closed_set(context),
            "embedding": self.semantic(context),
            "masked": masked,
        }

    def _grid(self, tokens, sequence, position):
        """Scatter flat tokens into a padded [sequences, words] grid.

        ARGS:
            tokens (torch.Tensor): [N, d_model]
            sequence (torch.Tensor): [N]
            position (torch.Tensor): [N]

        RETURNS:
            grid : [S, W, d_model], zero at padding
            valid : [S, W] bool, True at real words
        """
        n_sequences = int(sequence.max()) + 1
        width = int(position.max()) + 1

        #advanced-index assignment keeps the autograd link to every token
        grid = tokens.new_zeros(n_sequences, width, tokens.shape[1])
        grid[sequence, position] = tokens
        valid = torch.zeros(n_sequences, width, dtype=torch.bool, device=tokens.device)
        valid[sequence, position] = True
        return grid, valid

    def _contextualise(self, words, sequence, position, masked):
        """Run the sentence transformer once, over every sequence.

        ARGS:
            words (torch.Tensor): [N, d_model]
            sequence (torch.Tensor): [N]
            position (torch.Tensor): [N]
            masked (torch.Tensor): [N] bool, words replaced by the mask token

        RETURNS:
            context : [N, d_model], in the same order as `words`
        """
        #unsqueeze(1) broadcasts the mask token across the masked rows only
        tokens = torch.where(masked.unsqueeze(1), self.mask_token.expand_as(words), words)
        grid, valid = self._grid(tokens, sequence, position)
        out = self.sentence(grid, valid=valid, causal=self.causal)
        return out[sequence, position]

    def _each_masked(self, words, sequence, position):
        """D8: every word predicted from a copy of its sentence without it.

        One copy per word, so N short sequences instead of S; the sentence
        transformer is small, and sequences hold at most 9 words.

        ARGS:
            words (torch.Tensor): [N, d_model]
            sequence (torch.Tensor): [N]
            position (torch.Tensor): [N]

        RETURNS:
            context : [N, d_model]
        """
        grid, valid = self._grid(words, sequence, position)

        #[N, W, d]: each word's own sentence, then that word replaced
        copies = grid[sequence].clone()
        rows = torch.arange(len(words), device=words.device)
        copies[rows, position] = self.mask_token

        out = self.sentence(copies, valid=valid[sequence], causal=self.causal)
        return out[rows, position]
