"""The trial-10 network: d'Ascoli et al.'s word decoder, for one subject.

    tokens, flat       [N, 306, 150] + lengths, sequence, position
      -> WordEncoder                        [N, d_model]
      -> into a [sequences, words] grid     [S, W, d_model] + padding mask
      -> sentence transformer               16 x 1024 x 16 heads, rotary,
                                            every word reads the sentence
      -> back to tokens                     [N, d_model]
      -> to the t5 width, if not already    [N, 1024]

No closed-set head: the output is one vector per word, trained with D-SigLIP
against that word's t5-large embedding and scored by retrieval among the 50
candidates' embeddings.

F1 shrinks the sentence transformer to trial 8's 4 x 256; a linear map then
takes its output to the t5 width. At d_model 1024 there is no such map.

Trial 16's Variant K (kept, off by default): `second_head` adds a second linear map from the same
word vector, `embedding_second`, trained against a second target table (an
early t5 layer). It is always a Linear, even at d_model 1024, so the two
heads never share weights. Off by default; earlier checkpoints load without it.

Trial 18's Variant M (off by default): `refine` adds `PyramidRefiner`, HDND's
bounded correction of `embedding` from two conv layers' pooled outputs
(`refine.py`). With it, "embedding" is the refined prediction and the base's
own is "embedding_base"; trained on a frozen checkpoint (`lit.py`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import WordEncoder
from .refine import PyramidRefiner
from .transformer import SentenceTransformer

#t5-large's hidden width, the target space
T5_WIDTH = 1024


class ReferenceNet(nn.Module):
    """Per-word convolutional encoder, then a large transformer across words.

    ATTRS:
        encoder (WordEncoder)
        sentence (SentenceTransformer or None): None at sentence_depth 0
        project (nn.Module): d_model -> embed_dim, or Identity
        project_second (nn.Linear or None): Variant K's second head
        refiner (PyramidRefiner or None): Variant M's correction
        tap_layers (tuple): the conv layers the refiner reads
    """

    def __init__(
        self,
        n_channels,
        embed_dim=T5_WIDTH,
        sample_rate=50.0,
        clamp=5.0,
        baseline_seconds=0.5,
        d_model=1024,
        sentence_depth=16,
        n_heads=16,
        ffn_expansion=4,
        attention_dropout=0.1,
        dropout=0.0,
        drop_path=0.0,
        second_head=False,
        refine=None,
        n_subjects=1,
        **encoder_kwargs,
    ):
        """
        ARGS:
            n_channels (int): 306
            embed_dim (int): the target width, 1024
            sample_rate (float): of the cache, 50 Hz
            clamp (float): from the run config
            baseline_seconds (float): from the run config
            d_model (int): the encoder's output and the transformer's width
            sentence_depth (int): 16; 0 sends the encoder's vectors straight out
            n_heads (int): 16
            ffn_expansion (int): 4
            attention_dropout (float): 0.1
            dropout (float): feed-forward dropout, 0
            drop_path (float): 0
            second_head (bool): add Variant K's second d_model -> embed_dim map
            refine (dict or None): Variant M's refiner -- layers, bins,
                seconds, width, bound, query_skip, centre_momentum
            n_subjects (int): subject layers; above 1 for all-data training
            **encoder_kwargs: to WordEncoder -- spatial, positions, pca, hidden...
        """
        super().__init__()
        self.encoder = WordEncoder(
            n_channels,
            d_model,
            sample_rate,
            clamp=clamp,
            baseline_seconds=baseline_seconds,
            n_subjects=n_subjects,
            **encoder_kwargs,
        )

        self.sentence = None
        if sentence_depth > 0:
            self.sentence = SentenceTransformer(
                d_model=d_model,
                depth=sentence_depth,
                n_heads=n_heads,
                ffn_expansion=ffn_expansion,
                attention_dropout=attention_dropout,
                dropout=dropout,
                drop_path=drop_path,
            )

        #only a narrower transformer needs a map to the t5 width
        self.project = nn.Identity() if d_model == embed_dim else nn.Linear(d_model, embed_dim)

        #Variant K: the same word vector, mapped into the second target space
        self.project_second = nn.Linear(d_model, embed_dim) if second_head else None

        #Variant M: a bounded correction from chosen conv layers' outputs
        self.refiner = None
        self.tap_layers = ()
        if refine:
            self.tap_layers = tuple(sorted(int(layer) for layer in refine["layers"]))
            convs = self.encoder.convs

            #HDND reads GATED outputs, so every tap must sit on a gated conv
            for layer in self.tap_layers:
                if not 0 <= layer < len(convs.layers) or convs.glus[layer] is None:
                    gated = [index for index, glu in enumerate(convs.glus) if glu is not None]
                    raise ValueError(f"refine layer {layer} has no gated conv; gated layers are {gated}")

            #each tapped layer's width is its conv's output channels
            widths = {convs.layers[layer][0].out_channels for layer in self.tap_layers}
            if len(widths) != 1:
                raise ValueError(f"refine layers {self.tap_layers} differ in width: {sorted(widths)}")
            self.refiner = PyramidRefiner(
                hidden=widths.pop(),
                n_depths=len(self.tap_layers),
                embed_dim=embed_dim,
                bins=tuple(refine["bins"]),
                n_samples=int(round(float(refine["seconds"]) * sample_rate)),
                width=int(refine["width"]),
                bound=float(refine["bound"]),
                query_skip=bool(refine["query_skip"]),
                centre_momentum=float(refine["centre_momentum"]),
            )

    def forward(self, windows, lengths, sequence, position, subject=None):
        """
        ARGS:
            windows (torch.Tensor): [N, n_channels, n_times], zero-padded
            lengths (torch.Tensor): [N] real samples per window
            sequence (torch.Tensor): [N] which sequence each word belongs to
            position (torch.Tensor): [N] its place in that sequence
            subject (torch.Tensor or None): [N] subject layers; None is subject 0

        RETURNS:
            dict:
                "words" : [N, d_model] - the encoder's vectors
                "embedding" : [N, embed_dim] - unnormalised, compared by cosine
                "embedding_second" : [N, embed_dim] - the second head's,
                    present only with second_head
                with the refiner, "embedding" is the refined prediction and
                "embedding_base" [N, embed_dim] the base's at unit length;
                "delta" [N, embed_dim], "centre" [1, embed_dim] and
                "pyramids" [N, n_depths, n_features] feed the losses
        """
        #⚠️ every per-token array must describe the same tokens, in order
        n_words = windows.shape[0]
        assert sequence.shape == position.shape == (n_words,), (
            f"sequence {tuple(sequence.shape)} and position {tuple(position.shape)} for {n_words} windows"
        )

        words, tapped = self.encoder.features(windows, lengths, self.tap_layers, subject)
        context = words
        if self.sentence is not None:
            grid, valid = self._grid(words, sequence, position)
            context = self.sentence(grid, valid)[sequence, position]
        out = {"words": words, "embedding": self.project(context)}

        #both heads read the same context vector; only their maps differ
        if self.project_second is not None:
            out["embedding_second"] = self.project_second(context)

        #Variant M: the word's own pooled MEG corrects the prediction, in float32
        if self.refiner is not None:
            pyramids = self.refiner.pool(tapped, lengths)
            with torch.autocast(device_type=windows.device.type, enabled=False):
                refined = self.refiner(out["embedding"].float(), pyramids)
            out["embedding"] = refined["refined"]
            out["embedding_base"] = refined["base"]
            out["delta"] = refined["delta"]
            out["centre"] = refined["centre"]
            out["pyramids"] = pyramids
        return out

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
