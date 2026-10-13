"""P2 of the 13 Oct research note: a row-identification encoder in the style of CL-SSTER.

A row's 0.5 s timing-residualized window of the shared components should look
the same in every listener: what the stimulus drives, not who heard it. The
encoder learns exactly that, contrastively:

    positives   the same row in two different subjects of one dataset
    negatives   every other row of the batch EXCEPT rows of the same sentence
                (their windows overlap) and rows sharing any of the base
                mix's top-3 words with the anchor -- the note's "mild,
                controlled echo"
    loss        symmetric InfoNCE at temperature 0.1 (`info_nce`)

One space for every dataset: the holdout's MCCA fit defines it. Subject 0
heard both the holdout and the testbed, so its holdout projection reads its
testbed recording directly; each other testbed subject is mapped onto subject
0's components there by ridge regression, fitted on ses-11 alone
(`anchor_map`). So one encoder trains on the holdout and testbed ses-11
together, and ses-12, never seen, is its gate (cli/row_encoder.py).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import crossfeatures as cf
from .crosssubject import Whitener

#InfoNCE's temperature, CL-SSTER's and SimCLR's usual range
TEMPERATURE = 0.1

#the embedding's width
EMBED_DIM = 64

#the anchor regression's ridge, on unit-variance whitened inputs
ANCHOR_RIDGE = 1e-2

#per-subject PCA before the anchor regression, as before MCCA
ANCHOR_PCS = 60


def anchor_map(own_pieces, reference_components, n_pcs=ANCHOR_PCS, ridge=ANCHOR_RIDGE):
    """A subject's map into another fit's shared space, by regression onto subject 0's components.

    ARGS:
        own_pieces (list): the subject's low-passed, scaled [306, n_i] pieces
            of the fitting session
        reference_components (list): subject 0's [C, n_i] components of the
            same audio, from subject 0's own projection in that space
        n_pcs (int), ridge (float)

    RETURNS:
        whitener (Whitener), weights [n_pcs, C]: components are
        weights.T @ whitener(piece)
    """
    stacked = np.concatenate(own_pieces, axis=1)
    whitener = Whitener(stacked, n_pcs)
    inputs = whitener(stacked)
    targets = np.concatenate(reference_components, axis=1)

    #ridge regression: [P, P] Gram of the whitened inputs against [P, C] cross terms
    gram = inputs @ inputs.T / inputs.shape[1]
    cross = inputs @ targets.T / inputs.shape[1]
    return whitener, np.linalg.solve(gram + ridge * np.eye(len(gram)), cross)


class RowEncoder(nn.Module):
    """Each row's [C, 25] residual window -> one unit-length embedding."""

    def __init__(self, n_components, n_times=cf.WINDOW_LENGTH, spatial=32, hidden=256, embed_dim=EMBED_DIM, dropout=0.2):
        """
        ARGS:
            n_components (int): C, the shared components
            n_times (int): window samples
            spatial (int): components mixed to this many per time step
            hidden (int): the MLP's width
            embed_dim (int)
            dropout (float)
        """
        super().__init__()
        self.spatial = nn.Linear(n_components, spatial)
        self.head = nn.Sequential(nn.GELU(), nn.Flatten(start_dim=1), nn.Linear(spatial * n_times, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, embed_dim))

    def forward(self, windows):
        """
        ARGS:
            windows (torch.Tensor): [B, C, T]

        RETURNS:
            [B, embed_dim], unit length
        """
        #[B, C, T] -> [B, T, C]: the component mix is the same at every time step
        mixed = self.spatial(windows.transpose(1, 2))
        return F.normalize(self.head(mixed), dim=1)


def negative_mask(sentence, top_words):
    """Which pairs of batch rows may be each other's negatives.

    ARGS:
        sentence (torch.Tensor): [B] int64, unique across datasets
        top_words (torch.Tensor): [B, 3] int64, the base mix's top words

    RETURNS:
        [B, B] bool: other sentence and no top-3 word in common
    """
    other_sentence = sentence[:, None] != sentence[None, :]
    #[B, B, 3, 3] word-by-word equality, reduced over both word axes
    shared = (top_words[:, None, :, None] == top_words[None, :, None, :]).any(dim=3).any(dim=2)
    return other_sentence & ~shared


def info_nce(first, second, allowed, temperature=TEMPERATURE):
    """Symmetric InfoNCE over two views of B rows, negatives masked.

    ARGS:
        first, second (torch.Tensor): [B, D] unit embeddings of the same rows
            in two different subjects
        allowed (torch.Tensor): [B, B] bool from `negative_mask`
        temperature (float)

    RETURNS:
        scalar loss
    """
    batch = len(first)
    embeddings = torch.cat([first, second], dim=0)
    similarity = embeddings @ embeddings.T / temperature

    #each view's partner is the positive; a row never meets itself (same sentence)
    partner = torch.cat([torch.arange(batch, 2 * batch), torch.arange(0, batch)]).to(first.device)
    keep = allowed.repeat(2, 2)
    keep[torch.arange(2 * batch, device=first.device), partner] = True
    return F.cross_entropy(similarity.masked_fill(~keep, float("-inf")), partner)


def make_views(datasets, chosen, generator, device):
    """Two subjects' windows of every chosen row, with its sentence and top words.

    ARGS:
        datasets (list): dicts with windows [S, n, C, T] float32, sentence [n]
            (unique across datasets), top_words [n, 3]
        chosen (list): (dataset, row) pairs
        generator (np.random.Generator)
        device (torch.device)

    RETURNS:
        first, second [B, C, T]; sentence [B]; top_words [B, 3] -- tensors
    """
    first, second, sentence, top = [], [], [], []
    for dataset, row in chosen:
        data = datasets[dataset]
        #two different listeners of the same row
        one, other = generator.choice(len(data["windows"]), 2, replace=False)
        first.append(data["windows"][one, row])
        second.append(data["windows"][other, row])
        sentence.append(data["sentence"][row])
        top.append(data["top_words"][row])
    as_tensor = lambda values, dtype: torch.as_tensor(np.stack(values), dtype=dtype, device=device)
    return as_tensor(first, torch.float32), as_tensor(second, torch.float32), as_tensor(sentence, torch.int64), as_tensor(top, torch.int64)


def batch_loss(model, datasets, chosen, generator, device):
    """One batch's InfoNCE.

    ARGS:
        model (RowEncoder), datasets (list), chosen (list), generator, device

    RETURNS:
        scalar loss tensor
    """
    first, second, sentence, top = make_views(datasets, chosen, generator, device)
    return info_nce(model(first), model(second), negative_mask(sentence, top))


def train_encoder(datasets, n_components, epochs=300, patience=30, batch_size=256, learning_rate=1e-3,
                  weight_decay=1e-2, held_out_share=0.15, seed=0, device="cpu", log=print):
    """Train the encoder on every dataset's rows; stop on held-out sentences' loss.

    No label is read: rows are told apart by identity alone.

    ARGS:
        datasets (list): as `make_views`
        n_components (int)
        epochs, patience, batch_size (int); learning_rate, weight_decay,
            held_out_share (float); seed (int); device (str); log (callable)

    RETURNS:
        model (RowEncoder, eval mode, the best held-out epoch's weights),
        history (list of (epoch, train loss, held-out loss))
    """
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device(device)

    #each dataset's sentences split once: a share held out for stopping
    train_rows, held_rows = [], []
    for index, data in enumerate(datasets):
        sentences = np.unique(data["sentence"])
        held = set(generator.choice(sentences, max(int(round(held_out_share * len(sentences))), 1), replace=False).tolist())
        for row, sentence in enumerate(data["sentence"]):
            (held_rows if int(sentence) in held else train_rows).append((index, row))

    model = RowEncoder(n_components).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_loss, best_state, stale, history = np.inf, None, 0, []
    for epoch in range(epochs):
        #every training row once as an anchor, in a fresh order
        model.train()
        order = generator.permutation(len(train_rows))
        losses = []
        for start in range(0, len(order), batch_size):
            loss = batch_loss(model, datasets, [train_rows[i] for i in order[start : start + batch_size]], generator, device)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            losses.append(float(loss))

        #held-out sentences with the same views every epoch, so losses compare
        model.eval()
        held_generator = np.random.default_rng(seed + 1)
        with torch.no_grad():
            held_losses = [float(batch_loss(model, datasets, held_rows[start : start + batch_size], held_generator, device))
                           for start in range(0, len(held_rows), batch_size)]
        held_loss = float(np.mean(held_losses))
        history.append((epoch, float(np.mean(losses)), held_loss))
        if held_loss < best_loss - 1e-4:
            best_loss, best_state, stale = held_loss, {name: value.detach().clone() for name, value in model.state_dict().items()}, 0
        else:
            stale += 1
        if epoch % 10 == 0 or stale >= patience:
            log(f"[row_encoder] epoch {epoch}: train {history[-1][1]:.3f}, held-out {held_loss:.3f} (best {best_loss:.3f})")
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), history


def embed(model, windows, batch_size=1024):
    """Every subject's rows through the encoder.

    ARGS:
        model (RowEncoder), windows (np.ndarray): [S, n, C, T]
        batch_size (int)

    RETURNS:
        [S, n, D] float32
    """
    device = next(model.parameters()).device
    flat = windows.reshape(-1, *windows.shape[2:])
    pieces = []
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            pieces.append(model(torch.as_tensor(flat[start : start + batch_size], dtype=torch.float32, device=device)).cpu().numpy())
    return np.concatenate(pieces).reshape(windows.shape[0], windows.shape[1], -1)
