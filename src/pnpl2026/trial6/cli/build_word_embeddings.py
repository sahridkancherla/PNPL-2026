"""Build the fixed table of target word embeddings, once.

    PYTHONPATH=src python -m pnpl2026.trial6.cli.build_word_embeddings

Writes `data/word_embeddings_t5.npz` with `words` and `embeddings`, rows in the
competition's vocabulary order. The semantic head regresses onto these.

⚠️ Needs outbound internet the first time, to download the model. Run it on a
login node or a laptop, never on an HPC compute node.

⚠️ `transformers` and `sentencepiece` are NOT in this project's environment on
purpose -- adding them would resolve the whole dependency tree and could move
`torch` under a training run that is already going. Point PYTHONPATH at a
separate install instead:

    uv pip install --python .venv/bin/python --target /tmp/t5libs \\
        transformers sentencepiece
    PYTHONPATH=/tmp/t5libs:src .venv/bin/python -m \\
        pnpl2026.trial6.cli.build_word_embeddings

Following d'Ascoli et al.: **t5-large, the MIDDLE layer, each word encoded on
its own** so the embedding is non-contextualised. Multi-token words are averaged
over their tokens.
"""

from __future__ import annotations

import argparse

import numpy as np

from ...config import DATA_DIR
from ...data import vocab

#what the reference model used. t5-large's hidden width is 1024
DEFAULT_MODEL = "t5-large"

#the output file the config's `word_embeddings` field points at by default
DEFAULT_OUTPUT = DATA_DIR / "word_embeddings_t5.npz"


def parse_args():
    """Command-line arguments for the embedding build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"HuggingFace model id (default {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="where to write the .npz",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="hidden layer to read. Default is the middle one",
    )
    return parser.parse_args()


def word_embedding(model, tokeniser, word, layer):
    """Encode one word on its own and return one vector for it.

    ARGS:
        model: a loaded transformers encoder
        tokeniser: its tokeniser
        word (str): the word, encoded with no surrounding context
        layer (int): which hidden state to read

    RETURNS:
        embedding : [hidden_size] float32 numpy array
    """
    import torch

    #add_special_tokens=False: t5's sentinel would otherwise get averaged in
    #and every word would drift towards the same point
    encoded = tokeniser(word, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)

    #[1, n_tokens, hidden]
    hidden = out.hidden_states[layer]

    #dim=1 is the TOKEN axis: a word split into three pieces becomes their mean,
    #which is what the reference model does
    return hidden.mean(dim=1).squeeze(0).float().numpy()


def main():
    """Encode all 50 scored words and write the table."""
    args = parse_args()

    #imported here, not at module scope, so the rest of the package does not
    #need transformers installed to be importable
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "transformers is not importable. See this module's docstring for "
            "the isolated-install recipe that does not touch the project venv."
        ) from error

    print(f"[embed] loading {args.model} (first run downloads it)")
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).encoder
    model.eval()

    #the middle layer, per the reference model. hidden_states includes the
    #embedding output at index 0, so there are n_layers + 1 of them
    n_hidden = model.config.num_layers + 1
    layer = args.layer if args.layer is not None else n_hidden // 2
    print(f"[embed] reading hidden layer {layer} of {n_hidden}")

    words = list(vocab.PRIMARY)
    table = np.stack(
        [word_embedding(model, tokeniser, word, layer) for word in words]
    ).astype(np.float32)

    #a constant table would make every class identical and the loss unlearnable
    spread = float(np.std(table))
    if spread < 1e-6:
        raise SystemExit(
            f"Embeddings are almost constant (std {spread:.2e}) -- layer "
            f"{layer} of this model carries nothing word-specific."
        )

    #⚠️ written in vocab.PRIMARY order, which embeddings.load_target_embeddings
    #re-checks on every load. Row i must be PRIMARY[i]
    np.savez(args.out, words=np.array(words), embeddings=table)

    #the mean pairwise cosine says whether the space is actually discriminative
    unit = table / np.linalg.norm(table, axis=1, keepdims=True)
    similarity = unit @ unit.T
    off_diagonal = similarity[~np.eye(len(words), dtype=bool)]
    print(
        f"[embed] wrote {args.out}: {table.shape[0]} x {table.shape[1]}, "
        f"std {spread:.3f}, mean off-diagonal cosine {off_diagonal.mean():.3f}"
    )
    print(f"[embed] closest pair cosine {off_diagonal.max():.3f}")


if __name__ == "__main__":
    main()
