"""Build the all-word t5 table: one target embedding for EVERY word, once.

    PYTHONPATH=$HOME/t5libs:src python -m pnpl2026.trial10.cli.build_word_embeddings \\
        --config configs/base.yaml configs/trial10/trial10.yaml

Reads every word the reference cache's word tables hold (all partitions), adds
both scored vocabularies, and writes `data/word_embeddings_t5_all.npz` with
`words` and `embeddings`. Following d'Ascoli et al.: **t5-large, the MIDDLE
layer, each word encoded on its own**, sub-tokens averaged.

Test-partition words are encoded too. That is not leakage: a word's target is
a fixed function of its spelling, never of any MEG, and the reference scores
retrieval among the 50 vocabulary words only.

⚠️ Needs the model files. Download them on a login node (outbound internet),
then encode in a job with HF_HUB_OFFLINE=1 -- see pbs/trial10_embed.pbs.

⚠️ `transformers` and `sentencepiece` are NOT in the project environment on
purpose (trial 7's recipe): install them aside and put them on PYTHONPATH.

    uv pip install --python .venv/bin/python --target "$HOME/t5libs" \\
        transformers sentencepiece
"""

from __future__ import annotations

import argparse

import numpy as np

from ..config import load_trial10_config
from ..data import Vocabulary, load_manifest

#what the reference used; t5-large's hidden width is 1024
DEFAULT_MODEL = "t5-large"

#words per forward pass; each is a handful of tokens
BATCH_WORDS = 256


def parse_args():
    """Command-line arguments for the embedding build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", nargs="+", required=True, help="for the cache and the output path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace id (default {DEFAULT_MODEL})")
    parser.add_argument("--layer", type=int, default=None, help="hidden state to read; default the middle")
    parser.add_argument("--device", default=None, help="cuda or cpu; default cuda when available")
    return parser.parse_args()


def collect_words(cache_dir):
    """Every normalised word in the cache, plus both scored vocabularies.

    ARGS:
        cache_dir (Path): the reference cache

    RETURNS:
        sorted list of str, unique
    """
    manifest = load_manifest(cache_dir)
    words = set()
    for record in manifest["recordings"]:
        with np.load(cache_dir / f"words_{record['key']}.npz") as table:
            words.update(str(word) for word in table["word"])

    #⚠️ both vocabularies whole, so every candidate has a row even if a word
    #were somehow absent from every recording
    for name in ("primary", "moses"):
        words.update(Vocabulary(name).words)

    #the builder already dropped empty strings; this is the belt to its braces
    words.discard("")
    return sorted(words)


def encode_words(model, tokeniser, words, layer, device):
    """Encode words alone, in batches, one mean vector per word.

    ARGS:
        model: a t5 encoder
        tokeniser: its tokeniser
        words (list): of str
        layer (int): which hidden state to read
        device (torch.device)

    RETURNS:
        [len(words), hidden] float32 numpy array
    """
    import torch

    rows = []
    for start in range(0, len(words), BATCH_WORDS):
        chunk = words[start : start + BATCH_WORDS]

        #add_special_tokens=False: t5's end token would be averaged into every
        #word and pull them all towards one point
        encoded = tokeniser(chunk, return_tensors="pt", add_special_tokens=False, padding=True)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded, output_hidden_states=True).hidden_states[layer]

        #mean over each word's REAL tokens: dim=1 is the token axis
        mask = encoded["attention_mask"].unsqueeze(2).to(hidden.dtype)
        rows.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1)).float().cpu().numpy())
        print(f"[embed] {min(start + BATCH_WORDS, len(words)):,} / {len(words):,}", end="\r")
    print()
    return np.concatenate(rows, axis=0).astype(np.float32)


def main():
    """Encode every word and write the table."""
    args = parse_args()
    cfg = load_trial10_config(*args.config)

    #imported here so the rest of the package never needs transformers
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "transformers is not importable. See this module's docstring for the "
            "isolated install that leaves the project venv alone."
        ) from error

    words = collect_words(cfg.resolved_cache_dir())
    print(f"[embed] {len(words):,} distinct words from {cfg.resolved_cache_dir()}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[embed] loading {args.model} on {device}")
    tokeniser = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).encoder.to(device).eval()

    #hidden_states holds the embedding output too: n_layers + 1 entries
    n_hidden = model.config.num_layers + 1
    layer = args.layer if args.layer is not None else n_hidden // 2
    print(f"[embed] reading hidden state {layer} of {n_hidden}")

    table = encode_words(model, tokeniser, words, layer, device)

    #a near-constant table would make every word the same target
    spread = float(np.std(table))
    if spread < 1e-6 or not np.isfinite(table).all():
        raise SystemExit(f"Embeddings are degenerate (std {spread:.2e}); layer {layer} is unusable.")

    out_path = cfg.resolved_word_embeddings()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, words=np.array(words), embeddings=table, model=args.model, layer=layer)

    #how separable the 50 scored words are in this space
    primary = Vocabulary("primary").words
    index = {word: row for row, word in enumerate(words)}
    unit = table[[index[word] for word in primary]]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    off_diagonal = (unit @ unit.T)[~np.eye(len(primary), dtype=bool)]
    print(
        f"[embed] wrote {out_path}: {table.shape[0]:,} x {table.shape[1]}; "
        f"primary words' mean off-diagonal cosine {off_diagonal.mean():.3f}, max {off_diagonal.max():.3f}"
    )


if __name__ == "__main__":
    main()
