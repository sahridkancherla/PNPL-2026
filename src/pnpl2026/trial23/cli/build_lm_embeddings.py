"""Build a target table from any Hugging Face language model's middle layer.

    PYTHONPATH=$HOME/t5libs:src python -m pnpl2026.trial23.cli.build_lm_embeddings \\
        --model openai-community/gpt2-medium --out data/word_embeddings_gpt2medium_layer12.npz
    PYTHONPATH=$HOME/t5libs:src python -m pnpl2026.trial23.cli.build_lm_embeddings \\
        --model Qwen/Qwen2.5-1.5B --out data/word_embeddings_qwen15_layer14.npz

Trial 10's t5 builder, for other language models: Recipe 2 of the final-night
note, a second head against another model's space (`cli.anneal --mode head`).
Every word of the first head's table, in its order -- the two heads index
both tables with one target row -- each encoded alone:

    model        the encoder of an encoder-decoder (t5), else the model itself,
                 in float32
    layer        a hidden state, by default the middle one: 12 of 25 for
                 GPT-2-medium, 14 of 29 for Qwen2.5-1.5B, 12 of 25 for t5-large
    tokens       no special tokens; a leading space when the tokeniser reads
                 one (byte-level BPE: GPT-2, Qwen), so a word is tokenised as
                 it appears inside a sentence; padding on the right, so a
                 word's positions never depend on its batch
    vector       the mean over the word's real tokens

Writes {words, embeddings, model, layer} as TargetTable reads them, with
trial 10's degenerate-table check and the primary words' off-diagonal cosine.
Trial 16's `hub_check` then says whether the table's hub words differ from
t5's (`pbs/trial23_lm_tables.pbs` runs both).

WARNING: transformers lives aside from the project venv ($HOME/t5libs on
CX3; see trial 10's builder), and the model's files must already be in the
Hugging Face cache: a compute node has no internet.
"""

from __future__ import annotations

import argparse

import numpy as np

from ...config import REPO_ROOT
from ..perword.data import Vocabulary, normalise_word

#words per forward pass; each is a handful of tokens
BATCH_WORDS = 128


def parse_args():
    """Command-line arguments for the table build."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="a Hugging Face id, e.g. openai-community/gpt2-medium")
    parser.add_argument("--out", required=True, help="the table, repo-relative")
    parser.add_argument("--words-from", default="data/word_embeddings_t5_all.npz",
                        help="the first head's table, repo-relative: its words, in its order")
    parser.add_argument("--layer", type=int, default=None, help="hidden state to read; default the middle one")
    parser.add_argument("--leading-space", choices=("auto", "yes", "no"), default="auto",
                        help="auto: when the tokeniser tells ' word' from 'word'")
    parser.add_argument("--device", default=None, help="cuda or cpu; default cuda when available")
    parser.add_argument("--batch-words", type=int, default=BATCH_WORDS)
    return parser.parse_args()


def read_words(path):
    """The words of an existing table, in its row order.

    ARGS:
        path (Path)

    RETURNS:
        list of str
    """
    with np.load(path, allow_pickle=False) as stored:
        return [str(word) for word in stored["words"]]


def reads_leading_space(tokeniser):
    """Whether the tokeniser gives a word another token after a space.

    ARGS:
        tokeniser: a Hugging Face tokeniser

    RETURNS:
        bool: True for byte-level BPE (GPT-2, Qwen); False for sentencepiece t5
    """
    alone = tokeniser("the", add_special_tokens=False)["input_ids"]
    spaced = tokeniser(" the", add_special_tokens=False)["input_ids"]
    return alone != spaced


def load_model(name, device):
    """The network whose hidden states are read, in float32 and eval mode.

    ARGS:
        name (str): a Hugging Face id
        device (torch.device)

    RETURNS:
        model, tokeniser
    """
    from transformers import AutoModel, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(name)
    #float32 whatever the checkpoint's own type (Qwen ships bf16)
    model = AutoModel.from_pretrained(name).float()

    #GPT-2 has no pad token: its end token pads (the tokeniser's, else the
    #model's), and the attention mask hides it either way
    if tokeniser.pad_token is None:
        end = tokeniser.eos_token
        if end is None:
            end = tokeniser.convert_ids_to_tokens(model.config.eos_token_id or 0)
        tokeniser.pad_token = end
    #right padding: a causal model's real tokens then sit at positions 0..n-1
    tokeniser.padding_side = "right"

    if getattr(model.config, "is_encoder_decoder", False):
        model = model.get_encoder()
    return model.to(device).eval(), tokeniser


def encode_words(model, tokeniser, words, layer, device, leading_space, batch_words=BATCH_WORDS):
    """Encode words alone, in batches, one mean vector per word.

    ARGS:
        model: from `load_model`
        tokeniser: its tokeniser
        words (list): of str
        layer (int): which hidden state to read
        device (torch.device)
        leading_space (bool): prefix each word with a space
        batch_words (int)

    RETURNS:
        [len(words), hidden] float32 numpy array
    """
    import torch

    rows = []
    for start in range(0, len(words), batch_words):
        chunk = words[start : start + batch_words]
        texts = [" " + word if leading_space else word for word in chunk]

        #add_special_tokens=False: a shared start or end token would pull every word to one point
        encoded = tokeniser(texts, return_tensors="pt", add_special_tokens=False, padding=True)
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.no_grad():
            hidden = model(**encoded, output_hidden_states=True, use_cache=False).hidden_states[layer]

        #mean over each word's REAL tokens: dim=1 is the token axis
        mask = encoded["attention_mask"].unsqueeze(2).to(hidden.dtype)
        rows.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1)).float().cpu().numpy())
        print(f"[lm-table] {min(start + batch_words, len(words)):,} / {len(words):,}", end="\r")
    print()
    return np.concatenate(rows, axis=0).astype(np.float32)


def count_hidden_states(model, tokeniser, device):
    """How many hidden states the model returns: its layers plus the embeddings.

    ARGS:
        model, tokeniser: from `load_model`
        device (torch.device)

    RETURNS:
        int
    """
    import torch

    encoded = tokeniser(["word"], return_tensors="pt", add_special_tokens=False)
    with torch.no_grad():
        out = model(**{name: value.to(device) for name, value in encoded.items()}, output_hidden_states=True,
                    use_cache=False)
    return len(out.hidden_states)


def main():
    """Encode every word of the first head's table and write the new table."""
    args = parse_args()

    #imported here so the rest of the package never needs transformers
    try:
        import torch
        import transformers
    except ImportError as error:
        raise SystemExit(
            "transformers is not importable. Put trial 10's isolated install on PYTHONPATH ($HOME/t5libs on CX3)."
        ) from error

    words = read_words(REPO_ROOT / args.words_from)
    print(f"[lm-table] {len(words):,} words from {args.words_from}, kept in its order")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[lm-table] loading {args.model} on {device} (transformers {transformers.__version__})")
    model, tokeniser = load_model(args.model, device)

    #the middle hidden state unless asked otherwise
    n_hidden = count_hidden_states(model, tokeniser, device)
    layer = args.layer if args.layer is not None else n_hidden // 2
    if not 0 <= layer < n_hidden:
        raise SystemExit(f"--layer {layer} is outside the model's {n_hidden} hidden states.")
    leading_space = reads_leading_space(tokeniser) if args.leading_space == "auto" else args.leading_space == "yes"
    print(f"[lm-table] reading hidden state {layer} of {n_hidden}; leading space {'on' if leading_space else 'off'}")

    table = encode_words(model, tokeniser, words, layer, device, leading_space, args.batch_words)

    #a near-constant table would make every word the same target
    spread = float(np.std(table))
    if spread < 1e-6 or not np.isfinite(table).all():
        raise SystemExit(f"Embeddings are degenerate (std {spread:.2e}); layer {layer} is unusable.")

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, words=np.array(words), embeddings=table, model=args.model, layer=layer)

    #how separable the 50 scored words are in this space; rows found as
    #TargetTable finds them, by the normalised word
    index = {normalise_word(word): row for row, word in enumerate(words)}
    primary = Vocabulary("primary").words
    unit = table[[index[word] for word in primary]]
    unit = unit / np.linalg.norm(unit, axis=1, keepdims=True)
    off_diagonal = (unit @ unit.T)[~np.eye(len(primary), dtype=bool)]
    print(
        f"[lm-table] wrote {args.out}: {table.shape[0]:,} x {table.shape[1]}; "
        f"primary words' mean off-diagonal cosine {off_diagonal.mean():.3f}, max {off_diagonal.max():.3f}"
    )


if __name__ == "__main__":
    main()
