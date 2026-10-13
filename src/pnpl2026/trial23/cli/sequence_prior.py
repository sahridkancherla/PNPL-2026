"""P5 of the 13 Oct research note: a skip-aware sequence prior, gated on labelled chains.

    # 1. the lm prior's counts, from GPT-2 medium's own text (GPU: pbs/trial23_lm.pbs)
    PYTHONPATH=$HOME/t5libs:src python -m pnpl2026.trial23.cli.sequence_prior lm

    # 2. the local check, every task's chains38 rows; tuned on validation, gated on MOCHA test
    PYTHONPATH=src python -m pnpl2026.trial23.cli.sequence_prior check --members trial13_g1_s0 ...

    # 3. the holdout: a transduced mix rescored, before hubnorm_holdout and lone_fill
    PYTHONPATH=src python -m pnpl2026.trial23.cli.sequence_prior apply --prior em \\
        --base submissions/ens_t22_tc1b75_ts.csv

`lm` samples text from GPT-2 medium unconditionally -- nothing about the
stimulus goes in -- and counts adjacent and one-skip transitions between the
50 scored words (`sequence_prior.count_transitions`), into
data/trial23_lm_transitions.npz with a report beside the runs.

`check` mirrors the holdout on subject 0's held-out sessions, each task on
its own: the 70.68 recipe's transduced mix (the setup from the partner
partition), the prior by forward-backward, then the hub correction fitted on
the task's rows. The temperature and the gap-based skip model are fitted on
validation; each prior's weight is chosen on pooled validation from
{0, 0.025, 0.05, 0.075, 0.1}, and its test gains are the unbiased numbers.
The note's gate is MOCHA-TIMIT test (TIMIT's sessions are all in training);
Sherlock and TheMoth test rows are reported beside it. The em prior is fitted
on the evaluated rows themselves, cross-fitted, as on the holdout. Writes
data/trial23_sequence_prior_settings.json and runs/trial23_sequence_prior_check.md.

`apply` rescores the holdout's sentence rows at the checked weight and writes
`<base stem>_p5<prior>.csv`; lone rows keep the base. A prior whose check
chose weight 0 is refused unless --weight is given.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR
from .. import sequence_prior as sp
from ..local_checks import bacc
from .boundary import chain_rows, hubbed, transduced_mix
from .ensemble import read_submission, softmax_rows, standardise_rows, zscore_scores

#the general language model: GPT-2 medium (2019), by its Hugging Face id
DEFAULT_LM = "openai-community/gpt2-medium"

#where the lm counts and the checked settings live
DEFAULT_TRANSITIONS = "data/trial23_lm_transitions.npz"
DEFAULT_SETTINGS = "data/trial23_sequence_prior_settings.json"

#the transition weights tried: the note's ceiling is 0.1
WEIGHTS = (0.0, 0.025, 0.05, 0.075, 0.1)

#the em prior's cross-fitting folds over chains, drawn once
N_FOLDS = 5
FOLD_SEED = 0

#every member since trial 10 reads data/trial10_reference, at 50 Hz: dump onsets are its samples
DUMP_RATE = 50.0

#the note's gate: TIMIT/MOCHA test, and TIMIT is all in training
GATE_TASK = "MOCHATIMIT"

#the two priors
PRIORS = ("lm", "em")

#sampled texts kept for the lm report
SAMPLES_KEPT = 5


def parse_args():
    """Command-line arguments: a mode, then its options."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("lm", "check", "apply"))
    parser.add_argument("--model", default=DEFAULT_LM, help="lm: the language model, an id in the HF cache or a folder")
    parser.add_argument("--tokens", type=int, default=4_000_000, help="lm: tokens sampled in all")
    parser.add_argument("--batch", type=int, default=64, help="lm: samples drawn together")
    parser.add_argument("--length", type=int, default=512, help="lm: tokens per sample")
    parser.add_argument("--top-p", type=float, default=0.95, help="lm: nucleus sampling's mass")
    parser.add_argument("--seed", type=int, default=0, help="lm: the sampling seed")
    parser.add_argument("--device", default=None)
    parser.add_argument("--transitions", default=DEFAULT_TRANSITIONS, help="the lm counts, written by lm, read by check and apply")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS, help="written by check, read by apply")
    parser.add_argument("--report", default=None, help="the report's path; each mode has its default under runs/")
    parser.add_argument("--members", nargs="+", default=None, help="check: the mix's members, each with chains38 dumps")
    parser.add_argument("--prior", choices=PRIORS, default=None, help="apply: which prior")
    parser.add_argument("--base", default=None, help="apply: the holdout mix after transduction, before the hub correction")
    parser.add_argument("--weight", type=float, default=None, help="apply: overrides the checked weight")
    parser.add_argument("--holdout-dir", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sample_top_p(logits, top_p, generator):
    """One token per row by nucleus sampling.

    ARGS:
        logits (torch.Tensor): [batch, vocabulary] float32
        top_p (float): the mass of the smallest head kept
        generator (torch.Generator)

    RETURNS:
        [batch, 1] int64 token ids
    """
    import torch

    #dim=-1 is the vocabulary axis; the first token always stays in
    probabilities = torch.softmax(logits, dim=-1)
    ordered, order = probabilities.sort(dim=-1, descending=True)
    outside = (ordered.cumsum(dim=-1) - ordered) > top_p
    ordered = ordered.masked_fill(outside, 0.0)
    choice = torch.multinomial(ordered, 1, generator=generator)
    return order.gather(-1, choice)


def end_of_text(model, tokenizer):
    """The end-of-text token's id and its decoded text.

    The tokenizer names it when its files say so; the model's config always does.

    ARGS:
        model: a causal language model
        tokenizer: its tokenizer

    RETURNS:
        (int, str)
    """
    token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id
    if token_id is None:
        raise SystemExit("Neither the tokenizer nor the model config names an end-of-text token.")
    return int(token_id), tokenizer.decode([int(token_id)])


def sample_batch(model, tokenizer, batch, length, top_p, generator, device):
    """`batch` texts of `length` tokens each, from the end-of-text token alone.

    ARGS:
        model: a causal language model, in eval mode on `device`
        tokenizer: its tokenizer
        batch (int), length (int), top_p (float)
        generator (torch.Generator)
        device (torch.device)

    RETURNS:
        list of str; each end-of-text token becomes a line break
    """
    import torch

    start_id, end_text = end_of_text(model, tokenizer)
    current = torch.full((batch, 1), start_id, dtype=torch.long, device=device)
    past, pieces = None, []
    with torch.no_grad():
        #one token at a time, the cache carrying everything before it
        for _ in range(length):
            out = model(input_ids=current, past_key_values=past, use_cache=True)
            past = out.past_key_values
            current = sample_top_p(out.logits[:, -1, :].float(), top_p, generator)
            pieces.append(current)

    #a new document is a sentence break, never part of a chain
    tokens = torch.cat(pieces, dim=1).cpu().numpy()
    return [tokenizer.decode(row.tolist()).replace(end_text, "\n") for row in tokens]


def lm(args):
    """Sample GPT-2's own text and count the 50 words' transitions in it."""
    import torch
    from pnpl.competition import load_vocabulary

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise SystemExit("transformers is not importable: install it aside the project venv and put it on PYTHONPATH "
                         "(pbs/trial23_lm.pbs says how).") from error

    #the model in half precision on a GPU; dtype set after loading works on every transformers version
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device=device, dtype=torch.float16 if device.type == "cuda" else torch.float32).eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    words = list(load_vocabulary("primary"))
    print(f"[lm] {args.model} on {device}: {args.tokens:,} tokens in batches of {args.batch} x {args.length}, top-p {args.top_p}")

    #counts accumulate batch by batch; a few texts kept for the report
    n_words = len(words)
    adjacent, skip, unigram = np.zeros((n_words, n_words)), np.zeros((n_words, n_words)), np.zeros(n_words)
    words_read, samples = 0, []
    n_batches = max(int(np.ceil(args.tokens / (args.batch * args.length))), 1)
    for batch_index in range(n_batches):
        texts = sample_batch(model, tokenizer, args.batch, args.length, args.top_p, generator, device)
        counts = sp.count_transitions(texts, words)
        adjacent, skip, unigram = adjacent + counts[0], skip + counts[1], unigram + counts[2]
        words_read += counts[3]
        samples.extend(texts[: max(SAMPLES_KEPT - len(samples), 0)])
        if (batch_index + 1) % 10 == 0 or batch_index + 1 == n_batches:
            print(f"[lm] {batch_index + 1}/{n_batches} batches: {words_read:,} words, {int(adjacent.sum()):,} adjacent "
                  f"and {int(skip.sum()):,} one-skip pairs", flush=True)

    #the counts, with what made them
    out_path = REPO_ROOT / args.transitions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, adjacent=adjacent, skip=skip, unigram=unigram, words_read=words_read, words=np.array(words),
             model=args.model, top_p=args.top_p, seed=args.seed, tokens=n_batches * args.batch * args.length)

    #the report: shares against the holdout's, the strongest lifts, a few texts
    adjacent_pmi = sp.pmi(sp.conditional(adjacent))
    strongest = np.dstack(np.unravel_index(np.argsort(-adjacent_pmi, axis=None)[:12], adjacent_pmi.shape))[0]
    lines = ["# Trial 23 -- P5's lm prior: GPT-2's own text, counted", "",
             f"{args.model}, {n_batches * args.batch * args.length:,} tokens (top-p {args.top_p}, seed {args.seed}): "
             f"{words_read:,} words, {100 * unigram.sum() / max(words_read, 1):.1f}% of them scored words (holdout sentences: about 40%).",
             f"Chain pairs: {int(adjacent.sum()):,} adjacent, {int(skip.sum()):,} one-skip -- a one-skip share of "
             f"{100 * skip.sum() / max(adjacent.sum() + skip.sum(), 1):.1f}% (holdout: about 30%).", "",
             "Strongest adjacent lifts (PMI, shrunk with kappa = 50): "
             + ", ".join(f"{words[a]} {words[b]} ({adjacent_pmi[a, b]:+.2f})" for a, b in strongest), "",
             "First sampled texts (first 300 characters each):", ""]
    lines += [f"- {text[:300]!r}" for text in samples]
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_lm_transitions.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[lm] wrote {out_path} and {report}")


def load_transitions(path, words):
    """The lm prior's adjacent and one-skip conditionals, checked against the scored order.

    ARGS:
        path (Path)
        words (list): the scored vocabulary

    RETURNS:
        (adjacent, skip): `sequence_prior.conditional` tuples
    """
    with np.load(path) as stored:
        #⚠️ the counts' words in the scored order, or every transition is another word's
        stored_words = [sp.normalise_text(str(word)).strip() for word in stored["words"]]
        if stored_words != [sp.normalise_text(word).strip() for word in words]:
            raise SystemExit(f"{path}: its words are not the scored vocabulary, in order.")
        return sp.conditional(stored["adjacent"]), sp.conditional(stored["skip"])


def partition_rows(members, partition):
    """One partition's chains38 rows of every task: the transduced mix per task, and the chains.

    ARGS:
        members (list): run names
        partition (str)

    RETURNS:
        rows (dict, with onset_s), base [n, 50] before the hub correction, chains (list)
    """
    rows, scores = chain_rows(members, partition, tasks=None)
    missing = [name for name in ("onset", "table_index") if name not in rows]
    if missing:
        raise SystemExit(f"The {partition} chains38 dumps have no {missing}: dump them with trial 23's cli.embed.")

    #each task transduced on its own rows, as the holdout's one session is
    base = np.zeros((len(rows["labels"]), scores[0].shape[1]))
    for task in np.unique(rows["task"]):
        of_task = rows["task"] == task
        base[of_task] = transduced_mix(members, [table[of_task] for table in scores], rows["groups"][of_task], partition)
    rows["onset_s"] = rows["onset"] / DUMP_RATE
    return rows, base, sp.chain_order(rows["groups"], rows["onset"])


def task_scores(rows, scores):
    """BAcc@10 per task and pooled, each task's hub correction fitted on its own rows.

    ARGS:
        rows (dict)
        scores (np.ndarray): [n, 50]

    RETURNS:
        dict: task (and "pooled") -> BAcc@10
    """
    corrected = np.zeros_like(scores)
    tasks = np.unique(rows["task"])
    for task in tasks:
        corrected[rows["task"] == task] = hubbed(scores[rows["task"] == task])
    result = {str(task): bacc(corrected[rows["task"] == task], rows["labels"][rows["task"] == task]) for task in tasks}
    result["pooled"] = bacc(corrected, rows["labels"])
    return result


def pair_pmi_function(prior, base, chains, chain_skips, temperature, lm_conditionals):
    """One prior's PMI for every neighbouring pair of one set of chains.

    ARGS:
        prior (str): "lm" or "em"
        base (np.ndarray): [n, 50] the rows' base scores
        chains (list), chain_skips (list): from `sequence_prior`
        temperature (float): the base's calibrated temperature
        lm_conditionals (tuple or None): from `load_transitions`

    RETURNS:
        callable: (chain index, pair position) -> [50, 50] PMI
    """
    if prior == "lm":
        adjacent, skip = lm_conditionals
        return lambda chain, position: sp.mixed_pmi(adjacent, skip, chain_skips[chain][position])

    #em: the base's own posteriors, each fold's prior from the other folds
    posteriors = softmax_rows(temperature * zscore_scores(base))
    folds = sp.chain_folds(len(chains), N_FOLDS, FOLD_SEED)
    by_fold = sp.em_conditionals_by_fold(posteriors, chains, folds, chain_skips)
    return lambda chain, position: sp.mixed_pmi(*by_fold[folds[chain]], chain_skips[chain][position])


def check(args):
    """Tune on validation, measure on test, for every prior; write the settings."""
    from pnpl.competition import load_vocabulary
    from sklearn.metrics import roc_auc_score

    if not args.members:
        raise SystemExit("check needs --members.")
    words = list(load_vocabulary("primary"))
    parts = {partition: partition_rows(args.members, partition) for partition in ("validation", "test")}

    #the emission temperature and the skip model, from validation alone
    rows, base, chains = parts["validation"]
    temperature = sp.calibrate_temperature(zscore_scores(base), rows["labels"])
    steps = sp.pair_steps(rows["table_index"], chains)
    skip_model = sp.fit_skip_model(sp.pair_gap_features(rows["onset_s"], chains), steps == 2)
    test_rows, _, test_chains = parts["test"]
    test_steps = sp.pair_steps(test_rows["table_index"], test_chains)
    test_skips = sp.skip_probability(sp.pair_gap_features(test_rows["onset_s"], test_chains), skip_model)
    auc = roc_auc_score(test_steps == 2, test_skips) if 0 < (test_steps == 2).mean() < 1 else float("nan")

    #every prior at every weight, on both partitions
    transitions_path = REPO_ROOT / args.transitions
    lm_conditionals = load_transitions(transitions_path, words) if transitions_path.exists() else None
    priors = [prior for prior in PRIORS if prior != "lm" or lm_conditionals is not None]
    results = {}
    for partition, (rows, base, chains) in parts.items():
        chain_skips = sp.chain_skip_probabilities(rows["onset_s"], chains, skip_model)
        for prior in priors:
            pair_pmi = pair_pmi_function(prior, base, chains, chain_skips, temperature, lm_conditionals)
            for weight in WEIGHTS:
                results[(prior, partition, weight)] = task_scores(rows, sp.rescore(base, chains, temperature, weight, pair_pmi))
            print(f"[p5] {prior} on {partition}: " + ", ".join(f"{weight:g} -> {results[(prior, partition, weight)]['pooled']:.2f}"
                                                             for weight in WEIGHTS))

    #each prior's weight on pooled validation (ties to the smaller); its test gains, task by task
    settings = {"temperature": temperature, "skip_model": skip_model, "kappa": sp.KAPPA, "n_folds": N_FOLDS,
                "fold_seed": FOLD_SEED, "dump_rate": DUMP_RATE, "gate_task": GATE_TASK, "priors": {}}
    for prior in priors:
        chosen = max(WEIGHTS, key=lambda weight: (results[(prior, "validation", weight)]["pooled"], -weight))
        test_gain = {task: results[(prior, "test", chosen)][task] - results[(prior, "test", 0.0)][task]
                     for task in results[(prior, "test", 0.0)]}
        settings["priors"][prior] = {
            "weight": chosen,
            "validation_gain": results[(prior, "validation", chosen)]["pooled"] - results[(prior, "validation", 0.0)]["pooled"],
            "test_gain": test_gain,
            "passes": bool(chosen > 0 and test_gain.get(GATE_TASK, -np.inf) > 0),
        }
    settings_path = REPO_ROOT / args.settings
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")

    #the report: the rows, the skip model, every prior's table and verdict
    lines = ["# Trial 23 -- P5: a skip-aware sequence prior, checked on labelled chains", "",
             f"{len(args.members)} members; chains38 rows of every task, each transduced (70.68 recipe) and hub-corrected on its own rows.",
             *[f"- {partition}: {len(rows['labels']):,} rows in {len(chains):,} chains, tasks "
               + ", ".join(f"{task} {int((rows['task'] == task).sum())}" for task in np.unique(rows["task"]))
               for partition, (rows, _, chains) in parts.items()],
             f"- emission temperature (validation NLL): {temperature:.3f}",
             f"- skip model on validation's {len(steps):,} pairs ({100 * (steps == 2).mean():.1f}% skip a word; "
             f"steps other than 1 or 2: {int(((steps != 1) & (steps != 2)).sum())}); test AUC {auc:.3f}",
             "" if lm_conditionals is not None else f"- no {args.transitions}: the lm prior is not checked (run the lm mode first)", ""]
    tasks = [task for task in results[(priors[0], "test", 0.0)] if task != "pooled"]
    for prior in priors:
        lines += [f"## {prior}", "", "| weight | validation pooled | test pooled | " + " | ".join(f"test {task}" for task in tasks) + " |",
                  "|---" * (3 + len(tasks)) + "|"]
        for weight in WEIGHTS:
            test = results[(prior, "test", weight)]
            mark = " **(chosen)**" if weight == settings["priors"][prior]["weight"] else ""
            lines.append(f"| {weight:g}{mark} | {results[(prior, 'validation', weight)]['pooled']:.2f} | {test['pooled']:.2f} | "
                         + " | ".join(f"{test[task]:.2f}" for task in tasks) + " |")
        entry = settings["priors"][prior]
        verdict = ("passes the note's gate" if entry["passes"] else "does not pass the note's gate") + f" ({GATE_TASK} test)"
        lines += ["", f"Verdict: weight {entry['weight']:g} {verdict}; test gains " +
                  ", ".join(f"{task} {gain:+.2f}" for task, gain in entry["test_gain"].items()) + ".", ""]
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_sequence_prior_check.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[p5] wrote {settings_path} and {report}")


def holdout_chains(metadata):
    """The holdout's sentences as chains of canonical rows, in onset order; lone rows in none.

    ARGS:
        metadata (list): the loader's per-row metadata

    RETURNS:
        chains (list of canonical-row arrays), onsets_s [n_rows] (lone rows 0)
    """
    sentence = np.array([index for index, meta in enumerate(metadata) if meta["source"] == "sentence"])
    keys = np.array([f"{metadata[index]['subject']}/{metadata[index]['epoch']}" for index in sentence])
    groups = np.unique(keys, return_inverse=True)[1]
    onsets = np.array([float(metadata[index]["onset_s"]) for index in sentence])

    #chain positions back to canonical rows
    chains = [sentence[chain] for chain in sp.chain_order(groups, onsets)]
    onsets_s = np.zeros(len(metadata))
    onsets_s[sentence] = onsets
    return chains, onsets_s


def apply(args):
    """Rescore a holdout mix's sentence rows with one checked prior; write the file."""
    from pnpl.competition import LibriBrainCompetitionHoldout, load_vocabulary, write_submission

    if not args.prior or not args.base:
        raise SystemExit("apply needs --prior and --base.")
    settings = json.loads((REPO_ROOT / args.settings).read_text(encoding="utf-8"))
    if args.prior not in settings["priors"]:
        raise SystemExit(f"{args.settings} has no {args.prior} prior: run the check with it first.")
    entry = settings["priors"][args.prior]
    weight = entry["weight"] if args.weight is None else args.weight
    if weight <= 0:
        raise SystemExit(f"The check chose weight 0 for {args.prior}: nothing to submit (--weight overrides).")
    if not entry["passes"]:
        print(f"[p5] warning: {args.prior} did not pass the note's gate ({GATE_TASK} test); the note says do not submit it.")

    #the base, row-aligned with the loader
    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    index, primary, moses = read_submission(Path(args.base), *words)
    metadata = LibriBrainCompetitionHoldout(track="deep", data_path=args.holdout_dir).metadata
    if not np.array_equal(index, np.arange(len(metadata))):
        raise SystemExit(f"{args.base}: its rows are not the holdout's 0..{len(metadata) - 1}.")
    chains, onsets_s = holdout_chains(metadata)
    base = standardise_rows(primary)

    #the prior on the holdout's own chains: lm's counts, or em fitted here, cross-fitted
    chain_skips = sp.chain_skip_probabilities(onsets_s, chains, settings["skip_model"])
    lm_conditionals = load_transitions(REPO_ROOT / args.transitions, list(words[0])) if args.prior == "lm" else None
    pair_pmi = pair_pmi_function(args.prior, base, chains, chain_skips, settings["temperature"], lm_conditionals)
    rescored = sp.rescore(base, chains, settings["temperature"], weight, pair_pmi)

    #⚠️ lone rows and one-word sentences are never touched
    in_chain = np.zeros(len(metadata), dtype=bool)
    for chain in chains:
        in_chain[chain] = len(chain) >= 2
    if not np.allclose(rescored[~in_chain], zscore_scores(base[~in_chain])):
        raise SystemExit("A row outside every chain changed: refusing to write.")
    out_path = Path(args.base).with_name(f"{Path(args.base).stem}_p5{args.prior}.csv")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    write_submission(str(out_path), indices=index, primary_probs=softmax_rows(rescored),
                     secondary_probs=moses / moses.sum(axis=1, keepdims=True))

    #how much moved: each chained row's top 10 against the base's
    top = lambda table: np.argpartition(-table, 9, axis=1)[:, :10]
    overlap = np.array([len(set(a) & set(b)) for a, b in zip(top(base[in_chain]), top(rescored[in_chain]))])
    lines = [f"# Trial 23 -- P5 applied: `{Path(args.base).name}` rescored with the {args.prior} prior", "",
             f"Weight {weight:g} (checked: {entry['weight']:g}; passes the gate: {entry['passes']}); temperature "
             f"{settings['temperature']:.3f}; {len(chains)} sentences, {int(in_chain.sum())} chained rows.",
             f"Top-10 sets kept whole on {100 * np.mean(overlap == 10):.1f}% of chained rows; mean overlap {overlap.mean():.2f} of 10.",
             f"Wrote `{out_path.name}`. Then hubnorm_holdout and lone_fill, as for every candidate."]
    report = Path(args.report) if args.report else RUNS_DIR / f"trial23_sequence_prior_apply_{out_path.stem}.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[p5] wrote {out_path} and {report}")


def main():
    """Sample, check or apply."""
    args = parse_args()
    {"lm": lm, "check": check, "apply": apply}[args.mode](args)


if __name__ == "__main__":
    main()
