"""P2 of the 13 Oct research note: train the row encoder, gate it on the testbed, embed the holdout.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.row_encoder --members trial13_g1_s0 ... \\
        --base submissions/ens_t22_tc1b75_ts.csv --holdout-dir data/holdout

    1   the holdout's 40 subjects: matched sentence epochs (cli.cross_graph's
        reader), MCCA at C components (the testbed's graph C), every voting
        row's 0.5 s window, residualized on the timing design
    2   the testbed (Sherlock1 ses-11 and ses-12, cli.testbed's sessions) in
        the SAME space: subject 0 through its holdout projection, each other
        subject mapped onto subject 0's components by ridge regression on
        ses-11 alone (`row_encoder.anchor_map`); the same windows, residualized
    3   one encoder trained on the holdout and ses-11 together -- no label is
        read: InfoNCE on row identity, the note's negatives masked -- and
        stopped on held-out sentences
    4   the gate, on ses-12, never seen: neighbour purity of the 12 others'
        embeddings against the testbed's own residual MCCA features, P1's
        (the note: use it only if it beats them); and P1 graph, SNF, P3 LDA,
        P8a product on the embeddings, tuned on ses-11, tested on ses-12,
        beside the MCCA features' own testbed gains

Writes data/trial23_row_encoder_holdout.npz (every subject's embedding of
every voting row) and data/trial23_row_encoder_settings.json -- which
`cli.cross_graph --features data/trial23_row_encoder_holdout.npz --settings
data/trial23_row_encoder_settings.json` reads -- the weights, and
runs/trial23_row_encoder.md. CPU or GPU, minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...config import REPO_ROOT, RUNS_DIR
from .. import crossfeatures as cf
from .. import crossvotes as cv
from .. import row_encoder as encoder
from ..crosssubject import robust_stats
from .cross_graph import load_holdout
from .ensemble import read_submission, standardise_rows
from .testbed import SESSIONS, WEIGHTS, hub_bacc, parameter_grid, prepare_session, residual_features

#C when the testbed's settings are not there yet
DEFAULT_COMPONENTS = 30

#the base mix's words a negative may not share with its anchor
TOP_WORDS = 3

#sentence ids of the testbed rows, kept apart from the holdout's
TESTBED_SENTENCE_OFFSET = 10 ** 7


def parse_args():
    """Command-line arguments for the row encoder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="+", required=True, help="the base mix's members, each with chains38 dumps")
    parser.add_argument("--base", required=True, help="the holdout mix before the hub correction, for the negatives' top words")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--matches", default="data/trial22_matches.json", help="repo-relative, from cli.repeat_check")
    parser.add_argument("--subjects", default="1-39")
    parser.add_argument("--map", default="data/trial23_testbed_map.npz", help="repo-relative, from cli.testbed_check")
    parser.add_argument("--reference-cache", default="data/trial10_reference", help="subject 0's cache")
    parser.add_argument("--others-cache", default="data/trial18_others", help="the other subjects' cache")
    parser.add_argument("--testbed-settings", default="data/trial23_testbed_settings.json", help="for C and the MCCA gains")
    parser.add_argument("--components", type=int, default=None, help="C; default the testbed's graph C")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embeddings-out", default="data/trial23_row_encoder_holdout.npz", help="repo-relative")
    parser.add_argument("--settings-out", default="data/trial23_row_encoder_settings.json", help="repo-relative")
    parser.add_argument("--weights-out", default="data/trial23_row_encoder.pt", help="repo-relative")
    parser.add_argument("--report", default=None, help="default runs/trial23_row_encoder.md")
    return parser.parse_args()


def top_words(scores, count=TOP_WORDS):
    """Each row's highest-scoring words.

    ARGS:
        scores (np.ndarray): [n, 50]
        count (int)

    RETURNS:
        [n, count] int64
    """
    #axis=1 is the word axis
    return np.argsort(-scores, axis=1)[:, :count].astype(np.int64)


def holdout_windows(holdout, n_components):
    """The holdout's MCCA fit, and every subject's residual windows of the voting rows.

    ARGS:
        holdout (dict): from cli.cross_graph's `load_holdout`
        n_components (int)

    RETURNS:
        windows [S, n, C, T] float32 (subject 0 first), subject 0's projection
        (whitener, block), the MCCA eigenvalues
    """
    filtered, whiteners, blocks, eigenvalues = cv.fit_shared(holdout["pieces"], n_components)
    projector = cf.residual_projector(cf.timing_design(holdout["groups"], holdout["onsets_s"]))
    windows = []
    for whitener, block, pieces in zip(whiteners, blocks, filtered):
        series = [block.T @ whitener(piece) for piece in pieces]
        features = cv.row_features(series, holdout["piece_of_row"], holdout["onset_in_piece"])
        windows.append(cf.residualize(features, projector).reshape(len(features), n_components, -1))
    return np.stack(windows).astype(np.float32), (whiteners[0], blocks[0]), eigenvalues


def scaled_filtered(pieces, stats=None):
    """A subject's pieces robust-scaled -- by `stats`, else their own -- then low-passed.

    ARGS:
        pieces (list): [306, n_i]
        stats (tuple or None): (median, iqr) from the fitting session

    RETURNS:
        list of [306, n_i], and the (median, iqr) used
    """
    median, iqr = stats if stats is not None else robust_stats(pieces)
    return [cf.lowpass((piece - median[:, None]) / iqr[:, None]) for piece in pieces], (median, iqr)


def testbed_windows(sessions, reference_projection, n_components):
    """Both testbed sessions in the holdout's space; every map and scaler fitted on ses-11.

    ARGS:
        sessions (dict): "11" and "12" -> from cli.testbed's `prepare_session`
        reference_projection (tuple): subject 0's holdout (whitener, block)
        n_components (int)

    RETURNS:
        {session: windows [S, n, C, T] float32 (subject 0 first)}, and the
        other subjects present in both sessions, in that order
    """
    whitener_zero, block_zero = reference_projection
    common = [subject for subject in sessions["11"]["subjects"] if subject in sessions["12"]["subjects"]]
    maps, stats, out = {}, {}, {}
    for session in ("11", "12"):
        prepared = sessions[session]
        position = {subject: index + 1 for index, subject in enumerate(prepared["subjects"])}

        #subject 0 through its own holdout projection; scaled as on ses-11
        own, stats[0] = scaled_filtered(prepared["pieces"][0], stats.get(0))
        reference = [block_zero.T @ whitener_zero(piece) for piece in own]
        series_by_subject, onsets_by_subject = [reference], [prepared["onsets"][0]]

        #each other subject: its map onto subject 0's components, fitted on ses-11's same audio
        for subject in common:
            own, stats[subject] = scaled_filtered(prepared["pieces"][position[subject]], stats.get(subject))
            if subject not in maps:
                maps[subject] = encoder.anchor_map(own, reference)
            whitener, weights = maps[subject]
            series_by_subject.append([weights.T @ whitener(piece) for piece in own])
            onsets_by_subject.append(prepared["onsets"][position[subject]])

        #the same windows as the holdout's, residualized on this session's timing
        rows = prepared["rows"]
        projector = cf.residual_projector(cf.timing_design(rows["groups"], rows["onsets"] / cf.RATE))
        out[session] = np.stack([cf.residualize(cv.row_features(series, prepared["piece_of_row"], onsets), projector)
                                 .reshape(len(rows["labels"]), n_components, -1)
                                 for series, onsets in zip(series_by_subject, onsets_by_subject)]).astype(np.float32)
    return out, common


def embedding_vote(method, embeddings, rows, base, params):
    """One method's vote from the other subjects' embeddings.

    ARGS:
        method (str), embeddings (np.ndarray): [S, n, D] subject 0 first
        rows (dict), base (np.ndarray): [n, 50], params (dict)

    RETURNS:
        vote [n, 50], info
    """
    voters = [table.astype(np.float64) for table in embeddings[1:]]
    if "temperature_scale" in params:
        params = dict(params, temperature=params["temperature_scale"] / np.sqrt(len(voters)))
    return cv.vote(method, voters, rows["groups"], base, params)


def tune_votes(tune_features, tune, test_features, test):
    """Every method on a feature set: tuned on ses-11, its gain measured on ses-12.

    Shared with cli.megxl_features (P4).

    ARGS:
        tune_features, test_features (np.ndarray): [S, n, D] subject 0 first
        tune, test (dict): from cli.testbed's `prepare_session`

    RETURNS:
        dict: method -> components (None), params, weight, tune_gain, test_gain
    """
    settings = {}
    baseline_tune = hub_bacc(tune["base"], tune["rows"]["labels"])
    baseline_test = hub_bacc(test["base"], test["rows"]["labels"])
    for method in cv.METHODS:
        #the setting and weight with the largest ses-11 gain
        best = None
        for params in parameter_grid(method, len(tune_features) - 1):
            vote, _ = embedding_vote(method, tune_features, tune["rows"], tune["base"], params)
            for weight in WEIGHTS:
                gain = hub_bacc(cv.blend(tune["base"], vote, weight), tune["rows"]["labels"]) - baseline_tune
                if best is None or gain > best[0]:
                    best = (gain, params, weight)
        gain_tune, params, weight = best

        #that choice, once, on the held-out session
        vote, _ = embedding_vote(method, test_features, test["rows"], test["base"], params)
        gain_test = hub_bacc(cv.blend(test["base"], vote, weight), test["rows"]["labels"]) - baseline_test
        settings[method] = {"components": None, "params": params, "weight": weight, "tune_gain": gain_tune, "test_gain": gain_test}
    return settings


def mean_purity(features_by_subject, rows):
    """Neighbour purity of the subjects' mean z-scored similarity, and the base rate.

    ARGS:
        features_by_subject (list): [n, D] each, the voting subjects
        rows (dict)

    RETURNS:
        purity, base rate
    """
    valid = cf.valid_pairs(rows["groups"])
    return cf.neighbour_purity(np.mean(cv.similarity_matrices(features_by_subject, valid), axis=0), valid, rows["labels"])


def main():
    """Windows in one space, the encoder, the gate, the votes, the outputs."""
    args = parse_args()
    from pnpl.competition import load_vocabulary

    words = (load_vocabulary("primary"), load_vocabulary("moses"))
    testbed_settings_path = REPO_ROOT / args.testbed_settings
    testbed_settings = json.loads(testbed_settings_path.read_text(encoding="utf-8")) if testbed_settings_path.exists() else None
    components = args.components or (testbed_settings["methods"]["graph"]["components"] if testbed_settings else DEFAULT_COMPONENTS)

    #1. the holdout: its MCCA space, its windows, its negatives' top words
    holdout = load_holdout(args.holdout_dir, REPO_ROOT / args.matches, args.subjects)
    holdout_input, reference_projection, eigenvalues = holdout_windows(holdout, components)
    index, primary, _ = read_submission(Path(args.base), *words)
    if not np.array_equal(index, np.arange(len(holdout["metadata"]))):
        raise SystemExit(f"{args.base}: its rows are not the holdout's 0..{len(holdout['metadata']) - 1}.")
    holdout_top = top_words(standardise_rows(primary)[holdout["voting"]])

    #2. the testbed sessions, cut as cli.testbed cuts them, in the holdout's space
    mapping = np.load(REPO_ROOT / args.map)
    sessions = {session: prepare_session(args.members, session, mapping, args) for session in SESSIONS}
    testbed_input, common = testbed_windows(sessions, reference_projection, components)

    #every input scaled per component by the training windows' spread; [rows, C, T] pooled
    training = np.concatenate([holdout_input.reshape(-1, components, holdout_input.shape[3]),
                               testbed_input["11"].reshape(-1, components, holdout_input.shape[3])])
    spread = np.maximum(training.std(axis=(0, 2)), 1e-6)[None, None, :, None].astype(np.float32)
    holdout_input = holdout_input / spread
    testbed_input = {session: windows / spread for session, windows in testbed_input.items()}

    #3. one encoder on the holdout and ses-11 together; no label read
    datasets = [{"windows": holdout_input, "sentence": holdout["groups"].astype(np.int64), "top_words": holdout_top},
                {"windows": testbed_input["11"], "sentence": sessions["11"]["rows"]["groups"] + TESTBED_SENTENCE_OFFSET,
                 "top_words": top_words(sessions["11"]["base"])}]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, history = encoder.train_encoder(datasets, components, epochs=args.epochs, patience=args.patience, seed=args.seed, device=device)
    embeddings = {"holdout": encoder.embed(model, holdout_input), "11": encoder.embed(model, testbed_input["11"]),
                  "12": encoder.embed(model, testbed_input["12"])}

    #4. the gate on ses-12: the encoder against P1's residual MCCA features, same rows
    test = sessions["12"]
    encoder_purity, base_rate = mean_purity(list(embeddings["12"][1:]), test["rows"])
    residual, raw, _ = residual_features(test["pieces"], test["onsets"], test["piece_of_row"], test["rows"], components)
    residual_purity, _ = mean_purity([subject.reshape(len(subject), -1) for subject in residual[1:]], test["rows"])
    raw_purity, _ = mean_purity([subject.reshape(len(subject), -1) for subject in raw[1:]], test["rows"])
    passes = bool(encoder_purity > residual_purity)

    #every method on the embeddings: tuned on ses-11, tested on ses-12
    method_settings = tune_votes(embeddings["11"], sessions["11"], embeddings["12"], test)

    #5. the outputs: the holdout embeddings, the settings cross_graph reads, the weights
    embeddings_path = REPO_ROOT / args.embeddings_out
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(embeddings_path, embeddings=embeddings["holdout"], subjects=np.array([0] + holdout["others"]),
             rows=np.flatnonzero(holdout["voting"]))
    settings = {"features": "row_encoder", "components": components, "passes": passes, "methods": method_settings,
                "purity": {"encoder": encoder_purity, "residual": residual_purity, "raw": raw_purity, "base_rate": base_rate},
                "testbed_subjects": common, "epochs_run": len(history)}
    settings_path = REPO_ROOT / args.settings_out
    settings_path.write_text(json.dumps(settings, indent=1, default=float) + "\n", encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "components": components, "spread": spread.ravel()},
               REPO_ROOT / args.weights_out)

    #the report: the data, the training, the gate, the votes beside the MCCA features'
    mcca_gains = testbed_settings["methods"] if testbed_settings else {}
    lines = ["# Trial 23 -- P2: the row-identification encoder", "",
             f"- holdout: {len(holdout_input)} subjects x {holdout_input.shape[1]} voting rows; MCCA at C = {components} "
             f"(eigenvalues {', '.join(f'{value:.1f}' for value in eigenvalues[:4])} ...)",
             f"- testbed: subject 0 + {len(common)} others in the holdout's space; ses-11 {testbed_input['11'].shape[1]} rows "
             f"(training), ses-12 {testbed_input['12'].shape[1]} rows (the gate, never seen)",
             f"- training: {len(history)} epochs; last train loss {history[-1][1]:.3f}, best held-out {min(h[2] for h in history):.3f}", "",
             "## The gate: neighbour purity on ses-12 (top-10, other chains only)", "",
             f"- encoder embeddings: {encoder_purity:.3f}", f"- residual MCCA features (P1's): {residual_purity:.3f}",
             f"- raw MCCA features: {raw_purity:.3f}", f"- base rate: {base_rate:.3f}", "",
             f"Verdict: {'PASSED -- the embeddings may replace the features in P1' if passes else 'not passed -- the note says do not use it'}.", "",
             "## The votes with the embeddings: tuned on ses-11, tested on ses-12", "",
             "| method | setting | weight | ses-11 gain | ses-12 gain | MCCA features' ses-12 gain |", "|---|---|---|---|---|---|"]
    for method, chosen in method_settings.items():
        mcca = f"{mcca_gains[method]['test_gain']:+.2f}" if method in mcca_gains else "-"
        lines.append(f"| {method} | {json.dumps(chosen['params'])} | {chosen['weight']} | {chosen['tune_gain']:+.2f} | "
                     f"{chosen['test_gain']:+.2f} | {mcca} |")
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_row_encoder.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[row_encoder] wrote {embeddings_path}, {settings_path} and {report}")


if __name__ == "__main__":
    main()
