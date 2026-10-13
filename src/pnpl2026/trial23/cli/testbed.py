"""The labelled cross-subject testbed: subject 0's Sherlock chapters, 12 other listeners, real labels.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.testbed --members trial13_g1_s0 ... \\
        [--others-cache data/trial18_others] [--map data/trial23_testbed_map.npz]

Step 0 of the 13 Oct research note. Subject 0's validation and test chapters
(Sherlock1 ses-11, ses-12) were heard by the cached broad subjects too
(`cli/testbed_check.py` confirms it word by word). Cut into holdout-shaped
rows (`chains38`: 3-8 vocabulary words, `cli/embed.py`'s dumps), they are a
labelled replica of the holdout:

    base        each member's scores on the rows, transduced as the 70.68
                file was (v1, pull 1, blend 0.75; tau and prototypes from the
                OTHER partition's dumps, so no row sees its own label),
                mixed, then the hub correction fitted on the rows
    votes       the other subjects' aligned recordings of the same words,
                through `crossvotes` -- exactly the holdout's pipeline:
                graph (P1), SNF, LDA (P3), product (P8a)
    tune/test   every setting chosen on ses-11, reported on ses-12

Plus the note's diagnostics -- neighbour purity against the base rate (raw
and residual), split-half reliability, the echo with the base and with the
gap voter -- and the gain with 3, 6 and 12 subjects (5 draws each), the
square-root extrapolation to 39 as an upper bound.

Writes runs/trial23_testbed.md and data/trial23_testbed_settings.json, which
`cli/cross_graph.py` reads. CPU, minutes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR
from .. import crossfeatures as cf
from .. import crossvotes as cv
from ..gaps import GapModel, chain_features
from ..hubnorm import apply_correction, fit_zscore
from ..local_checks import bacc, ensemble, member_dumps
from .knn_vote import load as load_dump
from .transduce import member_setup, run_member

#the partition each session's rows come from, and the one its setup comes from
SESSIONS = {"11": ("validation", "test"), "12": ("test", "validation")}

#span around a chain for the MCCA fit: 0.5 s before its first onset, 1.7 s after its last
SPAN_BEFORE, SPAN_AFTER = 25, 85

#the grids, fixed in advance
COMPONENTS = (20, 30, 40)
ALPHAS = (0.7, 0.8, 0.9)
TEMPERATURE_SCALES = (0.5, 1.0, 2.0)
WEIGHTS = (0.1, 0.2, 0.35, 0.5, 0.75)
SUBSET_SIZES = (3, 6, 12)
SUBSET_DRAWS = 5


def parse_args():
    """Command-line arguments for the testbed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--members", nargs="+", required=True, help="the base mix's members, each with chains38 dumps")
    parser.add_argument("--map", default="data/trial23_testbed_map.npz", help="repo-relative, from cli.testbed_check")
    parser.add_argument("--reference-cache", default="data/trial10_reference", help="subject 0's cache")
    parser.add_argument("--others-cache", default="data/trial18_others", help="the other subjects' cache")
    parser.add_argument("--gap-model", default="data/trial22_gap_model.npz", help="the onset-gap voter, for the echo check")
    parser.add_argument("--settings", default="data/trial23_testbed_settings.json", help="repo-relative output")
    parser.add_argument("--out", default=str(RUNS_DIR / "trial23_testbed.md"))
    return parser.parse_args()


def session_rows(members, session):
    """Subject 0's testbed rows for one session, and every member's scores on them.

    ARGS:
        members (list): run names
        session (str): "11" or "12"

    RETURNS:
        rows (dict: labels, groups, onsets, table_index, key), member scores
        (list of [n, 50] standardised)
    """
    partition = SESSIONS[session][0]
    dumps = [load_dump(member, f"chains38_{partition}") for member in members]
    first = dumps[0]
    for dump in dumps[1:]:
        #⚠️ the same tokens in the same order, or the mix mixes words
        if not np.array_equal(dump["primary"], first["primary"]):
            raise SystemExit("The members' chains38 dumps list different tokens; re-run cli.embed.")
    keep = np.char.find(first["key"].astype(str), f"ses-{session}_task-Sherlock1") >= 0
    keep &= first["primary"] >= 0
    rows = {"labels": first["primary"][keep].astype(np.int64), "groups": first["sequence"][keep].astype(np.int64),
            "onsets": first["onset"][keep].astype(np.int64), "table_index": first["table_index"][keep].astype(np.int64),
            "key": str(first["key"][keep][0])}
    scores = [cf.zscore_rows(dump["logits"][keep].astype(np.float64)) for dump in dumps]
    return rows, scores


def base_pipeline(members, scores, rows, session):
    """The 70.68 pipeline on the rows: transduced members, mixed, before the hub.

    ARGS:
        members (list), scores (list), rows (dict), session (str)

    RETURNS:
        plain mix [n, 50], transduced mix [n, 50] (both standardised, pre-hub)
    """
    #the setup from the OTHER partition, so no row sees its own label
    setups = member_dumps(members, "reference", SESSIONS[session][1])
    settings = SimpleNamespace(pull=1.0, blend=0.75, neighbours=3, share_floor=0.005)
    transduced = []
    for member_scores, dump in zip(scores, setups):
        temperature, prototypes, _ = member_setup(dump)
        transduced.append(run_member(member_scores, member_scores, rows["groups"], temperature, prototypes, settings)[0])
    return ensemble(scores), ensemble(transduced)


def hub_bacc(scores, labels):
    """BAcc@10 after the hub correction fitted on these rows, as on the holdout.

    ARGS:
        scores (np.ndarray): [n, 50]
        labels (np.ndarray): [n]

    RETURNS:
        float
    """
    return bacc(apply_correction(scores, fit_zscore(scores)), labels)


def aligned_pieces(rows, session, mapping, args):
    """Every subject's piece of MEG around each chain, aligned to subject 0's.

    ARGS:
        rows (dict): from `session_rows`
        session (str)
        mapping (NpzFile): from cli.testbed_check
        args (argparse.Namespace)

    RETURNS:
        pieces (list per subject, 0 first, of lists per chain of [306, n]),
        onsets (list per subject of [n_rows] onset within the row's piece),
        piece_of_row [n_rows], subjects (list of str, the others), covered [n_rows] bool
    """
    reference = np.load(REPO_ROOT / args.reference_cache / f"meg_{rows['key']}.npy", mmap_mode="r")
    subjects = [str(subject) for subject in mapping["subjects"] if f"sub{subject}_ses{session}_pairs" in mapping.files]

    #each other subject's table rows for subject 0's: exact pairs only
    others = []
    covered = np.ones(len(rows["labels"]), dtype=bool)
    for subject in subjects:
        pairs = mapping[f"sub{subject}_ses{session}_pairs"]
        lookup = dict(zip(pairs[:, 0].tolist(), pairs[:, 1].tolist()))
        other_key = str(mapping[f"sub{subject}_ses{session}_other_key"])
        with np.load(REPO_ROOT / args.others_cache / f"words_{other_key}.npz") as table:
            other_onsets = table["onset_sample"].astype(np.int64)
        mapped = np.array([lookup.get(int(index), -1) for index in rows["table_index"]])
        covered &= mapped >= 0
        others.append({"meg": np.load(REPO_ROOT / args.others_cache / f"meg_{other_key}.npy", mmap_mode="r"),
                       "onsets": np.where(mapped >= 0, other_onsets[np.maximum(mapped, 0)], -1),
                       "offset": int(mapping[f"sub{subject}_ses{session}_offset50"])})

    #one piece per chain: subject 0's span, shifted by each subject's offset
    chains = np.unique(rows["groups"])
    piece_of_row = np.searchsorted(chains, rows["groups"])
    pieces = [[] for _ in range(len(subjects) + 1)]
    onsets = [np.zeros(len(rows["labels"]), dtype=np.int64) for _ in range(len(subjects) + 1)]
    for piece, chain in enumerate(chains):
        members = np.flatnonzero(rows["groups"] == chain)
        start = max(int(rows["onsets"][members].min()) - SPAN_BEFORE, 0)
        stop = min(int(rows["onsets"][members].max()) + SPAN_AFTER, reference.shape[0])
        pieces[0].append(np.asarray(reference[start:stop], dtype=np.float64).T)
        onsets[0][members] = rows["onsets"][members] - start
        for index, other in enumerate(others, start=1):
            #the same audio span in this subject's recording, clipped to it
            other_start = min(max(start + other["offset"], 0), other["meg"].shape[0] - (stop - start))
            pieces[index].append(np.asarray(other["meg"][other_start : other_start + stop - start], dtype=np.float64).T)
            onsets[index][members] = np.clip(np.where(other["onsets"][members] >= 0, other["onsets"][members] - other_start,
                                                      rows["onsets"][members] - start), 0, stop - start - 1)
    return pieces, onsets, piece_of_row, subjects, covered


def prepare_session(members, session, mapping, args):
    """One session's rows every subject heard: their pieces and the base mixes.

    Shared with cli.row_encoder (P2), so both cut a session identically.

    ARGS:
        members (list), session (str), mapping (NpzFile), args (argparse.Namespace)

    RETURNS:
        dict: rows, pieces, onsets, piece_of_row, subjects, covered (before the
        cut), plain and base (the transduced mix), both pre-hub
    """
    rows, scores = session_rows(members, session)
    pieces, onsets, piece_of_row, subjects, covered = aligned_pieces(rows, session, mapping, args)
    if not covered.all():
        #only rows every subject heard: the vote must have the same voters everywhere
        keep = np.flatnonzero(covered)
        rows = {name: (value[keep] if isinstance(value, np.ndarray) else value) for name, value in rows.items()}
        scores = [score[keep] for score in scores]
        pieces, onsets, piece_of_row, subjects, _ = aligned_pieces(rows, session, mapping, args)
    plain, base = base_pipeline(members, scores, rows, session)
    return {"rows": rows, "pieces": pieces, "onsets": onsets, "piece_of_row": piece_of_row, "subjects": subjects,
            "covered": covered, "plain": plain, "base": base}


def residual_features(pieces, onsets, piece_of_row, rows, n_components):
    """Every subject's residual window features at C components, subject 0's included.

    ARGS:
        pieces, onsets, piece_of_row: from `aligned_pieces`
        rows (dict)
        n_components (int): the largest C of the grid; smaller C slice it

    RETURNS:
        residual features per subject [n, C, 25] (C = n_components), raw
        (unresidualized) the same, the MCCA eigenvalues
    """
    series, eigenvalues = cv.shared_series(pieces, n_components)
    projector = cf.residual_projector(cf.timing_design(rows["groups"], rows["onsets"] / cf.RATE))
    raw = [cv.row_features(subject_series, piece_of_row, subject_onsets).reshape(len(piece_of_row), n_components, -1)
           for subject_series, subject_onsets in zip(series, onsets)]
    residual = [cf.residualize(features.reshape(len(features), -1), projector).reshape(features.shape) for features in raw]
    return residual, raw, eigenvalues


def gap_scores(rows, model):
    """The timing-only voter's scores on the rows: zero where a row has no gap.

    ARGS:
        rows (dict)
        model (GapModel)

    RETURNS:
        [n, 50] float64
    """
    scores = np.zeros((len(rows["labels"]), 50))
    for chain in np.unique(rows["groups"]):
        members = np.flatnonzero(rows["groups"] == chain)
        features, _ = chain_features(rows["onsets"][members] / cf.RATE, np.zeros(len(members)))
        if len(features):
            scores[members[:-1]] = model.log_likelihood(features)
    return scores


def method_votes(method, features, rows, base, n_components, params):
    """One method's vote from the other subjects at C components.

    ARGS:
        method (str), features (list): residual [n, C_max, 25] per subject, 0 first
        rows (dict), base (np.ndarray): [n, 50] the transduced mix
        n_components (int), params (dict)

    RETURNS:
        vote [n, 50], info dict
    """
    others = [subject[:, :n_components, :].reshape(len(subject), -1) for subject in features[1:]]

    #the product's temperature is stored as a scale: T = scale / sqrt(subjects voting)
    if "temperature_scale" in params:
        params = dict(params, temperature=params["temperature_scale"] / np.sqrt(len(others)))
    return cv.vote(method, others, rows["groups"], base, params)


def parameter_grid(method, n_others):
    """Every setting tried for a method.

    ARGS:
        method (str)
        n_others (int): subjects voting (the product's scale is per sqrt of them)

    RETURNS:
        list of dict
    """
    if method in ("graph", "snf"):
        return [{"alpha": alpha} for alpha in ALPHAS]
    if method == "product":
        return [{"temperature_scale": scale} for scale in TEMPERATURE_SCALES]
    return [{}]


def main():
    """Tune every method on ses-11, test it on ses-12, scale it, diagnose it, save the settings."""
    args = parse_args()
    mapping = np.load(REPO_ROOT / args.map)
    with np.load(REPO_ROOT / args.gap_model) as stored:
        gap_model = GapModel(stored["means"], stored["spreads"], stored["counts"])

    lines = ["# Trial 23 -- the labelled Sherlock testbed", "",
             f"{len(args.members)} members; rows: chains38 on subject 0's Sherlock1 ses-11 (tuning) and ses-12 (test); "
             "each setting chosen on ses-11, reported on ses-12. BAcc@10 after the hub correction fitted on the rows.", ""]
    results = {}
    prepared = {}
    for session in SESSIONS:
        session_data = prepare_session(args.members, session, mapping, args)
        rows, subjects, covered = session_data["rows"], session_data["subjects"], session_data["covered"]
        plain, base = session_data["plain"], session_data["base"]
        residual, raw, eigenvalues = residual_features(session_data["pieces"], session_data["onsets"], session_data["piece_of_row"],
                                                       rows, max(COMPONENTS))
        prepared[session] = {"rows": rows, "base": base, "residual": residual, "raw": raw, "subjects": subjects}
        lines.append(f"- ses-{session}: {len(rows['labels'])} rows in {len(np.unique(rows['groups']))} chains, "
                     f"{len(subjects)} other subjects (rows every one heard: {int(covered.sum())}); plain mix "
                     f"{hub_bacc(plain, rows['labels']):.2f}, transduced {hub_bacc(base, rows['labels']):.2f}; "
                     f"MCCA eigenvalues {', '.join(f'{value:.1f}' for value in eigenvalues[:6])} ...")

    #the grid on ses-11, the chosen setting on ses-12
    lines += ["", "## Each method: tuned on ses-11, tested on ses-12", "",
              "| method | C | setting | weight | ses-11 gain | ses-12 gain |", "|---|---|---|---|---|---|"]
    settings = {}
    for method in cv.METHODS:
        best = None
        tune = prepared["11"]
        baseline = hub_bacc(tune["base"], tune["rows"]["labels"])
        for components in COMPONENTS:
            for params in parameter_grid(method, len(tune["subjects"])):
                vote, _ = method_votes(method, tune["residual"], tune["rows"], tune["base"], components, params)
                for weight in WEIGHTS:
                    gain = hub_bacc(cv.blend(tune["base"], vote, weight), tune["rows"]["labels"]) - baseline
                    if best is None or gain > best[0]:
                        best = (gain, components, params, weight)
        gain_tune, components, params, weight = best
        test = prepared["12"]
        vote, info = method_votes(method, test["residual"], test["rows"], test["base"], components, params)
        gain_test = hub_bacc(cv.blend(test["base"], vote, weight), test["rows"]["labels"]) - hub_bacc(test["base"], test["rows"]["labels"])
        settings[method] = {"components": components, "params": params, "weight": weight,
                            "tune_gain": gain_tune, "test_gain": gain_test}
        results[method] = (vote, info)
        lines.append(f"| {method} | {components} | {json.dumps(params)} | {weight} | {gain_tune:+.2f} | {gain_test:+.2f} |")

    #the note's diagnostics, on ses-12
    test = prepared["12"]
    valid = cf.valid_pairs(test["rows"]["groups"])
    components = settings["graph"]["components"]
    residual_matrices = cv.similarity_matrices([subject[:, :components, :].reshape(len(subject), -1) for subject in test["residual"][1:]], valid)
    raw_matrices = cv.similarity_matrices([subject[:, :components, :].reshape(len(subject), -1) for subject in test["raw"][1:]], valid)
    residual_purity, base_rate = cf.neighbour_purity(np.mean(residual_matrices, axis=0), valid, test["rows"]["labels"])
    raw_purity, _ = cf.neighbour_purity(np.mean(raw_matrices, axis=0), valid, test["rows"]["labels"])
    halves = cf.split_half_reliability(residual_matrices, valid)
    gaps = gap_scores(test["rows"], gap_model)
    lines += ["", "## Diagnostics on ses-12", "",
              f"- neighbour purity (top-10, other chains only): residual graph {residual_purity:.3f}, raw graph "
              f"{raw_purity:.3f}, base rate {base_rate:.3f} -- the note's bar is 2x the base rate "
              f"({'PASSED' if residual_purity >= 2 * base_rate else 'not passed'})",
              f"- split-half reliability: halves r = {halves['half_correlation']:+.3f}, Spearman-Brown "
              f"{halves['spearman_brown']:+.3f}; neighbour overlap {halves['neighbour_overlap']:.3f} against chance {halves['chance']:.3f}"]
    for method, (vote, _) in results.items():
        lines.append(f"- echo, {method}: row-specific r with the base {cf.row_specific_correlation(vote, test['base']):+.3f}, "
                     f"with the gap voter {cf.row_specific_correlation(vote, gaps):+.3f}")

    #the gain with fewer subjects, and its square-root extrapolation
    lines += ["", "## Gain on ses-12 with fewer subjects (5 draws each; chosen settings)", "",
              "| method | " + " | ".join(f"{size} subjects" for size in SUBSET_SIZES) + " | extrapolated to 39 |",
              "|---|" + "---|" * (len(SUBSET_SIZES) + 1)]
    generator = np.random.default_rng(0)
    baseline = hub_bacc(test["base"], test["rows"]["labels"])
    for method, chosen in settings.items():
        means = []
        for size in SUBSET_SIZES:
            gains = []
            for _ in range(SUBSET_DRAWS if size < len(test["subjects"]) else 1):
                chosen_subjects = generator.choice(len(test["subjects"]), min(size, len(test["subjects"])), replace=False)
                features = [test["residual"][0]] + [test["residual"][1 + index] for index in chosen_subjects]
                vote, _ = method_votes(method, features, test["rows"], test["base"], chosen["components"], chosen["params"])
                gains.append(hub_bacc(cv.blend(test["base"], vote, chosen["weight"]), test["rows"]["labels"]) - baseline)
            means.append(float(np.mean(gains)))
        extrapolated = means[-1] * np.sqrt(39 / SUBSET_SIZES[-1])
        settings[method]["scaling"] = dict(zip([str(size) for size in SUBSET_SIZES], means))
        lines.append(f"| {method} | " + " | ".join(f"{mean:+.2f}" for mean in means) + f" | {extrapolated:+.2f} (upper bound) |")

    settings_path = REPO_ROOT / args.settings
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps({"methods": settings, "purity": {"residual": residual_purity, "raw": raw_purity, "base_rate": base_rate},
                                         "split_half": halves}, indent=1, default=float), encoding="utf-8")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[testbed] wrote {out_path} and {settings_path}")


if __name__ == "__main__":
    main()
