"""P4 of the 13 Oct research note: MEG-XL's frozen features for every row, gated on the testbed.

    PYTHONPATH=$HOME/megxllibs:src python -m pnpl2026.trial23.cli.megxl_features \\
        --megxl-repo <a clone of MEG-XL> --members trial13_g1_s0 ... --holdout-dir data/holdout

    1   MEG-XL frozen, loaded as its README loads it (`megxl.load_megxl`);
        every sensor's position, orientation and type from LibriBrain's
        fine-calibration file (`megxl.sensor_layout`)
    2   every row of every subject -- the testbed's 13 on ses-11 and ses-12
        (cli.testbed's sessions) and the holdout's 40 (cli.cross_graph's
        reader) -- one 3 s word window; 50 windows to a 150 s segment, in one
        seeded shuffled order; each row's 3 tokens averaged over channels;
        residualized on the timing design, as P1's features are
    3   the note's gate, on ses-12: neighbour purity against the testbed's
        own residual MCCA features (use it only if it beats them); and the
        four votes on these features, tuned on ses-11, tested on ses-12
    4   only when the gate passes (or --always-holdout): the holdout's
        features, in P2's layout, which cli.cross_graph reads --
        `--features data/trial23_megxl_holdout.npz --settings data/trial23_megxl_settings.json`

Writes data/trial23_megxl_settings.json and runs/trial23_megxl.md, and the
holdout features. GPU: pbs/trial23_megxl.pbs (the A100), about an hour.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...config import REPO_ROOT, RUNS_DIR
from .. import crossfeatures as cf
from .. import megxl
from ..data.dataset import load_manifest
from .cross_graph import load_holdout
from .row_encoder import DEFAULT_COMPONENTS, mean_purity, tune_votes
from .testbed import SESSIONS, prepare_session, residual_features

#MEG-XL's weights and LibriBrain's fine calibration, by their Hugging Face names
MEGXL_REPO_ID, MEGXL_FILE = "pnpl/MEG-XL", "meg-xl-med-v2.ckpt"
LIBRIBRAIN_REPO_ID, CALIBRATION_FILE = "pnpl/LibriBrain", "metadata/neo/sss_cal.dat"


def parse_args():
    """Command-line arguments for MEG-XL's features."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--megxl-repo", required=True, help="a clone of github.com/neural-processing-lab/MEG-XL")
    parser.add_argument("--checkpoint", default=None, help=f"default {MEGXL_REPO_ID}/{MEGXL_FILE} from the Hugging Face cache")
    parser.add_argument("--tokenizer", default=None, help="default <repo>/brainstorm/neuro_tokenizers/biocodec_ckpt.pt")
    parser.add_argument("--fine-calibration", default=None, help=f"default {LIBRIBRAIN_REPO_ID}/{CALIBRATION_FILE} from the cache")
    parser.add_argument("--channel-names-from", default="data/trial10_reference", help="the cache whose manifest orders the 306 channels")
    parser.add_argument("--members", nargs="+", required=True, help="the base mix's members, each with chains38 dumps")
    parser.add_argument("--holdout-dir", default="data/holdout")
    parser.add_argument("--matches", default="data/trial22_matches.json", help="repo-relative, from cli.repeat_check")
    parser.add_argument("--subjects", default="1-39")
    parser.add_argument("--map", default="data/trial23_testbed_map.npz", help="repo-relative, from cli.testbed_check")
    parser.add_argument("--reference-cache", default="data/trial10_reference", help="subject 0's cache")
    parser.add_argument("--others-cache", default="data/trial18_others", help="the other subjects' cache")
    parser.add_argument("--testbed-settings", default="data/trial23_testbed_settings.json", help="for the MCCA C and gains")
    parser.add_argument("--order", choices=("shuffled", "sentence"), default="shuffled", help="the rows' order inside a segment")
    parser.add_argument("--windows-per-segment", type=int, default=megxl.WINDOWS_PER_SEGMENT,
                        help="3 s word windows per forward pass: 50 is MEG-XL's 150 s context (10, its LibriBrain fine-tuning length)")
    parser.add_argument("--bf16", action="store_true", help="bf16 autocast on the GPU; default fp32, as MEG-XL's README runs it")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--always-holdout", action="store_true", help="the holdout's features even when the gate fails")
    parser.add_argument("--device", default=None)
    parser.add_argument("--features-out", default="data/trial23_megxl_holdout.npz", help="repo-relative")
    parser.add_argument("--settings-out", default="data/trial23_megxl_settings.json", help="repo-relative")
    parser.add_argument("--report", default=None, help="default runs/trial23_megxl.md")
    return parser.parse_args()


def cached_file(path, repo_id, filename, repo_type=None):
    """A file given on the command line, or the Hugging Face cache's copy.

    ARGS:
        path (str or None), repo_id (str), filename (str), repo_type (str or None)

    RETURNS:
        Path
    """
    if path:
        return Path(path)
    from huggingface_hub import hf_hub_download

    #a compute node is offline: the login node must have fetched it first
    return Path(hf_hub_download(repo_id, filename, repo_type=repo_type))


def residual_rows(model, preprocess, pieces, piece_of_row, onsets, groups, onsets_s, layout, device, args):
    """One subject's MEG-XL row features, residualized on the rows' timing design.

    ARGS:
        model, preprocess: from `megxl.load_megxl`
        pieces (list), piece_of_row, onsets: the subject's rows
        groups, onsets_s (np.ndarray): the rows' sentences and onsets, for the design
        layout (tuple), device, args

    RETURNS:
        [n_rows, 3 * latent] float32
    """
    features = megxl.subject_features(model, preprocess, pieces, piece_of_row, onsets, layout, device, args.seed, args.order == "shuffled",
                                      args.windows_per_segment, args.bf16)
    projector = cf.residual_projector(cf.timing_design(groups, onsets_s))
    return cf.residualize(features.reshape(len(features), -1).astype(np.float64), projector).astype(np.float32)


def main():
    """MEG-XL, the testbed's features and gate, then the holdout's features."""
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    repo = Path(args.megxl_repo)
    testbed_settings_path = REPO_ROOT / args.testbed_settings
    testbed_settings = json.loads(testbed_settings_path.read_text(encoding="utf-8")) if testbed_settings_path.exists() else None
    components = testbed_settings["methods"]["graph"]["components"] if testbed_settings else DEFAULT_COMPONENTS

    #1. the model and the sensors, in the data's channel order
    checkpoint = cached_file(args.checkpoint, MEGXL_REPO_ID, MEGXL_FILE)
    tokenizer = Path(args.tokenizer) if args.tokenizer else repo / "brainstorm" / "neuro_tokenizers" / "biocodec_ckpt.pt"
    calibration = cached_file(args.fine_calibration, LIBRIBRAIN_REPO_ID, CALIBRATION_FILE, repo_type="dataset")
    xyzdir, types = megxl.sensor_layout(calibration, load_manifest(REPO_ROOT / args.channel_names_from)["channel_names"])
    layout = (megxl.normalise_positions(xyzdir), types)
    model, preprocess, diagnostics = megxl.load_megxl(repo, checkpoint, tokenizer, device)
    print(f"[megxl] {checkpoint.name} on {device}: {diagnostics['loaded_keys']} tensors, {len(diagnostics['missing_keys'])} missing, "
          f"{len(diagnostics['unexpected_keys'])} unexpected; {int((types == 1).sum())} magnetometers, {int((types == 0).sum())} gradiometers")

    #2. the testbed: every subject of both sessions, row by row
    mapping = np.load(REPO_ROOT / args.map)
    sessions = {session: prepare_session(args.members, session, mapping, args) for session in SESSIONS}
    features = {}
    for session, prepared in sessions.items():
        rows = prepared["rows"]
        features[session] = np.stack([residual_rows(model, preprocess, pieces, prepared["piece_of_row"], onsets, rows["groups"],
                                                    rows["onsets"] / cf.RATE, layout, device, args)
                                      for pieces, onsets in zip(prepared["pieces"], prepared["onsets"])])
        print(f"[megxl] testbed ses-{session}: {features[session].shape[0]} subjects x {features[session].shape[1]} rows")

    #3. the gate on ses-12, against P1's residual MCCA features on the same rows
    test = sessions["12"]
    megxl_purity, base_rate = mean_purity(list(features["12"][1:]), test["rows"])
    residual, raw, _ = residual_features(test["pieces"], test["onsets"], test["piece_of_row"], test["rows"], components)
    residual_purity, _ = mean_purity([subject.reshape(len(subject), -1) for subject in residual[1:]], test["rows"])
    raw_purity, _ = mean_purity([subject.reshape(len(subject), -1) for subject in raw[1:]], test["rows"])
    passes = bool(megxl_purity > residual_purity)
    method_settings = tune_votes(features["11"], sessions["11"], features["12"], test)

    #4. the holdout's features, when the gate passes or when asked
    holdout_note = "not computed: the gate failed (--always-holdout computes them anyway)"
    if passes or args.always_holdout:
        holdout = load_holdout(args.holdout_dir, REPO_ROOT / args.matches, args.subjects)
        holdout_features = np.stack([residual_rows(model, preprocess, pieces, holdout["piece_of_row"], holdout["onset_in_piece"],
                                                   holdout["groups"], holdout["onsets_s"], layout, device, args)
                                     for pieces in holdout["pieces"]])
        features_path = REPO_ROOT / args.features_out
        features_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(features_path, embeddings=holdout_features, subjects=np.array([0] + holdout["others"]),
                 rows=np.flatnonzero(holdout["voting"]))
        holdout_note = f"{holdout_features.shape[0]} subjects x {holdout_features.shape[1]} rows -> `{args.features_out}`"

    #the settings cross_graph reads, and the report
    settings = {"features": "megxl", "passes": passes, "methods": method_settings, "order": args.order, "components": components,
                "purity": {"megxl": megxl_purity, "residual": residual_purity, "raw": raw_purity, "base_rate": base_rate},
                "load": {"missing": len(diagnostics["missing_keys"]), "unexpected": len(diagnostics["unexpected_keys"])}}
    settings_path = REPO_ROOT / args.settings_out
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(settings, indent=1, default=float) + "\n", encoding="utf-8")
    mcca_gains = testbed_settings["methods"] if testbed_settings else {}
    lines = ["# Trial 23 -- P4: MEG-XL as a frozen feature extractor", "",
             f"- model: `{checkpoint.name}`; {diagnostics['loaded_keys']} tensors loaded, {len(diagnostics['missing_keys'])} missing, "
             f"{len(diagnostics['unexpected_keys'])} unexpected (RoPE tables are rebuilt, not loaded); missing: "
             + (", ".join(f"`{key}`" for key in diagnostics["missing_keys"][:6]) or "none")
             + (" -- anything beyond the RoPE table means the weights did not load" if diagnostics["missing_keys"] else ""),
             f"- rows as 3 s word windows, {args.windows_per_segment} to a segment ({3 * args.windows_per_segment} s), order: {args.order}; "
             "features: 3 tokens x latent, averaged over channels, "
             f"residualized; dimension {features['12'].shape[2]}",
             f"- holdout: {holdout_note}", "",
             "## The gate: neighbour purity on ses-12 (top-10, other chains only)", "",
             f"- MEG-XL features: {megxl_purity:.3f}", f"- residual MCCA features (P1's, C = {components}): {residual_purity:.3f}",
             f"- raw MCCA features: {raw_purity:.3f}", f"- base rate: {base_rate:.3f}", "",
             f"Verdict: {'PASSED -- an alternative feature set for P1/P3' if passes else 'not passed -- the note says do not use it'}.", "",
             "## The votes on MEG-XL features: tuned on ses-11, tested on ses-12", "",
             "| method | setting | weight | ses-11 gain | ses-12 gain | MCCA features' ses-12 gain |", "|---|---|---|---|---|---|"]
    for method, chosen in method_settings.items():
        mcca = f"{mcca_gains[method]['test_gain']:+.2f}" if method in mcca_gains else "-"
        lines.append(f"| {method} | {json.dumps(chosen['params'])} | {chosen['weight']} | {chosen['tune_gain']:+.2f} | "
                     f"{chosen['test_gain']:+.2f} | {mcca} |")
    report = Path(args.report) if args.report else RUNS_DIR / "trial23_megxl.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[megxl] wrote {settings_path} and {report}")


if __name__ == "__main__":
    main()
