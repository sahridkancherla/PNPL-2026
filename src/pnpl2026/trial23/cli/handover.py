"""Hand the final picks to the organisers: pack every checkpoint they need, rebuild any pick from them.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.handover pack [--out handover/pnpl2026_deep]
    PYTHONPATH=src python -m pnpl2026.trial23.cli.handover reproduce --pick robustpn [--reuse-member-csvs]

The organisers verify top teams with runnable checkpoints and code. A final
pick is not one network: it is several trained members, each predicting the
holdout, combined by a fixed pipeline with no random draw (trial 23's
`pbs/trial23_cpu.pbs` robust step):

    1. every member predicts the holdout (`cli.predict`, the fixed training
       scaler `data/trial23_training_scaler.npz`)
    2. each "mix": the members' CSVs transduced together (`cli.transduce` v1:
       TransCLIP-style clustering of the 868 sentence rows, each member's
       temperature and prototypes from its validation dump) at one pull,
       neighbour count and blend
    3. the pick's mixes averaged (`cli.ensemble`), one vote each
    4. the holdout hub correction (`cli.hubnorm_holdout`, zscore)
    5. the lone-word rows from the I3 specialist's three seeds (`cli.lone_fill`)

pack       on CX3 at the repo root: copies into --out every member's
           best.ckpt, config.yaml and evaluation.json, the validation dumps
           steps 2 and 5 read, the training scaler, every member CSV and every
           submitted file (references for the rebuild's comparison), and the
           code at HEAD (`git archive`); writes recipes.json (each pick's
           mixes), HANDOVER.md and MANIFEST.sha256. Fails, listing every
           missing file, unless --allow-missing
reproduce  from the code's root with the package unpacked there and the
           holdout under data/holdout: runs steps 1-5 for --pick into
           --work-dir and compares the result with the submitted file (largest
           difference, top-1 and top-10 agreement). --reuse-member-csvs skips
           step 1 and starts from the shipped member CSVs (CPU, minutes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import REPO_ROOT
from ..local_checks import find_dump

#the 23 members every final pick starts from: configs/trial23/members.txt without trial 21's B
BASE = (
    "trial13_g1_s0", "trial13_g1_s1", "trial13_g1_s2", "trial13_g2_s0", "trial13_g2_s1", "trial13_g2_s2",
    "trial16_kn_s0", "trial16_kn_s1", "trial16_kn_s2", "trial18_k2_s0", "trial18_k2_s1", "trial18_k2_s2",
    "trial18_n_s0", "trial18_n_s1", "trial18_n_s2", "trial10_f1_s0", "trial12_w1024_s0",
    "trial18_k18_s0", "trial18_k18_s1", "trial18_k18_s2", "trial19_spw_s0", "trial19_spw_s1", "trial19_spw_s2",
)

#each block's fourth mix reads SPW's other three seeds in place of trial 19's
SPW_SWAP = {"trial19_spw_s0": "trial20_spw_s3", "trial19_spw_s1": "trial20_spw_s4", "trial19_spw_s2": "trial20_spw_s5"}

#the lone-word specialist whose three seeds fill every pick's lone rows
LONE = ("trial15_i3_s0", "trial15_i3_s1", "trial15_i3_s2")

#members only some picks add
SSL = ("trial18_sslg2_s0", "trial18_sslg2_s1", "trial18_sslg2_s2")
HYBRID_B = ("trial21_b_s0", "trial21_b_s1", "trial21_b_s2")

#the transduction's share floor, the same in every mix
SHARE_FLOOR = 0.005

#which trainer made each family's checkpoints; trial 12 ran trial 10's code
TRAINERS = {"trial10": "trial10", "trial12": "trial10", "trial13": "trial13", "trial15": "trial15", "trial16": "trial16",
            "trial18": "trial18", "trial19": "trial19", "trial20": "trial20", "trial21": "trial21"}


def block(extra=(), pull=1.0, neighbours=3, blends=(0.65, 0.75, 0.9), swap=True):
    """The robust step's mixes: one per blend, and the 0.75 blend again with SPW's other seeds.

    ARGS:
        extra (tuple): members added to BASE, in order
        pull (float), neighbours (int): the transduction's settings
        blends (tuple): one mix each
        swap (bool): add the fourth mix with SPW_SWAP applied

    RETURNS:
        list of dict: members, pull, neighbours, blend
    """
    members = list(BASE) + list(extra)
    mixes = [dict(members=members, pull=pull, neighbours=neighbours, blend=blend) for blend in blends]
    #the fourth mix: trial 20's SPW seeds where trial 19's stood
    if swap:
        mixes.append(dict(members=[SPW_SWAP.get(member, member) for member in members], pull=pull,
                          neighbours=neighbours, blend=0.75))
    return mixes


def built_in_recipes():
    """The ten final picks of 16 Oct 2026, as they were built.

    RETURNS:
        dict: lone, share_floor, and picks -- each with its submitted file,
        public score, mixes, and whether the lone rows are filled
    """
    robust = block()
    pull2nb6 = block(pull=2.0, neighbours=6)
    picks = {
        "robustpn": (robust + pull2nb6, "ens_t23_robustpn_robust_ts_hubh_lone.csv", 71.432, True),
        "sslpn": (block(extra=SSL) + pull2nb6, "ens_t23_sslpn_robust_ts_hubh_lone.csv", 71.032, True),
        "robustpn3": (robust + pull2nb6 + block(pull=1.5, neighbours=5),
                      "ens_t23_robustpn3_robust_ts_hubh_lone.csv", 71.032, True),
        "robust": (robust, "ens_t23_robust_ts_hubh_lone.csv", 70.849, True),
        #trial 22's B mix: the 23 and B's three seeds, transduced once at blend 0.75
        "robustb": (robust + block(extra=HYBRID_B, blends=(0.75,), swap=False),
                    "ens_t23_robustb_ts_hubh_lone.csv", 70.849, True),
        "f0fixed": (block(extra=("trial18_f0fixed_s0",)), "ens_t23_f0fixed_robust_ts_hubh_lone.csv", 70.849, True),
        "sslg2": (block(extra=SSL), "ens_t23_sslg2_robust_ts_hubh_lone.csv", 70.849, True),
        #submitted before its lone rows were filled
        "pull2nb6": (pull2nb6, "ens_t23_pull2nb6_robust_ts_hubh.csv", 70.842, False),
        "blendavg": (block(swap=False), "ens_t23_blendavg_ts_hubh_lone.csv", 70.682, True),
        "tc1b75": (block(blends=(0.75,), swap=False), "ens_t22_tc1b75_ts_hubh_lone.csv", 70.682, True),
    }
    return {
        "lone": list(LONE),
        "share_floor": SHARE_FLOOR,
        "picks": {name: {"mixes": mixes, "submitted": submitted, "public": public, "lone_fill": lone}
                  for name, (mixes, submitted, public, lone) in picks.items()},
    }


def parse_args():
    """Command-line arguments for both subcommands."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    pack = commands.add_parser("pack", help="copy every file the final picks need into one folder")
    pack.add_argument("--root", default=None, help="where runs/, submissions/ and data/ live; default the repo")
    pack.add_argument("--out", default="handover/pnpl2026_deep", help="the package folder, repo-relative")
    pack.add_argument("--recipes", default=None, help="a recipes JSON instead of the built-in final picks")
    pack.add_argument("--allow-missing", action="store_true", help="write the package even with files missing")
    pack.add_argument("--force", action="store_true", help="write into an existing --out")
    pack.add_argument("--allow-uncommitted", action="store_true",
                      help="tests only: pack although this tool is not committed (HANDOVER.md says so)")

    rebuild = commands.add_parser("reproduce", help="rebuild one pick from the packed checkpoints")
    rebuild.add_argument("--pick", required=True)
    rebuild.add_argument("--root", default=".", help="the code's root, with the package unpacked there")
    rebuild.add_argument("--recipes", default="recipes.json", help="relative to --root")
    rebuild.add_argument("--holdout-dir", default="data/holdout", help="holds COMPETITION_HOLDOUT/; relative to --root")
    rebuild.add_argument("--work-dir", default=None, help="default handover_rebuild/<pick>, relative to --root")
    rebuild.add_argument("--reuse-member-csvs", action="store_true", help="start from the shipped member CSVs")
    rebuild.add_argument("--device", default=None, help="for cli.predict; default cuda when available")
    return parser.parse_args()


# -- pack ---------------------------------------------------------------


def pick_members(pick):
    """Every member a pick's mixes read, in first-use order.

    ARGS:
        pick (dict): one entry of recipes["picks"]

    RETURNS:
        list of str
    """
    return list(dict.fromkeys(member for mix in pick["mixes"] for member in mix["members"]))


def isolated_dump(run_dir):
    """The isolated validation dump lone_fill's hub bank reads, or None.

    ARGS:
        run_dir (Path)

    RETURNS:
        Path or None
    """
    #cli.lone_check's read_scored looks in dumps/, then embeddings/
    for folder in ("dumps", "embeddings"):
        path = run_dir / folder / "isolated_validation.npz"
        if path.exists():
            return path
    return None


def sha256(path):
    """A file's SHA-256, read in chunks.

    ARGS:
        path (Path)

    RETURNS:
        str: hex digest
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_copies(recipes, root):
    """Every file the package carries: (source, destination relative to the package, required).

    ARGS:
        recipes (dict)
        root (Path): where runs/, submissions/ and data/ live

    RETURNS:
        list of (Path or None, str, bool) -- source None when it was not found
    """
    mixed = list(dict.fromkeys(member for pick in recipes["picks"].values() for member in pick_members(pick)))
    copies = []
    for member in mixed + [run for run in recipes["lone"] if run not in mixed]:
        run_dir = root / "runs" / member
        copies.append((run_dir / "best.ckpt", f"runs/{member}/best.ckpt", True))
        for name in ("config.yaml", "evaluation.json"):
            copies.append((run_dir / name, f"runs/{member}/{name}", False))

        #the transduction's tau and prototypes come from the reference validation dump
        if member in mixed:
            found = find_dump(str(run_dir), "reference", "validation", runs_dir=root / "runs")
            copies.append((found, f"runs/{member}/dumps/reference_validation.npz", True))
        #the lone rows' hub bank comes from the specialist's isolated validation dump
        if member in recipes["lone"]:
            copies.append((isolated_dump(run_dir), f"runs/{member}/dumps/isolated_validation.npz", True))
        copies.append((root / "submissions" / f"{member}_in_context_ts.csv", f"submissions/{member}_in_context_ts.csv", False))

    #the submitted files, the fixed scaler, and the SSL members' pretrained encoder
    for pick in recipes["picks"].values():
        copies.append((root / "submissions" / pick["submitted"], f"submissions/{pick['submitted']}", False))
    copies.append((root / "data" / "trial23_training_scaler.npz", "data/trial23_training_scaler.npz", True))
    for name in ("encoder.pt", "pretrain.json"):
        copies.append((root / "runs" / "trial23_ssl" / name, f"runs/trial23_ssl/{name}", False))
    return [(source if source is not None and Path(source).exists() else None, destination, required)
            for source, destination, required in copies]


def git_state(allow_uncommitted=False):
    """The code's commit, and whether the code in the working tree differs from it.

    ⚠️ `git archive HEAD` carries committed files only: this very tool must be
    committed, or the package would lack the command that rebuilds a pick.

    ARGS:
        allow_uncommitted (bool): warn instead of stopping when this tool is not committed

    RETURNS:
        commit (str), dirty (bool) -- dirty counts untracked files under src/, configs/ and pbs/
    """
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", "src/pnpl2026/trial23/cli/handover.py"], cwd=REPO_ROOT,
                             capture_output=True, text=True)
    if tracked.returncode != 0 and allow_uncommitted:
        print("[pack] WARNING: handover.py is not committed; code.tar.gz will not carry it")
    elif tracked.returncode != 0:
        raise SystemExit("src/pnpl2026/trial23/cli/handover.py is not committed: commit it (and pull on CX3) first, "
                         "or code.tar.gz will lack the rebuild command.")
    status = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "pbs"], cwd=REPO_ROOT,
                            capture_output=True, text=True, check=True)
    return commit.stdout.strip(), bool(status.stdout.strip())


def trainer_of(member):
    """The trainer module that made a member's checkpoint.

    ARGS:
        member (str): e.g. "trial18_k2_s0"

    RETURNS:
        str
    """
    trial = member.split("_")[0]
    return f"pnpl2026.{TRAINERS.get(trial, trial)}.cli.train"


def handover_text(recipes, commit, dirty):
    """HANDOVER.md: what a pick is, the picks, the members, how to rebuild one.

    ARGS:
        recipes (dict)
        commit (str)
        dirty (bool)

    RETURNS:
        str, Markdown
    """
    lines = [
        "# PNPL 2026 deep track -- the final picks as runnable checkpoints",
        "",
        f"Code: commit `{commit}`, archived as `code.tar.gz`." + (" WARNING: the working tree had uncommitted changes "
                                                                    "when this package was written." if dirty else ""),
        "",
        "## What a pick is",
        "",
        "Not one network: several trained members, each predicting the holdout, combined by a fixed pipeline",
        "with no random draw -- (1) every member predicts the holdout with the fixed training scaler",
        "(`cli.predict`); (2) per mix, the members' predictions are transduced together over the 868 sentence",
        "rows (`cli.transduce`: one Gaussian cluster per word started at the word's validation prototype, a",
        "graph of nearest rows in other sentences, blended into each member's own scores); (3) the mixes are",
        "averaged (`cli.ensemble`); (4) the holdout hub correction (`cli.hubnorm_holdout`, zscore); (5) the",
        "lone-word rows come from the I3 specialist's three seeds (`cli.lone_fill`). Every module is",
        "`pnpl2026.trial23.cli.<name>` in the archived code.",
        "",
        "## The picks",
        "",
        "| pick | submitted file | public BAcc@10 | mixes | mixed members | lone rows |",
        "|---|---|---|---|---|---|",
    ]
    for name, pick in recipes["picks"].items():
        lines.append(f"| {name} | `{pick['submitted']}` | {pick['public']:.3f} | {len(pick['mixes'])} | "
                     f"{len(pick_members(pick))} | {'I3' if pick['lone_fill'] else 'the mix' } |")
    lines += ["", "Each pick's mixes (members, pull, neighbours, blend) are in `recipes.json`.", "",
              "## The members", "",
              "Each run's exact merged training config is `runs/<run>/config.yaml`; its local scores are",
              "`runs/<run>/evaluation.json`. The job scripts are `pbs/<trial>.pbs`, with the run lines in",
              "`configs/<trial>/runs.txt`. The SSL members (`trial18_sslg2_*`) were fine-tuned from",
              "`runs/trial23_ssl/encoder.pt` (`pnpl2026.trial23.cli.pretrain_encoder`, `pbs/trial23_ssl.pbs`).", "",
              "| run | trainer |", "|---|---|"]
    members = list(dict.fromkeys(member for pick in recipes["picks"].values() for member in pick_members(pick)))
    for member in members + [run for run in recipes["lone"] if run not in members]:
        lines.append(f"| {member} | `{trainer_of(member)}` |")
    lines += [
        "", "## Rebuild a pick", "",
        "```bash",
        "mkdir pnpl2026 && tar -xzf code.tar.gz -C pnpl2026      # the code at the commit above",
        "cp -r runs submissions data recipes.json pnpl2026/      # this package, at the code's root",
        "cd pnpl2026 && uv sync --frozen                          # the locked environment (uv.lock)",
        "# the competition holdout files under data/holdout/COMPETITION_HOLDOUT/",
        "PYTHONPATH=src uv run python -m pnpl2026.trial23.cli.handover reproduce --pick robustpn",
        "```",
        "",
        "It predicts every member of the pick (a GPU takes minutes per member; `--device cpu` works, slowly),",
        "runs the mixes, the average, the hub correction and the lone rows into `handover_rebuild/<pick>/`, and",
        "compares the result with the submitted file: largest difference, top-1 and top-10 agreement.",
        "`--reuse-member-csvs` starts from the shipped member CSVs instead and takes minutes on a CPU.",
        "",
        "## Holdout use",
        "",
        "Steps 2 and 4 fit on the holdout's own rows, unsupervised; no step reads a label. The organisers",
        "confirmed on 12 Oct 2026 that the holdout data may be used in any way before a submission is",
        "generated (relayed to the team that day; recorded in `handoff-docs/route-to-73-2026-10-12.md`).",
        "",
        "`MANIFEST.sha256` lists every file of this package with its SHA-256.",
    ]
    return "\n".join(lines) + "\n"


def pack(args):
    """Copy every file the picks need into --out, with the code, recipes, README and manifest."""
    root = Path(args.root) if args.root else REPO_ROOT
    out = REPO_ROOT / args.out
    recipes = json.loads(Path(args.recipes).read_text(encoding="utf-8")) if args.recipes else built_in_recipes()
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to write into it.")

    #every required file found before anything is copied
    copies = plan_copies(recipes, root)
    missing = [destination for source, destination, required in copies if source is None and required]
    optional = [destination for source, destination, required in copies if source is None and not required]
    for destination in optional:
        print(f"[pack] optional, not found: {destination}")
    if missing:
        print("\n".join(f"[pack] REQUIRED, not found: {destination}" for destination in missing))
        if not args.allow_missing:
            raise SystemExit(f"{len(missing)} required file(s) missing; nothing written (--allow-missing to go on).")

    out.mkdir(parents=True, exist_ok=True)
    for source, destination, _ in copies:
        if source is None:
            continue
        target = out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    #the code at HEAD, and the documents that describe the package
    commit, dirty = git_state(args.allow_uncommitted)
    subprocess.run(["git", "archive", "--format=tar.gz", "-o", str(out / "code.tar.gz"), "HEAD"], cwd=REPO_ROOT, check=True)
    (out / "recipes.json").write_text(json.dumps(recipes, indent=2), encoding="utf-8")
    (out / "HANDOVER.md").write_text(handover_text(recipes, commit, dirty), encoding="utf-8")

    #the manifest last, over everything above
    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "MANIFEST.sha256")
    (out / "MANIFEST.sha256").write_text("".join(f"{sha256(path)}  {path.relative_to(out).as_posix()}\n" for path in files),
                                         encoding="utf-8")
    size = sum(path.stat().st_size for path in files)
    n_checkpoints = sum(1 for path in files if path.name == "best.ckpt")
    print(f"[pack] {out}: {len(files)} files, {n_checkpoints} checkpoints, {size / 1e9:.2f} GB; code {commit[:10]}"
          + (" (WARNING: the working tree had uncommitted changes)" if dirty else ""))


# -- reproduce ----------------------------------------------------------


def run_tool(name, *arguments):
    """Run one of trial 23's command-line tools beside this one, failing loudly.

    ARGS:
        name (str): e.g. "transduce"
        *arguments: its command line
    """
    #the package this module was started from: pnpl2026.trial23.cli or src.pnpl2026.trial23.cli
    package = __spec__.name.rsplit(".", 1)[0]
    command = [sys.executable, "-m", f"{package}.{name}", *[str(argument) for argument in arguments]]
    print(f"[reproduce] {name} " + " ".join(str(argument) for argument in arguments[:6]) + " ...")
    subprocess.run(command, check=True)


def compare(rebuilt, submitted):
    """How far a rebuilt CSV is from the submitted one.

    ARGS:
        rebuilt (Path), submitted (Path)

    RETURNS:
        dict: rows, largest difference, top-1 and top-10 agreement, verdict
    """
    first, second = pd.read_csv(rebuilt), pd.read_csv(submitted)
    if list(first.columns) != list(second.columns) or len(first) != len(second):
        return {"verdict": "different shape", "rows": [len(first), len(second)]}
    if not np.array_equal(first.iloc[:, 0].to_numpy(), second.iloc[:, 0].to_numpy()):
        return {"verdict": "different row order"}

    #the 50 primary columns rank the scored words; every value column is compared
    values_a, values_b = first.iloc[:, 1:].to_numpy(dtype=np.float64), second.iloc[:, 1:].to_numpy(dtype=np.float64)
    primary_a, primary_b = values_a[:, :50], values_b[:, :50]
    top10_a, top10_b = np.argsort(-primary_a, axis=1)[:, :10], np.argsort(-primary_b, axis=1)[:, :10]
    same_sets = np.array([set(a) == set(b) for a, b in zip(top10_a, top10_b)])
    largest = float(np.abs(values_a - values_b).max())
    record = {
        "rows": int(len(first)),
        "largest_difference": largest,
        "top1_agreement": float(np.mean(primary_a.argmax(axis=1) == primary_b.argmax(axis=1))),
        "top10_identical_rows": float(same_sets.mean()),
    }
    #identical to print precision, or the same rankings, or neither
    record["verdict"] = ("identical" if largest < 1e-6 else
                         "same top-10 on every row" if same_sets.all() else "differs")
    return record


def reproduce(args):
    """Rebuild one pick from its members' checkpoints, then compare it with the submitted file."""
    root = Path(args.root).resolve()
    recipes = json.loads((root / args.recipes).read_text(encoding="utf-8"))
    if args.pick not in recipes["picks"]:
        raise SystemExit(f"Unknown pick {args.pick!r}; the picks are {sorted(recipes['picks'])}.")
    pick = recipes["picks"][args.pick]
    work = root / (args.work_dir or f"handover_rebuild/{args.pick}")
    (work / "members").mkdir(parents=True, exist_ok=True)
    holdout = root / args.holdout_dir
    scaler = root / "data" / "trial23_training_scaler.npz"

    #1. every member's holdout CSV: predicted afresh, or the shipped one
    members = pick_members(pick) + (list(recipes["lone"]) if pick["lone_fill"] else [])
    csv_of, member_checks = {}, {}
    for member in members:
        shipped = root / "submissions" / f"{member}_in_context_ts.csv"
        if args.reuse_member_csvs:
            if not shipped.exists():
                raise SystemExit(f"--reuse-member-csvs: {shipped} is not in the package.")
            csv_of[member] = shipped
            continue
        target = work / "members" / f"{member}_in_context_ts.csv"
        device = ["--device", args.device] if args.device else []
        run_tool("predict", "--ckpt", root / "runs" / member / "best.ckpt", "--holdout-dir", holdout,
                 "--scaler-file", scaler, "--out", target, *device)
        csv_of[member] = target
        #how the fresh prediction compares with the one the pick was built from
        if shipped.exists():
            member_checks[member] = compare(target, shipped)

    #2. each mix transduced, as the robust step ran it
    mixes = []
    for number, mix in enumerate(pick["mixes"]):
        out = work / f"mix{number:02d}.csv"
        run_tool("transduce", "--inputs", *[csv_of[member] for member in mix["members"]],
                 "--runs", *[root / "runs" / member for member in mix["members"]],
                 "--pull", mix["pull"], "--neighbours", mix["neighbours"], "--share-floor", recipes["share_floor"],
                 "--blend", mix["blend"], "--check", 0, "--holdout-dir", holdout, "--force", "--out", out,
                 "--report", work / f"mix{number:02d}.md")
        mixes.append(out)

    #3. the mixes averaged, one vote each; a single mix is the pick itself
    combined = work / f"{args.pick}.csv"
    if len(mixes) > 1:
        run_tool("ensemble", "--inputs", *mixes, "--out", combined, "--force")
    else:
        shutil.copy2(mixes[0], combined)

    #4. the hub correction fitted on the holdout's rows; 5. the lone rows
    run_tool("hubnorm_holdout", "--csv", combined, "--method", "zscore", "--holdout-dir", holdout, "--force",
             "--report", work / "hubnorm.md")
    final = work / f"{args.pick}_hubh.csv"
    if pick["lone_fill"]:
        run_tool("lone_fill", "--csv", final, "--holdout-dir", holdout, "--force",
                 "--lone-csvs", *[csv_of[member] for member in recipes["lone"]],
                 "--lone-runs", *[root / "runs" / member for member in recipes["lone"]])
        final = work / f"{args.pick}_hubh_lone.csv"

    #the rebuilt file against the submitted one
    submitted = root / "submissions" / pick["submitted"]
    result = {"pick": args.pick, "rebuilt": str(final), "submitted": str(submitted), "members": len(members),
              "mixes": len(mixes), "member_csvs": "shipped" if args.reuse_member_csvs else "predicted",
              "comparison": compare(final, submitted) if submitted.exists() else {"verdict": "no submitted file"},
              "members_against_shipped": member_checks}
    (work / "reproduce.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    comparison = result["comparison"]
    print(f"[reproduce] {args.pick}: {final}")
    print(f"[reproduce] against {pick['submitted']}: {comparison['verdict']}"
          + (f" -- largest difference {comparison['largest_difference']:.2e}, top-1 agreement "
             f"{100 * comparison['top1_agreement']:.2f}%, identical top-10 on {100 * comparison['top10_identical_rows']:.2f}% "
             f"of rows" if "largest_difference" in comparison else ""))


def main():
    """Pack the picks, or rebuild one."""
    args = parse_args()
    if args.command == "pack":
        pack(args)
    else:
        reproduce(args)


if __name__ == "__main__":
    main()
