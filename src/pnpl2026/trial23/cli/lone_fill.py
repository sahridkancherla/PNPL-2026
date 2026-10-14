"""Fill a submission's lone-word rows with the lone-word specialist, hub-corrected.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.lone_fill \\
        --csv submissions/ens_g1g2_f1w1024_ts_hub.csv \\
        --lone-csvs submissions/trial15_i3_s0_in_context_ts.csv ... \\
        --lone-runs runs/trial15_i3_s0 runs/trial15_i3_s1 runs/trial15_i3_s2 \\
        --holdout-dir data/holdout

The 92 lone words are scored on the private board only. `cli.lone_check`
compared every recipe for them locally, in their exact format (2026-10-08):
the I3 specialist's 3-seed ensemble, hub-corrected, was chosen on validation
(test 44.9, against 27.2 for the 66.39 mix). This writes that recipe:

    sentence rows   the input CSV's, byte for byte in value -- the public board
                    sees no change
    lone rows       the `--lone-csvs` ensemble (rows z-scored, averaged,
                    z-scored again, as cli.ensemble), then the frozen hub
                    correction: each word centred and scaled by its mean and
                    spread over the specialist's own VALIDATION lone words
                    (`--lone-runs` dumps, isolated protocol; labels never
                    used). `--no-hub` skips the correction
    Moses           the input CSV's, renormalised

`--lone-group CSVS=RUNS` (repeatable, instead of --lone-csvs/--lone-runs)
builds the lone rows exactly as cli.lone_check mixes its groups: each group
ensembled on its own, the groups averaged at EQUAL weight, and the hub
correction fitted on that same mix of the groups' validation lone words.

Writes `<csv stem>_lone.csv`. Each row is handled on its own, its source taken
from the holdout loader's metadata: within the holdout principle.

⚠️ ROW ORDERING IS THE WHOLE CONTRACT: every CSV must carry the canonical index
0..N-1 in order; the output keeps it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..hubnorm import apply_correction, fit_zscore
from .ensemble import describe, holdout_sources, read_submission, softmax_rows, standardise_rows, zscore_scores
from .lone_check import group_scores


def parse_args():
    """Command-line arguments for filling the lone rows."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="the submission whose lone rows are replaced")
    parser.add_argument("--lone-csvs", nargs="+", default=None, help="the specialist's compliant holdout CSVs")
    parser.add_argument("--lone-group", action="append", default=None,
                        help="trial 23: CSV,CSV,...=RUN,RUN,... -- one group of cli.lone_check's mix; give it once per group")
    parser.add_argument("--lone-runs", nargs="+", default=None, help="the specialist's run directories, for the hub bank")
    parser.add_argument("--protocol", default="isolated", help="the dumps the hub bank comes from")
    parser.add_argument("--no-hub", action="store_true", help="the specialist ensemble without the hub correction")
    parser.add_argument("--holdout-dir", default=None, help="folder holding COMPETITION_HOLDOUT/")
    parser.add_argument("--track", default="deep", help="competition track")
    parser.add_argument("--out", default=None, help="default <csv stem>_lone.csv beside the input")
    parser.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    return parser.parse_args()


def main():
    """Read, rebuild the lone rows, check, write."""
    args = parse_args()
    from pnpl.competition import load_vocabulary, write_submission

    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path.with_name(f"{in_path.stem}_lone.csv")
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} exists; pass --force to overwrite it.")
    if bool(args.lone_group) == bool(args.lone_csvs):
        raise SystemExit("Give either --lone-csvs (one specialist) or --lone-group (lone_check's groups), not both.")
    if args.lone_csvs and not args.no_hub and not args.lone_runs:
        raise SystemExit("The hub correction needs --lone-runs (the specialist's dumps); or pass --no-hub.")

    #each group's CSVs and runs; a plain specialist is one group
    groups = []
    for spec in args.lone_group or []:
        csvs, _, runs = spec.partition("=")
        groups.append(([path for path in csvs.split(",") if path], [path for path in runs.split(",") if path]))
    if args.lone_csvs:
        groups.append((args.lone_csvs, args.lone_runs or []))

    #the base and every specialist CSV, all on the canonical rows
    primary_words, moses_words = load_vocabulary("primary"), load_vocabulary("moses")
    index, primary, moses = read_submission(in_path, primary_words, moses_words)
    if not np.array_equal(index, np.arange(len(index))):
        raise SystemExit(f"{in_path}: the index column is not 0..{len(index) - 1} in order.")
    group_tables = []
    for csvs, _ in groups:
        specialists = []
        for path in csvs:
            other_index, other_primary, _ = read_submission(Path(path), primary_words, moses_words)
            if not np.array_equal(other_index, index):
                raise SystemExit(f"{path}: its index column differs from {in_path}'s.")
            specialists.append(standardise_rows(other_primary))
        #each group as cli.ensemble builds it; axis 0 is the file axis
        group_tables.append(zscore_scores(np.mean(specialists, axis=0)))

    #each row's source, item by item, from the loader's own metadata
    sources = holdout_sources(args.track, args.holdout_dir, len(index))
    lone = sources == "word"

    #the groups at equal weight, as cli.lone_check mixes them; one group is its own ensemble
    lone_scores = zscore_scores(np.mean(group_tables, axis=0))[lone]

    #the frozen hub correction, fitted on the same mix of the groups' validation lone words
    if not args.no_hub:
        banks, reference = [], None
        for _, runs in groups:
            if not runs:
                raise SystemExit("Every group needs its runs for the hub bank; or pass --no-hub.")
            bank, reference = group_scores(runs, args.protocol, "validation", reference)
            banks.append(bank)
        bank = banks[0] if len(banks) == 1 else zscore_scores(np.mean(banks, axis=0))
        lone_scores = apply_correction(lone_scores, fit_zscore(bank))
        print(f"[lone_fill] hub correction fitted on {len(bank)} validation lone words of {len(groups)} group(s)")

    #sentence rows untouched; lone rows replaced, z-scored first so every rank stays readable
    filled = primary.copy()
    filled[lone] = softmax_rows(zscore_scores(lone_scores))
    moses = moses / moses.sum(axis=1, keepdims=True)
    print(describe("input", primary, sources))
    print(describe("filled", filled, sources))

    #⚠️ DO NOT REMOVE: the same checks predict makes before writing
    if not np.array_equal(filled[~lone], primary[~lone]):
        raise SystemExit("A sentence row changed; only lone rows may.")
    for name, values in (("primary", filled / filled.sum(axis=1, keepdims=True)), ("moses", moses)):
        sums = values.sum(axis=1)
        if not np.isfinite(values).all() or not np.allclose(sums, 1.0, atol=1e-6):
            raise SystemExit(f"{name} rows are not distributions (sums {sums.min():.6f}..{sums.max():.6f}).")

    #rows renormalised only against CSV rounding; rankings do not move
    filled = filled / filled.sum(axis=1, keepdims=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(str(out_path), indices=index, primary_probs=filled, secondary_probs=moses)
    print(f"[lone_fill] wrote {out_path}: {int(lone.sum())} lone rows from {sum(len(csvs) for csvs, _ in groups)} files in {len(groups)} group(s)"
          + ("" if args.no_hub else ", hub-corrected"))
    print("[lone_fill] the public board cannot see this change; lone words are scored on the private board only.")


if __name__ == "__main__":
    main()
