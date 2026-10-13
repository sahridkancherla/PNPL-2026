"""Which holdout subject a prediction reads, and which subject layer it goes through.

Change 1 of the Route to 73: subjects 1-39's holdout files hold recordings of
subject 0's 156 sentences (`cli/repeat_check.py`). `cli.predict --subject K`
scores subject K's own files, in subject K's own canonical row order; the
rows are mapped onto subject 0's later, by `cli/cross_votes.py`.

A per-word all-data run (trial 18's `all_g2`) has one subject layer per
published subject. `--subject-layer auto` finds subject K's layer in the
other subjects' cache manifest: layer 1 + the subject's position in its
`subjects` list, exactly as `perword/data/dataset.py` assigns them in
training. Layer 0 is subject 0's.
"""

from __future__ import annotations

import json

from ..config import REPO_ROOT


def open_holdout(args):
    """The holdout a run predicts: the track's subjects, or one named subject.

    ARGS:
        args (argparse.Namespace): with track, holdout_dir and subject

    RETURNS:
        LibriBrainCompetitionHoldout
    """
    from pnpl.competition import LibriBrainCompetitionHoldout

    #the default: the track, which for "deep" is subject 0 alone
    if args.subject is None:
        return LibriBrainCompetitionHoldout(track=args.track, data_path=args.holdout_dir)
    return LibriBrainCompetitionHoldout(subjects=[int(args.subject)], data_path=args.holdout_dir)


def subject_suffix(args):
    """What a file name gains when another subject is predicted.

    ARGS:
        args (argparse.Namespace): with subject

    RETURNS:
        str: "" for the track or subject 0, "_subj07" for subject 7
    """
    if args.subject is None or int(args.subject) == 0:
        return ""
    return f"_subj{int(args.subject):02d}"


def subject_names(subject):
    """The spellings a subject id may take in a manifest.

    ARGS:
        subject (int)

    RETURNS:
        list of str
    """
    return [str(subject), f"{subject:02d}", f"sub-{subject}", f"sub-{subject:02d}"]


def resolve_subject_layer(args, n_subjects):
    """The subject layer every window goes through, or None for subject 0's map.

    ARGS:
        args (argparse.Namespace): with subject, subject_layer, others_manifest
        n_subjects (int): the run's subject layers, from its config

    RETURNS:
        int or None
    """
    choice = str(args.subject_layer).lower()
    if choice == "none":
        return None

    #an explicit index, checked against the run
    if choice != "auto":
        layer = int(choice)
    else:
        if args.subject is None:
            raise SystemExit("--subject-layer auto needs --subject: whose layer?")
        path = REPO_ROOT / args.others_manifest
        if not path.exists():
            raise SystemExit(f"No {path}: the other subjects' cache manifest names each layer's subject.")
        listed = [str(name) for name in json.loads(path.read_text(encoding="utf-8"))["subjects"]]

        #the first spelling the manifest uses; layer 0 is subject 0
        found = [name for name in subject_names(int(args.subject)) if name in listed]
        if not found:
            raise SystemExit(f"Subject {args.subject} has no layer: the manifest lists {listed}.")
        layer = 1 + listed.index(found[0])

    #⚠️ a layer past the run's count would index another subject's weights or crash
    if not 0 <= layer < n_subjects:
        raise SystemExit(f"Subject layer {layer} does not exist: the run has {n_subjects}.")
    return layer
