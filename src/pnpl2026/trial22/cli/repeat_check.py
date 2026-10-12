"""Do other subjects' holdout files hold recordings of subject 0's sentences?

    PYTHONPATH=src python -m pnpl2026.trial22.cli.repeat_check [--subjects 1-39] [--local data/holdout]

Change 1 of the Route to 73 (`handoff-docs/route-to-73-2026-10-12.md`). pnpl
releases a sentence file for each of 40 subjects. If the other subjects heard
the sentences subject 0 heard, every such row has more recordings of the same
word -- extra votes, or material to align and average.

The check reads only each sentence file's timing arrays (`word_onsets_s`,
`word_mask`, `sentence_n_times`, `sfreq`), by HTTP range requests on the
Hugging Face dataset, so no MEG is downloaded. `--local` reads files already
on disk instead. A sentence of subject k MATCHES a sentence of subject 0 when
both hold the same number of words and every onset, measured from the first,
lies within one sample (4 ms at 250 Hz) of its partner: the same audio. A
match counts only when it is unique both ways and the sentence has at least
two words -- a one-word "sentence" carries no timing. Lone words cannot be
matched this way.

Never read: `meg` (not needed) and `shuffle_seed` (undoing a file's shuffle
is off-limits under rule 4).

Writes runs/trial22_repeat_check.md and data/trial22_matches.json, the
subject -> [[epoch of subject 0, epoch of subject k], ...] table every later
step of change 1 reads.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ...config import REPO_ROOT, RUNS_DIR

#the timing arrays a match needs; nothing else is ever read
TIMING_KEYS = ("word_onsets_s", "word_mask", "sentence_n_times", "sfreq")

#the holdout's sample rate, when a file does not say
DEFAULT_RATE = 250.0


def parse_args():
    """Command-line arguments for the repeat check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subjects", default="1-39", help="other subjects: '1-39', or '1,2,5'")
    parser.add_argument("--local", default=None, help="a folder holding COMPETITION_HOLDOUT/; default: range reads from HF")
    parser.add_argument("--tolerance-samples", type=float, default=1.0, help="largest onset difference, in samples")
    parser.add_argument("--min-words", type=int, default=2, help="shortest sentence a match may rest on")
    parser.add_argument("--out", default=str(RUNS_DIR / "trial22_repeat_check.md"), help="the report")
    parser.add_argument("--matches", default="data/trial22_matches.json", help="repo-relative match table")
    return parser.parse_args()


def parse_subjects(spec):
    """Subject ids from a range or a list.

    ARGS:
        spec (str): "1-39", "3", or "1,2,5"

    RETURNS:
        list of int, ascending, without duplicates
    """
    subjects = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        #a range is inclusive at both ends
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            subjects.update(range(low, high + 1))
        else:
            subjects.add(int(part))
    return sorted(subjects)


def sentence_file_name(subject):
    """The repository path of one subject's sentence file.

    ARGS:
        subject (int): 0-39

    RETURNS:
        str, relative to the dataset root
    """
    from pnpl.competition.holdout import HOLDOUT_SUBDIR

    return f"{HOLDOUT_SUBDIR}/subj{subject:02d}_holdout01_sentence.npz"


def read_timing(subject, local_dir=None):
    """One subject's sentence timing, and nothing else.

    ARGS:
        subject (int)
        local_dir (str or None): a folder holding COMPETITION_HOLDOUT/; None
            reads the Hugging Face dataset by range requests

    RETURNS:
        dict: onsets [n_sentences, n_slots] float64 seconds (NaN-padded),
        mask [n_sentences, n_slots] bool, n_times [n_sentences] int64,
        sfreq float
    """
    name = sentence_file_name(subject)

    #a file on disk, if one was named
    if local_dir is not None:
        with np.load(Path(local_dir) / name, allow_pickle=False) as stored:
            return unpack_timing(stored)

    #⚠️ range reads: np.load on the remote file seeks into the zip and
    #decompresses only the members asked for, so the MEG never travels
    from huggingface_hub import HfFileSystem
    from pnpl.competition.holdout import HOLDOUT_REPO

    filesystem = HfFileSystem()
    with filesystem.open(f"datasets/{HOLDOUT_REPO}/{name}", "rb", block_size=1 << 20) as handle:
        with np.load(handle, allow_pickle=False) as stored:
            return unpack_timing(stored)


def unpack_timing(stored):
    """The timing arrays of an opened sentence file.

    ARGS:
        stored (np.lib.npyio.NpzFile)

    RETURNS:
        dict, as `read_timing`
    """
    missing = [key for key in TIMING_KEYS[:3] if key not in stored.files]
    if missing:
        raise SystemExit(f"A sentence file lacks {missing}; found {stored.files}")

    #sfreq is a scalar array; an older file without it is at the holdout's rate
    sfreq = float(stored["sfreq"]) if "sfreq" in stored.files else DEFAULT_RATE
    return {
        "onsets": np.asarray(stored["word_onsets_s"], dtype=np.float64),
        "mask": np.asarray(stored["word_mask"], dtype=bool),
        "n_times": np.asarray(stored["sentence_n_times"]).astype(np.int64),
        "sfreq": sfreq,
    }


def relative_patterns(timing):
    """Each sentence's onsets, measured from its first word.

    ARGS:
        timing (dict): from `read_timing`

    RETURNS:
        list of [n_words] float64 arrays, one per sentence, in stored order
    """
    patterns = []
    for onsets, mask in zip(timing["onsets"], timing["mask"]):
        #the valid slots, in slot order, which is the loader's word order
        valid = onsets[mask]
        patterns.append(valid - valid[0] if len(valid) else valid)
    return patterns


def match_sentences(reference, other, tolerance_s, min_words, exclude_same_index=False):
    """Which of `other`'s sentences are recordings of `reference`'s.

    ARGS:
        reference (list): subject 0's patterns, from `relative_patterns`
        other (list): another subject's patterns
        tolerance_s (float): largest onset difference allowed, in seconds
        min_words (int): shortest sentence a match may rest on
        exclude_same_index (bool): for subject 0 against itself, never pair
            a sentence with itself

    RETURNS:
        pairs : list of (reference epoch, other epoch), unique both ways
        nearest : [n_other] float64, the smallest onset difference to any
            same-length reference sentence (inf without one), for near misses
    """
    #reference sentences by word count: only those can match
    by_length = defaultdict(list)
    for index, pattern in enumerate(reference):
        by_length[len(pattern)].append(index)

    nearest = np.full(len(other), np.inf)
    candidates = {}
    for other_index, pattern in enumerate(other):
        same_length = [index for index in by_length.get(len(pattern), []) if not (exclude_same_index and index == other_index)]
        if not same_length:
            continue

        #the worst onset disagreement with each same-length reference sentence
        distances = np.array([np.max(np.abs(reference[index] - pattern)) if len(pattern) else 0.0 for index in same_length])
        nearest[other_index] = distances.min()
        hits = [index for index, distance in zip(same_length, distances) if distance <= tolerance_s]

        #one candidate only, and long enough to carry timing
        if len(hits) == 1 and len(pattern) >= min_words:
            candidates[other_index] = hits[0]

    #unique both ways: a reference sentence claimed twice is ambiguous
    claims = defaultdict(list)
    for other_index, reference_index in candidates.items():
        claims[reference_index].append(other_index)
    pairs = [(reference_index, owners[0]) for reference_index, owners in sorted(claims.items()) if len(owners) == 1]
    return pairs, nearest


def describe_subject(subject, timing, reference_timing, pairs, nearest):
    """One report row for one subject.

    ARGS:
        subject (int)
        timing (dict): this subject's, from `read_timing`
        reference_timing (dict): subject 0's
        pairs (list): from `match_sentences`
        nearest (np.ndarray): from `match_sentences`

    RETURNS:
        str: a markdown table row
    """
    n_sentences = len(timing["mask"])
    n_words = int(timing["mask"].sum())

    #subject 0's words that gain a second recording
    covered = int(sum(reference_timing["mask"][reference_epoch].sum() for reference_epoch, _ in pairs))
    reference_words = int(reference_timing["mask"].sum())

    #near misses among this subject's unmatched sentences, in milliseconds
    matched_other = {other_epoch for _, other_epoch in pairs}
    unmatched = [distance for index, distance in enumerate(nearest) if index not in matched_other and np.isfinite(distance)]
    near = f"{1000 * np.median(unmatched):.0f}" if unmatched else "-"
    return (f"| {subject} | {n_sentences} | {n_words} | {timing['sfreq']:g} | {len(pairs)} | "
            f"{covered} of {reference_words} ({100 * covered / max(reference_words, 1):.0f}%) | {near} |")


def main():
    """Read every subject's timing, match it against subject 0's, report."""
    args = parse_args()
    subjects = [subject for subject in parse_subjects(args.subjects) if subject != 0]
    where = args.local or "Hugging Face (range reads)"
    print(f"[repeat] subject 0 against {len(subjects)} subjects; reading from {where}")

    #subject 0 first: every match is to its sentences
    reference_timing = read_timing(0, args.local)
    reference = relative_patterns(reference_timing)
    tolerance = args.tolerance_samples / reference_timing["sfreq"] + 1e-9

    lines = [
        "# Trial 22 -- does any other subject's holdout repeat subject 0's sentences?",
        "",
        f"Timing arrays only (no MEG). A match: same word count, every onset from the first within "
        f"{args.tolerance_samples:g} sample(s) ({1000 * tolerance:.1f} ms), unique both ways, at least "
        f"{args.min_words} words. 'near miss' is the median smallest onset difference of the unmatched "
        f"sentences to any same-length sentence of subject 0.",
        "",
    ]

    #subject 0 against itself: repeats inside its own file
    own_pairs, own_nearest = match_sentences(reference, reference, tolerance, args.min_words, exclude_same_index=True)
    finite = own_nearest[np.isfinite(own_nearest)]
    lines.append(f"Subject 0 against itself: {len(reference)} sentences, {len(own_pairs)} repeated pairs; "
                 f"nearest same-length pattern median {1000 * np.median(finite):.0f} ms.")
    lines += ["", "| subject | sentences | words | Hz | matched sentences | subject 0's words covered | near miss (ms) |",
              "|---|---|---|---|---|---|---|"]

    table = {}
    for subject in subjects:
        #a subject whose file cannot be read is reported, not fatal
        try:
            timing = read_timing(subject, args.local)
        except Exception as error:
            lines.append(f"| {subject} | unreadable: {type(error).__name__} | - | - | - | - | - |")
            print(f"[repeat] subject {subject}: unreadable ({error})")
            continue
        pairs, nearest = match_sentences(reference, relative_patterns(timing), tolerance, args.min_words)
        table[str(subject)] = [[int(reference_epoch), int(other_epoch)] for reference_epoch, other_epoch in pairs]
        lines.append(describe_subject(subject, timing, reference_timing, pairs, nearest))
        print(f"[repeat] subject {subject}: {len(pairs)} of {len(reference)} sentences matched")

    #the verdict the schedule turns on
    with_repeats = [subject for subject, pairs in table.items() if len(pairs) >= len(reference) // 2]
    lines += ["", f"Subjects with at least half of subject 0's sentences matched: {len(with_repeats)} "
              f"({', '.join(with_repeats) if with_repeats else 'none'})."]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    matches_path = REPO_ROOT / args.matches
    matches_path.parent.mkdir(parents=True, exist_ok=True)
    matches_path.write_text(json.dumps({
        "tolerance_s": tolerance,
        "min_words": args.min_words,
        "reference_sentences": len(reference),
        "subjects": table,
    }, indent=1), encoding="utf-8")
    print("\n".join(lines))
    print(f"[repeat] wrote {out_path} and {matches_path}")


if __name__ == "__main__":
    main()
