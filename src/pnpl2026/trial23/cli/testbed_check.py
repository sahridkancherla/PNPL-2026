"""Did the other subjects hear subject 0's Sherlock validation and test chapters? Word by word.

    PYTHONPATH=src python -m pnpl2026.trial23.cli.testbed_check [--data-dir data]

Step 0 of the 13 Oct research note. Subject 0's validation and test
recordings are Sherlock1 sessions 11 and 12 (run 2); pnpl's manifest lists
the broad-track subjects' Sherlock1 sessions 11 and 12 (run 1). If both are
the same audio, the 12 cached subjects' recordings of those chapters are a
LABELLED replica of the holdout's cross-subject setting -- the testbed every
cross-subject method is gated on.

Per subject and session, both events tables at 250 Hz (`timeds`, exact):

    1   the word sequences aligned (difflib, on normalised words)
    2   every aligned pair's onset offset. Measured 14 Oct: the text is
        identical, and the offset is constant to the sample within a
        stretch of continuous playback but jumps by 2-8 samples (8-32 ms) at
        the 1-3 s pauses between sentences -- playback timing at the pauses
        differed per session -- so it drifts by up to ~1.6 s over a chapter
    3   a pair is exact when it lies in a run of at least 3 consecutive
        aligned words whose offsets step by at most one sample (4 ms); the
        report gives the share exact, the runs, the offsets' range, and the
        largest deviation from a run's own median offset

Writes runs/trial23_testbed_check.md and data/trial23_testbed_map.npz: for
each (subject, session), subject 0's word-table rows, the subject's rows of
the same words (exact pairs only), and the offset at 50 Hz -- what
`cli/testbed.py` reads. The rows index the reference caches' word tables,
which `load_word_table` builds from the same events with the same filtering.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from ...config import DATA_DIR, REPO_ROOT, RUNS_DIR
from ..data_processing.build_reference import discover_recordings, load_word_table

#the events' own rate, and the caches'
EVENTS_RATE = 250.0
CACHE_RATE = 50.0

#subject 0's recordings of the two chapters, and the others' (pnpl's manifest)
REFERENCE_RUN = "2"
OTHERS_RUN = "1"
SESSIONS = ("11", "12")
TASK = "Sherlock1"

#consecutive aligned words belong to one run when their offsets differ by at most this
EXACT_SAMPLES = 1

#a run this long is continuous playback; shorter ones are dropped as chance alignments
MIN_RUN = 3


def parse_args():
    """Command-line arguments for the testbed check."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="the download root holding the task folders")
    parser.add_argument("--out", default=str(RUNS_DIR / "trial23_testbed_check.md"))
    parser.add_argument("--map", default="data/trial23_testbed_map.npz", help="repo-relative")
    return parser.parse_args()


def align_words(reference_words, other_words):
    """Index pairs of the same words in two transcripts of (perhaps) one text.

    ARGS:
        reference_words (np.ndarray): [n] normalised words
        other_words (np.ndarray): [m]

    RETURNS:
        [k, 2] int64: (reference row, other row) of every aligned word
    """
    matcher = SequenceMatcher(None, list(reference_words), list(other_words), autojunk=False)
    pairs = []
    for block in matcher.get_matching_blocks():
        #each block is a run of identical words in both transcripts
        pairs += [(block.a + offset, block.b + offset) for offset in range(block.size)]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def exact_pairs(reference_onsets, other_onsets, pairs, min_run=MIN_RUN):
    """The aligned pairs inside runs of continuous playback: one offset, to within a sample.

    ARGS:
        reference_onsets (np.ndarray): [n] samples at 250 Hz
        other_onsets (np.ndarray): [m]
        pairs (np.ndarray): [k, 2] from `align_words`
        min_run (int): consecutive pairs a run needs

    RETURNS:
        exact [j, 2], median offset (int, samples at 250 Hz), each exact
        pair's deviation from its run's median offset [j], runs kept (int)
    """
    if len(pairs) == 0:
        return pairs, 0, np.zeros(0), 0
    offsets = other_onsets[pairs[:, 1]] - reference_onsets[pairs[:, 0]]

    #a new run wherever the offset steps by more than a sample: a playback pause
    breaks = np.flatnonzero(np.abs(np.diff(offsets)) > EXACT_SAMPLES) + 1
    keep = np.zeros(len(pairs), dtype=bool)
    deviations = np.zeros(len(pairs))
    runs = 0
    for run in np.split(np.arange(len(pairs)), breaks):
        deviations[run] = np.abs(offsets[run] - np.median(offsets[run]))
        if len(run) >= min_run:
            keep[run] = True
            runs += 1
    return pairs[keep], int(np.round(np.median(offsets))), deviations[keep], runs


def find(recordings, subject, session, run):
    """One recording's entry, or None.

    ARGS:
        recordings (list): from `discover_recordings`
        subject (str), session (str), run (str)

    RETURNS:
        dict or None
    """
    return next((record for record in recordings if record["subject"] == subject and record["task"] == TASK
                 and record["session"] == session and record["run"] == run), None)


def main():
    """Align every cached subject's chapters with subject 0's, report and save the map."""
    args = parse_args()
    reference_recordings = discover_recordings(Path(args.data_dir))
    other_recordings = discover_recordings(Path(args.data_dir), subjects=None)
    subjects = sorted({record["subject"] for record in other_recordings
                       if record["task"] == TASK and record["session"] in SESSIONS}, key=int)
    if not subjects:
        raise SystemExit(f"No other subject's {TASK} sessions {SESSIONS} under {args.data_dir}.")

    lines = ["# Trial 23 -- the Sherlock testbed: did the others hear subject 0's chapters?", "",
             f"Subject 0's {TASK} sessions {', '.join(SESSIONS)} run {REFERENCE_RUN} against each subject's run "
             f"{OTHERS_RUN}. Words aligned by text; a pair is exact when it lies in a run of {MIN_RUN}+ consecutive "
             f"aligned words whose offsets step by at most {EXACT_SAMPLES} sample (4 ms): continuous playback, "
             "the offset jumping only at pauses.", "",
             "| subject | session | subject 0 words | subject's words | aligned | exact | runs | offset range (s) | "
             "largest in-run deviation (ms) |",
             "|---|---|---|---|---|---|---|---|---|"]
    stored = {}
    for session in SESSIONS:
        record = find(reference_recordings, "0", session, REFERENCE_RUN)
        if record is None:
            raise SystemExit(f"No subject-0 {TASK} ses-{session} run {REFERENCE_RUN} events under {args.data_dir}.")
        reference = load_word_table(record["events_path"], sfreq=EVENTS_RATE)
        for subject in subjects:
            other_record = find(other_recordings, subject, session, OTHERS_RUN)
            if other_record is None:
                lines.append(f"| {subject} | {session} | - | missing | - | - | - | - |")
                continue
            other = load_word_table(other_record["events_path"], sfreq=EVENTS_RATE)

            #the same text first, then the same timing
            pairs = align_words(reference["word"], other["word"])
            exact, median, deviations, runs = exact_pairs(reference["onset_sample"], other["onset_sample"], pairs)
            offsets = other["onset_sample"][exact[:, 1]] - reference["onset_sample"][exact[:, 0]] if len(exact) else np.zeros(1)
            in_run = 1000.0 * deviations.max() / EVENTS_RATE if len(deviations) else float("nan")
            lines.append(f"| {subject} | {session} | {len(reference['word'])} | {len(other['word'])} | "
                         f"{len(pairs)} | {len(exact)} ({100 * len(exact) / max(len(reference['word']), 1):.1f}%) | {runs} | "
                         f"{offsets.min() / EVENTS_RATE:+.3f} .. {offsets.max() / EVENTS_RATE:+.3f} | {in_run:.1f} |")
            key = f"sub{subject}_ses{session}"
            stored[f"{key}_pairs"] = exact
            stored[f"{key}_offset50"] = np.array(int(np.round(median * CACHE_RATE / EVENTS_RATE)))
            stored[f"{key}_other_key"] = np.array(other_record["key"])
            print(f"[testbed] subject {subject} ses-{session}: {len(exact)} exact of {len(reference['word'])} words")
        stored[f"ses{session}_reference_key"] = np.array(record["key"])

    #the verdict the testbed turns on
    good = [key for key in stored if key.endswith("_pairs") and len(stored[key]) > 0]
    lines += ["", f"Recordings with exact pairs: {len(good)} of {2 * len(subjects)}."]
    map_path = REPO_ROOT / args.map
    map_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(map_path, subjects=np.array(subjects), **stored)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[testbed] wrote {out_path} and {map_path}")


if __name__ == "__main__":
    main()
