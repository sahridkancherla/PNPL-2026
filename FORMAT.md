# FORMAT.md

**The competition, the data, and how to get onto the HPC.** Reference material,
not rules — the rules are in `CLAUDE.md`, the repo map is in `REPO.md`.

Nothing in this file should be restated elsewhere. Link to it instead.

---

# 1. The competition

**PNPL 2026, deep track** — within-subject MEG decoding.

| | |
|---|---|
| Subject | `"0"` — the *deep* subject, the one with many hours of recording |
| Task | `Sherlock1`, run `"1"`; sessions are numeric strings (`"1"`, `"2"`, … `"28"`) |
| Input | one MEG window per word, `(306 sensors, 250 samples)` |
| Window | `tmin = 0.0` to `tmax = 1.0` s relative to word onset, at `sfreq = 250` Hz |
| Output | one of **50** words |
| Metric | **BAcc@10** |

`T = (tmax - tmin) * sfreq`. All four numbers live in `Config`, so read them
from the config rather than hard-coding 250.

## Why this is hard

**Single trial.** The 2025 phoneme competition allowed averaging many
repetitions of the same stimulus, which raises SNR enormously. Here each
prediction comes from **one** window, so every window must carry its own
evidence. MEG single-trial SNR is poor; expect to be working close to the
noise floor and to need the metric to tell you what your eyes cannot.

**A long tail.** Natural speech is Zipfian. Even inside the 50-word vocabulary
the class counts are far from balanced, which is exactly why the metric is
*balanced* accuracy — a model that always guesses the most frequent words
scores no better than chance on it.

## The metric — BAcc@k

Macro-averaged **Recall@k**: for each class present in the labels, the rate at
which the true word appears in the model's top `k`; then the mean over those
classes.

```
chance = k / n_classes
```

| metric | chance | role |
|---|---|---|
| **BAcc@10** | **0.20** | the leaderboard number |
| BAcc@1 | 0.02 | logged as a tiebreaker only, never scored |

⚠️ **Read every number against the right floor.** A BAcc@1 of 0.05 is 2.5x
chance and a good sign; a BAcc@10 of 0.20 is a model that has learnt nothing.
Confusing the two is the single easiest way to waste a day here.

The implementation is `src/pnpl2026/metrics.py`, mirroring the tutorial
notebook's reference version. Only classes that actually appear in the labels
contribute, so BAcc computed on a small validation split is noisier than the
sample count alone suggests.

---

# 2. The data

## Where it comes from

`pnpl.datasets.LibriBrain100` (and `LibriBrainWord` in the tutorial), fed a
`WordClassification` task. The package downloads and caches on demand.

```bash
uv run python -m pnpl2026.cli.prepare      # login node only
```

⚠️ **`prepare` must run on a login node.** HPC compute nodes have no outbound
internet, so a job that tries to download will hang or fail rather than fetch.

The cache lands in `data/` at the repo root — derived from `REPO_ROOT` in
`config.py`, gitignored via the **anchored** `/data/` pattern.

## Run keys and sample tuples

A run is identified by a 4-tuple `(subject, session, task, run)`, e.g.
`("0", "5", "Sherlock1", "1")`.

Each entry of `dataset.samples` is an 8-tuple:

```
(subject, session, task, run, onset, word, sent_idx, word_idx)
     0        1      2     3     4      5       6         7
```

`src/pnpl2026/data/datasets.py` reads index `1` for the session and index `5`
for the raw word string. Those two positions are load-bearing — if the `pnpl`
task ever changes its tuple layout, the filtering silently keeps the wrong
column rather than raising.

## Sessions are the unit

Sessions vary in size by roughly **30x**. Two consequences:

- **Never split randomly.** Neighbouring windows inside one session are
  near-duplicates; a random split puts them on both sides and validation reads
  far above the leaderboard.
- **Size a split by fraction of samples, not by session count.** Holding out
  "two sessions" can mean 1% or 30% of the data.

`src/pnpl2026/data/splits.py` does this, and sorts session ids *numerically* —
plain string order runs `0, 1, 10, 11, … 2, 20`, which makes "the last session"
mean `9` rather than `28`.

## Event files

Each session has a TSV of events; `dataset.get_events_path(*run_key)` returns
its path. Rows with `kind == "word"` carry the word in the `segment` column.
Useful for counting the vocabulary or inspecting onsets without loading MEG.

---

# 3. The label space — the expensive trap

**Read this before touching targets.**

`WordClassification` numbers its classes as `sorted(set(every word in the
dataset))` — thousands of classes, **alphabetical**. The competition scores 50
classes in `load_vocabulary("primary")` order, which is **frequency-ordered**.

⚠️ **These two numberings are unrelated, and mixing them fails silently.** The
model trains, the loss descends, validation looks plausible, and the
submission scores at chance — because column 0 means a different word to you
than to the scorer.

**Every target in this project comes from `vocab.to_id()`, never from
`task.label_info`.** `src/pnpl2026/data/vocab.py` is the only place that
defines the mapping; do not build a second one.

## Normalisation

Words are matched case-insensitively. The tutorial also unifies the curly
apostrophe (`it’s` → `it's`) and strips whitespace; `vocab.py` currently does
`.lower()` only.

## The two vocabularies

| | |
|---|---|
| `load_vocabulary("primary")` | the 50 scored words — **the leaderboard metric** |
| `load_vocabulary("moses")` | a second 50-word set, a secondary metric |

This project trains and predicts on **primary** only. `cli/predict.py` passes
`primary_probs` and lets `write_submission` own everything else.

---

# 4. Holdout and submission

```bash
uv run python -m pnpl2026.cli.predict --ckpt runs/<name>/best.ckpt
```

`LibriBrainCompetitionHoldout(track="deep")` yields the holdout windows;
`pnpl.competition.write_submission` writes the CSV into `submissions/`.

## ⚠️ Row ordering is the whole contract

`iter_windows()` yields windows in the canonical order, and the host builds its
label CSV by iterating **the same loader**. Indices therefore align by
construction — and *only* by construction.

**Never sort, shuffle, filter or reorder the outputs of `iter_windows()`.**
Probabilities are concatenated in exactly the order yielded. Reordering them
breaks alignment with **no error and no warning**; the only symptom is a
chance-level leaderboard score against a validation number that looked fine.

`predict.py` checks the prediction count against `len(holdout.indices)` and
raises on a mismatch. That check is cheap and catches a whole class of silent
failure — do not remove it.

## Before submitting

Validate the file for row count, column count, ordering, finiteness (no `NaN`
or `inf`) and per-row probability sums.

⚠️ **No Kaggle credentials are configured on this machine** (`~/.kaggle/` is
empty). Submission is manual, through the web UI. Nothing in this repo can
submit, so nothing should ever claim to have submitted.

---

# 5. The HPC

> ## ⚠️ DO NOT USE THE HPC UNLESS EXPLICITLY TOLD TO
>
> This section is reference material for when that instruction comes. Do not
> `ssh`, do not `qsub`, do not `scp`, and do not start a job on your own
> initiative. Shared allocation, real queue, other people's time.

## What it is

Imperial's **CX3** cluster, running **PBS Pro**. Jobs are submitted from a
login node and run on compute nodes; storage is **RDS**, visible from both.

## Access — fill these in before first use

These are the machine-specific details, deliberately left blank rather than
guessed. Confirm each against Imperial's current RCS documentation, then record
it here once — not in six places.

| what | value |
|---|---|
| login host | `<TO CONFIRM>` |
| username | `<TO CONFIRM>` |
| VPN / on-campus requirement | `<TO CONFIRM>` |
| RDS project path | `<TO CONFIRM>` |
| repo location on RDS | `$HOME/PNPL-2026` (assumed by `pbs/*.pbs`) |

⚠️ **Never hard-code any of these into code.** `pbs/*.pbs` uses `$HOME`
precisely so it stays portable; keep it that way.

## Login node vs compute node

| | login node | compute node |
|---|---|---|
| outbound internet | yes | **no** |
| heavy compute | **no** | yes |
| use it for | `prepare`, `git`, `qsub`, tailing logs | training, sweeps |

Running training on a login node is antisocial and will be noticed. Downloading
data from a compute node cannot work.

## Submitting

```bash
cd "$HOME/PNPL-2026"
qsub pbs/train.pbs          # one run
qsub pbs/sweep.pbs          # array job, one task per line of configs/sweep.txt
qstat -u "$USER"            # what is queued or running
qdel <jobid>                # cancel
```

Both scripts do the same preamble: `cd "$HOME/PNPL-2026"`,
`source "$HOME/.local/bin/env"` (the `uv` installer's env), set `PYTHONPATH` to
`src`, and set `PYTHONUNBUFFERED=1`.

⚠️ **`configs/sweep.txt` does not exist yet.** `sweep.pbs` reads line
`$PBS_ARRAY_INDEX` from it and exits with an error if the line is empty. Create
it — one config path per line — before submitting a sweep.

## Job sizing

The current request is `select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=RTX6000`
with `walltime=08:00:00`.

⚠️ **Verify `gpu_type` and the cpus/mem/walltime combination against Imperial's
current job-sizing table before submitting.** CX3 accepts only combinations
matching an allowed job class; anything else is rejected outright or queues
indefinitely with no explanation.

⚠️ **`num_workers` in the config must match `ncpus` in the PBS script.** The
data loader is the bottleneck, so a mismatch either starves the GPU or wastes
half the allocation.

## Watching a running job

PBS spools the `-o` file on the compute node's local disk and copies it back to
`logs/` only **when the job exits** — so it cannot be tailed while running.
Both scripts `tee` to a second file on RDS, which does update live:

```bash
tail -f logs/<jobid>.live.log              # train.pbs
tail -f logs/<jobid>.<index>.live.log      # sweep.pbs array task
```

The job prints its own live-log path as its first line.

## Walltime is a kill, not a failure

Both scripts pass `--resume`, which continues from `last.ckpt`. An eight-hour
overrun therefore means **resubmitting the same script**, not starting over.
Plan runs on that basis rather than trying to fit training inside one window.

`set -euo pipefail` is set, so a Python failure still fails the job through the
`tee` rather than being swallowed by the pipe.
