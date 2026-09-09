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

## Scale — measured, 2026-09-09

Counted directly from the 162 event tables, matched against the 50-word
vocabulary in `cache/v2/meta.npz`. Re-measure if the partition changes.

| | |
|---|---|
| Word events, all tasks | 780,277 |
| **In-vocabulary windows (`N`)** | **313,068** — 40.1% of all words |
| Recordings | 162, over 12 tasks — **`sub-0` only**, every file |
| Classes present | 50 / 50 |
| Median class | 3,482 |
| **Class imbalance** | **121x** |

⚠️ The vocabulary is **not** the corpus top-50. `people`, `new`, `really`,
`always`, `think`, `good` and `time` do not crack the frequency top-50 — they
come from TheMoth and TIMIT. That is good for decodability (content words carry
more signal than function words) and bad for sampling (see below).

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

## On-disk layout

Each task folder holds two directories of 1 file per recording:

```
<task>/derivatives/events/sub-0_ses-<n>_task-<task>_run-<r>_events.tsv
<task>/derivatives/serialised/sub-0_ses-<n>_task-<task>_run-<r>_proc-<chain>_meg.h5
```

The processing chain is baked into the filename:
`bads+headpos+sss+notch+bp+ds` — bad-channel repair, head-position correction,
**SSS**, notch, bandpass, downsample. The `sss` term is the one with
consequences — see *The input is rank 73* below.

`.h5` holds **continuous** MEG, one `(306, T_run)` array per recording — not
epochs. Windows are cut at training time from the event tables.

⚠️ The dataset currently lives on an external volume, and **that path must never
appear in code or config.** It is not mounted on the cluster, not mounted on a
collaborator's machine, and not guaranteed to be mounted here. `config.py`
resolves `data/` from `REPO_ROOT`; keep it that way.

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

## Sample rate and window bounds

Windows arrive at **250 Hz**. The signal is bandpassed upstream (`bp` in the
processing chain), and an earlier pipeline decimated to **125 Hz** (`decim: 2`)
and kept it — so half the native rate is known to be enough, not a guess.
Confirm against the PSD before relying on it.

That pipeline also cut a **wider window than this repo does**: `tmin = -0.32`,
`tmax = 1.28`, 200 samples at 125 Hz, with the `[0, 1] s` core at samples
40-165. The 40 pre-onset samples are the margin that makes **time-jitter
augmentation possible at all** — at this repo's `0.0` to `1.0` there is nothing
to jitter into.

## The input is rank 73, not 306

**The single most consequential fact about this data.** SSS (Signal Space
Separation) projects the sensor array onto a basis of internal-source
harmonics and discards the rest, so the 306 sensors span far fewer than 306
dimensions. Measured from the noise-covariance eigenspectrum in
`cache/v2/meta.npz`:

```
eig[  0] = 1.46e+01     cumulative variance   15.40%
eig[ 29] = 3.72e-01                           94.52%
eig[ 71] = 2.01e-02                           99.89%
eig[ 73] = 1.26e-02                           99.92%   <- rank cut taken here
eig[ 99] = 8.45e-05                           99.999%  <- 135x cliff
eig[149] = 4.64e-07                          100.000%  <- numerical zero
```

**Effective rank is 73.** Everything past it is linear combinations of the
rest, and any layer given more than 73 output dimensions at the sensor
boundary is free to amplify numerical noise in directions that carry nothing.

Consequences:

- **Whiten to ~73 components at the data boundary**, not inside the model. A
  `(73, 306)` whitener plus per-channel `mean`/`std` is the natural stem.
- **Re-derive the whitener from the training split.** Whitening is
  unsupervised so leakage is mild, but there is no reason to inherit another
  split's statistics.
- After whitening, **channel-wise reasoning about magnetometers vs
  gradiometers no longer applies** — the whitener mixes them.
- Storage falls out of it: `313k x 73 x 200` fp16 is **9.1 GB** and fits in
  RAM, against **47.9 GB** for `313k x 306 x 250`. That is the difference
  between an I/O-bound and a compute-bound training loop.

## Class imbalance and the validation noise floor

The 50 classes span **121x**:

| | n |
|---|---|
| `the` | 41,143 |
| `and` | 21,439 |
| *median class* | 3,482 |
| `it's` | 698 |
| `really` | 467 |
| `always` | 406 |
| `new` | 371 |
| **`people`** | **341** |

⚠️ **Full inverse-frequency class weights span the same 121x**, which makes the
batch gradient depend on which rare tokens happen to land in it. Prefer
**square-root inverse frequency** (ratio ~11x), a weight cap, or a per-class
sample cap — the earlier pipeline capped at 3,500/class.

⚠️ **The rare classes set the validation noise floor.** `people` is spread over
126 of 162 recordings, so no session holdout isolates it cleanly. A 10% holdout
leaves it ~34 validation examples; per-class recall at n=34 has **SE ~8.6
points**, and BAcc averages 50 such estimates. **Size the validation split by
the rarest class, not by overall sample count**, and treat small BAcc
differences as noise.

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
