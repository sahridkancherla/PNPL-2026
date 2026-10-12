# CLAUDE.md

**The competition, the data, and the rules for working here.**
The repo map is `REPO.md`. Nothing in this file should be restated there.

---

# 1. The goal

**PNPL 2026, deep track** — within-subject MEG decoding on `sub-0`.

Classify a **single word** from a **single trial**, over a fixed 50-word
vocabulary. Single-trial is the whole difficulty: there is no trial averaging
to lean on, so every window must carry its own evidence.

| | |
|---|---|
| Subject | `"0"` — the *deep* subject, the one with many hours of recording |
| Input | one MEG window per word, `(306 sensors, 250 samples)` |
| Window | `tmin = 0.0` to `tmax = 1.0` s relative to word onset, `sfreq = 250` Hz |
| Output | one of **50** words |
| Metric | **BAcc@10** — **chance is 20%**, not 2% |

**Data policy (revised 2026-10-07).** Any training data may be used, provided
its use complies with its licence and with ethical requirements. Data from
outside LibriBrain does not come with LibriBrain's preprocessing
(`bads+headpos+sss+notch+bp+ds`); we must do that ourselves. LibriBrain100's
32 broad-track subjects come preprocessed, but they recorded only Sherlock1
ses-11/12 -- subject 0's validation and test stories.

⚠️ **Holdout use (revised 2026-10-12).** The organisers have confirmed that
the holdout data may be used in any way before a submission is generated --
pooled statistics, transductive inference, training on holdout MEG, and the
other 39 subjects' holdout files (relayed 12 Oct; wording and date in
`handoff-docs/route-to-73-2026-10-12.md`). Still off-limits under rule 4:
reading labels off the leaderboard, looking the sentences up outside the data
or rebuilding the stimulus list, and undoing a file's shuffle with its stored
`shuffle_seed`. Every step that adapts on the holdout must sit inside the
pipeline, seeded and re-runnable: top teams hand over runnable code.

`T = (tmax - tmin) * sfreq`. All four numbers live in `Config`; read them from
the config rather than hard-coding 250.

The immediate aim is a pipeline that trains end to end and produces a valid
submission, so that model choices can be **measured rather than argued about**.
`cnn_baseline` is the reference point, not the destination.

## Why this is hard

**Single trial.** The 2025 phoneme competition allowed averaging many
repetitions of the same stimulus, which raises SNR enormously. Here each
prediction comes from **one** window. MEG single-trial SNR is poor; expect to
work close to the noise floor and to need the metric to tell you what your eyes
cannot.

**A long tail.** Natural speech is Zipfian. Even inside the 50-word vocabulary
the class counts are far from balanced — which is exactly why the metric is
*balanced* accuracy. A model that always guesses frequent words scores no
better than chance.

## The metric — BAcc@k

Macro-averaged **Recall@k**: for each class present in the labels, the rate at
which the true class appears in the model's top `k`; then the mean over
classes. Implementation is `src/pnpl2026/metrics.py`.

**BAcc@10 is the scored metric.** BAcc@1 is logged as a tiebreaker only. A
BAcc@1 of 0.05 is not "broken" — read every number against the right floor.

---

# 2. The data

## Scale — measured 2026-09-09

Counted from the 162 event tables against the 50-word vocabulary.

| | |
|---|---|
| Word events, all tasks | 780,277 |
| **In-vocabulary windows (`N`)** | **313,068** — 40.1% of all words |
| Recordings | **162**, over 12 tasks, `sub-0` only |
| Sessions | 31 (`0`–`30`) |
| Classes present | 50 / 50 |
| Median class | 3,482 |
| **Class imbalance** | **121x** (`the` 41,143 / `people` 341) |

⚠️ The vocabulary is **not** the corpus top-50. `people`, `new`, `really`,
`always`, `think`, `good` and `time` come from TheMoth and TIMIT. Good for
decodability (content words carry more signal), bad for sampling.

## Where it comes from

`pnpl.datasets.LibriBrain100` fed a `WordClassification` task; the package
downloads and caches on demand.

```bash
uv run python -m pnpl2026.cli.prepare      # login node only
```

⚠️ **`prepare` must run where there is outbound internet.** HPC compute nodes
have none, so a job that tries to download will hang rather than fetch.

The cache lands in `data/` at the repo root, derived from `REPO_ROOT` in
`config.py` and gitignored via the **anchored** `/data/` pattern.

⚠️ **`/data/` in `.gitignore` must keep its leading slash.** Unanchored `data/`
also matches `src/pnpl2026/data/` and silently drops the package from git. This
has already happened once.

## On-disk layout

Each task folder holds two directories, one file per recording:

```
<task>/derivatives/events/sub-0_ses-<n>_task-<task>_run-<r>_events.tsv
<task>/derivatives/serialised/sub-0_ses-<n>_task-<task>_run-<r>_proc-<chain>_meg.h5
```

The processing chain is baked into the filename:
`bads+headpos+sss+notch+bp+ds`. The `sss` term is the consequential one — see
*rank* below.

⚠️ The dataset currently lives on an external volume, and **that path must
never appear in code or config.** It is not mounted on the cluster or on a
collaborator's machine.

### Verified file contents — 2026-09-14

**`.h5`** holds continuous MEG, not epochs:

| | |
|---|---|
| `data` | `(306, n_times)` float32 — the recording |
| `times` | `(n_times,)` float32, starts at 0, step exactly `1/250` |
| attrs | `sample_frequency` 250.0, `highpass_cutoff` 0.1, `lowpass_cutoff` 125.0, `n_meg_channels` 306, `channel_types` (`mag, grad, grad` repeating) |

Recording length **varies by task** — MOCHATIMIT runs ~1329 s `(306, 332250)`.
Do not assume a fixed length.

**Events TSV** columns: `idx, wavile, kind, segment, sentenceidx, wordidx,
phonemeidx, timemeg, timeds, timesentence, duration`.

- Word rows are `kind == "word"`; the word is in `segment`. The table also
  holds phoneme rows.
- ⚠️ **There is no `onset` column. Use `timeds`.** `timemeg` is the original
  1 kHz timeline and lands off the 250 Hz sample grid 6% of the time — a random
  sub-sample shift per trial. `timeds` is the downsampled timeline and matches
  the h5 exactly. `timesentence` is time *within the sentence*.

## Run keys and sample tuples

A run is `(subject, session, task, run)`, e.g. `("0", "5", "Sherlock1", "1")`.
Each entry of `dataset.samples` is an 8-tuple:

```
(subject, session, task, run, onset, word, sent_idx, word_idx)
     0        1      2     3     4      5       6         7
```

`data/datasets.py` reads index `1` for session and `5` for the raw word. Those
positions are load-bearing — if the tuple layout changes, the filtering keeps
the wrong column rather than raising.

## Sessions are the unit

Sessions vary in size by roughly **30x**. Two consequences:

- **Never split randomly.** Neighbouring windows inside one session are
  near-duplicates; a random split puts them on both sides and validation reads
  far above the leaderboard.
- **Size a split by fraction of samples, not session count.** Holding out "two
  sessions" can mean 1% or 30% of the data.

`data/splits.py` does this, and sorts session ids **numerically** — plain string
order runs `0, 1, 10, 11, … 2, 20`, making "the last session" mean `9`.

## Sample rate and windows

Windows arrive at **250 Hz**. The signal is bandpassed **0.1–125 Hz**, i.e.
right up to its own Nyquist with no anti-alias headroom, and **9.97% of total
power sits above 62.5 Hz**.

An earlier pipeline decimated to 125 Hz and cut a wider window: `tmin = -0.32`,
`tmax = 1.28`, 200 samples at 125 Hz with the `[0,1]s` core at samples 40–165.
The pre-onset margin is what makes time-jitter augmentation possible at all.

⚠️ **Decimation cannot be a cache stage.** A proper anti-alias filter needs
~200 taps, which is fine on a continuous recording but impossible on a
250-sample holdout window — and `iter_windows()` only ever yields pre-cut
windows. Any downsampling must be identical on both, so it belongs in the
model or the dataset, not in the builder.

## The input is rank 73 — or nearer 96

SSS projects the sensor array onto a basis of internal-source harmonics, so the
306 sensors span far fewer than 306 dimensions.

Measured on 3 MOCHATIMIT runs, 2026-09-14:

```
eig[ 0] = 6.48e+01
eig[72] = 8.14e-02     last kept at rank 73
eig[73] = 3.93e-02     only 2.1x down -- no cliff here
eig[96] = ---          largest gap in the range: 46.6x
eig[99] = 5.29e-09     effectively zero
eig[149]= 4.48e-15     numerical zero
```

Rank 73 keeps 99.98% of variance, but **the actual collapse is at ~96, not 73**
— the cut at 73 does not sit on a cliff. `cache/v2`'s stored spectrum agrees,
placing its 135x cliff at `eig[99]`.

Consequences:

- **Whiten at the data boundary**, not inside the model. Any layer given more
  than the true rank at the sensor boundary can amplify numerical noise in
  directions carrying nothing.
- **Re-derive the whitener from the training split.** Whitening is
  unsupervised so leakage is mild, but there is no reason to inherit another
  split's statistics.
- After whitening, **mag-vs-grad reasoning no longer applies** — the whitener
  mixes them.

Magnetometers and gradiometers differ in scale by **~21x** (measured 20.7x and
21.6x independently), so per-channel standardisation is required before any
covariance: `C_gg / C_mm ~ 21.6^2 ~ 466`, and every eigenvector would otherwise
point into the gradiometer subspace.

## Class imbalance and the validation noise floor

⚠️ **Full inverse-frequency class weights span the same 121x**, making the batch
gradient depend on which rare tokens land in it.

⚠️ **The rare classes set the validation noise floor.** `people` is spread over
126 of 162 recordings, so no session holdout isolates it cleanly. A 10% holdout
leaves it ~34 validation examples; per-class recall at n=34 has **SE ~8.6
points**, and BAcc averages 50 such estimates. **Size the validation split by
the rarest class**, and treat small BAcc differences as noise.

---

# 3. The label space — the expensive trap

**Read this before touching targets.**

`WordClassification` numbers its classes as `sorted(set(every word in the
dataset))` — thousands of classes, **alphabetical**. The competition scores 50
classes in `load_vocabulary("primary")` order, which is **frequency-ordered**.

⚠️ **These two numberings are unrelated, and mixing them fails silently.** The
model trains, the loss descends, validation looks plausible, and the submission
scores at chance — because column 0 means a different word to you than to the
scorer.

**Every target comes from `vocab.to_id()`, never from `task.label_info`.**
`data/vocab.py` is the only place that defines the mapping; do not build a
second one.

Words are matched case-insensitively. The tutorial also unifies the curly
apostrophe (`it’s` → `it's`) and strips whitespace; `vocab.py` currently does
`.lower()` only.

| | |
|---|---|
| `load_vocabulary("primary")` | the 50 scored words — **the leaderboard metric** |
| `load_vocabulary("moses")` | a second 50-word set, a secondary metric |

This project trains and predicts on **primary** only.

---

# 4. Holdout and submission

```bash
uv run python -m pnpl2026.cli.predict --ckpt runs/<name>/best.ckpt
```

## ⚠️ Row ordering is the whole contract

`iter_windows()` yields windows in the canonical order, and the host builds its
label CSV by iterating **the same loader**. Indices align by construction — and
*only* by construction.

**Never sort, shuffle, filter or reorder the outputs of `iter_windows()`.**
Reordering breaks alignment with **no error and no warning**; the only symptom
is a chance-level leaderboard score against a validation number that looked
fine.

`predict.py` checks the prediction count against `len(holdout.indices)` and
raises on a mismatch. **Do not remove that check.**

⚠️ `iter_windows()` yields **raw 306-channel windows**. Any transform the model
depends on — standardiser, whitener, crop — must be applied here too, or the
holdout is served something the model never saw.

## Before submitting

Validate row count, column count, ordering, finiteness and per-row probability
sums.

⚠️ **No Kaggle credentials are configured on this machine** (`~/.kaggle/` is
empty). Submission is manual, through the web UI. Nothing in this repo can
submit, so **nothing should ever claim to have submitted**.

---

# 5. Rules

## Git — Claude does not commit

> ⚠️ **Kamil commits. Claude does not.** Never run `git commit`, and do not
> stage files on your own initiative either. Leave changes in the working tree
> and say plainly what is uncommitted.

This **overrides** the machine-global `~/.claude/rules/git-workflow.md` ("commit
freely, never push unasked"). The push half still holds — never push.

Writing files, editing them, `git mv` and `git status` are all fine. When work
reaches a natural savepoint, say so and offer a commit message rather than
running it.

## Data and storage

- **All bulk data, checkpoints, logs and submissions stay out of git** — under
  `data/`, `runs/`, `logs/`, `submissions/`, all gitignored.
- **Never hard-code an absolute path**, in code or in prose. Paths derive from
  `REPO_ROOT` in `config.py`; one machine's path is broken on the next.

## Compute

> ⚠️ **Do not use the HPC unless explicitly told to.** No `ssh`, no `qsub`, no
> `scp`, no starting a job on your own initiative. Shared allocation, real
> queue, other people's time. §6 is reference for when the instruction comes.

- **State expected duration and load before launching anything long**, wherever
  it runs.
- **Never run two compute-heavy jobs concurrently** on the same machine.

## Experimental discipline

- **Split by whole session, never randomly**, sized by fraction of samples.
- **Log training metrics alongside validation metrics.** Without the gap you
  cannot tell underfitting from overfitting, and so cannot choose between
  adding capacity and adding regularisation.
- **Record negative results.** The cheapest thing here to lose and the most
  expensive to rediscover.

## Adding a model

1. **Write it.** A new file in `models/`, a plain `nn.Module` taking
   `n_channels` and `n_classes`.
2. **Register it.** One line in `models/__init__.py`.
3. **Write a config.** `configs/my_model.yaml` with **only the lines that
   differ** from `base.yaml`.

Run it against the same session split as the baseline, or the comparison means
nothing. **Run the `shape-check` skill before the first training run** — a
silent shape bug trains happily to exactly chance.

## Miscellaneous traps

- **PyYAML implements YAML 1.1**: `lr: 1e-5` loads as the *string* `"1e-5"`.
  `config._coerce` catches it, but write `1.0e-5` in YAML regardless.

---

# 6. The HPC — reference only

> ## ⚠️ DO NOT USE THE HPC UNLESS EXPLICITLY TOLD TO

Imperial's **CX3** cluster, running **PBS Pro**. Jobs are submitted from a login
node and run on compute nodes; storage is **RDS**, visible from both.

## Access — fill these in before first use

Deliberately blank rather than guessed. Confirm against Imperial's current RCS
documentation, then record here once.

| what | value |
|---|---|
| login host | `<TO CONFIRM>` |
| username | `<TO CONFIRM>` |
| VPN / on-campus requirement | `<TO CONFIRM>` |
| RDS project path | `<TO CONFIRM>` |
| repo location on RDS | `$HOME/PNPL-2026` (assumed by `pbs/*.pbs`) |

⚠️ **Never hard-code any of these into code.**

| | login node | compute node |
|---|---|---|
| outbound internet | yes | **no** |
| heavy compute | **no** | yes |
| use it for | `prepare`, `git`, `qsub`, tailing logs | training, sweeps |

## Submitting

```bash
cd "$HOME/PNPL-2026"
qsub pbs/train.pbs          # one run
qsub pbs/sweep.pbs          # array job, one task per line of configs/sweep.txt
qstat -u "$USER"            # what is queued or running
qdel <jobid>                # cancel
```

⚠️ **`configs/sweep.txt` does not exist yet.** `sweep.pbs` reads line
`$PBS_ARRAY_INDEX` from it and errors if the line is empty.

## Job sizing

Current request: `select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=RTX6000`,
`walltime=08:00:00`.

⚠️ **Verify `gpu_type` and the cpus/mem/walltime combination against Imperial's
current job-sizing table.** CX3 accepts only combinations matching an allowed
job class; anything else is rejected or queues indefinitely with no explanation.

⚠️ **`num_workers` must match `ncpus`.** A mismatch either starves the GPU or
wastes half the allocation.

## Watching a running job

PBS copies the `-o` file back only **when the job exits**, so it cannot be
tailed. Both scripts `tee` to a second file on RDS, which does update live:

```bash
tail -f logs/<jobid>.live.log              # train.pbs
tail -f logs/<jobid>.<index>.live.log      # sweep.pbs array task
```

The job prints its own live-log path as its first line.

## Walltime is a kill, not a failure

Both scripts pass `--resume`, which continues from `last.ckpt`. An eight-hour
overrun means **resubmitting the same script**, not starting over.
