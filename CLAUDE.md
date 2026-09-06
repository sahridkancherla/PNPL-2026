# CLAUDE.md

**What this project is trying to do, and the rules for doing it.**

The repo map is `REPO.md` — where things live is recorded there, not here.
Findings, measurements and per-feature state go in `handoff-docs/`.

---

## The goal

**PNPL 2026 Competition, deep track** — within-subject MEG decoding on subj00.

Classify a **single word** from a **single trial**, over a fixed 50-word
vocabulary. Single-trial is the whole difficulty: there is no trial averaging
to lean on, so every window must carry its own evidence.

- **Metric: BAcc@10** — macro-averaged Recall@10 over the classes present.
- **Chance is 20%** (`metric_k / n_classes`), not 2%.
- Submission is a CSV written by `pnpl`'s `write_submission`, in the row order
  `LibriBrainCompetitionHoldout` yields.

The immediate aim is a pipeline that trains end to end on the HPC and produces
a valid submission, so that model choices can be measured rather than argued
about. `cnn_baseline` is the reference point, not the destination — it exists
to establish what a plain convnet gets, and anything new is judged against it
on the same session split.

---

## Settled facts — do not re-derive

- **BAcc@10 is the scored metric.** BAcc@1 is logged as a tiebreaker only. A
  BAcc@1 of 0.05 is not "broken"; read every number against the right floor.
- **PyYAML implements YAML 1.1**: `lr: 1e-5` loads as the *string* `"1e-5"`.
  `config._coerce` catches it, but write `1.0e-5` in YAML regardless.
- **`/data/` in `.gitignore` must keep its leading slash.** Unanchored `data/`
  also matches `src/pnpl2026/data/` and silently drops the package from git.
  This has already happened once.
- **Never reorder holdout predictions.** `iter_windows()` yields the canonical
  row order and the host builds its label CSV from the same iteration. Sorting
  or shuffling breaks alignment with no error and no warning.

## Data and storage

- **All bulk data, checkpoints, logs and submissions stay out of git** — under
  `data/`, `runs/`, `logs/`, `submissions/`, all gitignored.
- **Never hard-code an absolute path**, in code or in prose. Paths derive from
  `REPO_ROOT` in `config.py`; one machine's path is broken on the next.
- `cli.prepare` downloads the cache and **must run on a login node** — HPC
  compute nodes have no outbound internet.

## Compute — a hard constraint, not a courtesy

Calibrated for the Imperial CX3 job class currently requested in `pbs/`.
Recalibrate the numbers for other hardware; the principles hold.

- **Verify `gpu_type` and the cpus/mem/walltime combination against Imperial's
  current job-sizing table before submitting.** CX3 rejects or indefinitely
  queues jobs that match no allowed job class.
- **`num_workers` must match `ncpus` in the PBS script.** The loader is the
  bottleneck; a mismatch wastes the allocation.
- **Walltime is a kill, not a failure.** `train.pbs` passes `--resume`, so an
  overrun means resubmitting the same script, not starting over.
- **PBS only copies its `-o` file back when the job exits.** To watch a running
  job, tail the live log the script tees into `logs/`.
- **State expected duration and load before launching anything long.**

## Experimental discipline

- **Split by whole session, never randomly.** Neighbouring windows from one
  session landing in both halves makes validation read far above the
  leaderboard. Session sizes vary ~30x, so size the split by *fraction of
  samples*, not session count.
- **Log training metrics alongside validation metrics.** Without the gap you
  cannot tell underfitting from overfitting, and so cannot choose between
  adding capacity and adding regularisation.
- **Record negative results.** They are the cheapest thing here to lose and the
  most expensive to rediscover.

## Submission

- **No Kaggle credentials are configured on this machine** (`~/.kaggle/` is
  empty). Submission is manual. **Do not claim anything was submitted.**
- Validate row count, ordering and finiteness before writing. `predict.py`
  already checks row count — do not remove that check.
