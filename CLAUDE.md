# CLAUDE.md

**What this project is trying to do, and the rules for doing it.**

- The repo map is `REPO.md`.
- The competition, the data, the label space and HPC access are `FORMAT.md`.
  **Do not restate any of it here.**
- Findings, measurements and per-feature state go in `handoff-docs/`.

---

## The goal

**PNPL 2026 Competition, deep track** — within-subject MEG decoding on subj00.

Classify a **single word** from a **single trial**, over a fixed 50-word
vocabulary. Single-trial is the whole difficulty: there is no trial averaging
to lean on, so every window must carry its own evidence.

- **Metric: BAcc@10.** **Chance is 20%**, not 2%.
- Windows are `(306, 250)`; the label space is the 50 primary-vocabulary words.

The immediate aim is a pipeline that trains end to end and produces a valid
submission, so that model choices can be measured rather than argued about.
`cnn_baseline` is the reference point, not the destination — it exists to
establish what a plain convnet gets, and anything new is judged against it on
the same session split.

---

## Settled facts — do not re-derive

- **BAcc@10 is the scored metric.** BAcc@1 is logged as a tiebreaker only. A
  BAcc@1 of 0.05 is not "broken"; read every number against the right floor.
- **Targets come from `vocab.to_id()`, never from `task.label_info`.** The
  task's numbering is alphabetical over thousands of words; the competition's
  is frequency-ordered over 50. Mixing them scores at chance and raises
  nothing. `FORMAT.md` §3 has the detail.
- **Never reorder holdout predictions.** `iter_windows()` yields the canonical
  row order. Sorting or shuffling breaks alignment with no error.
- **PyYAML implements YAML 1.1**: `lr: 1e-5` loads as the *string* `"1e-5"`.
  `config._coerce` catches it, but write `1.0e-5` in YAML regardless.
- **`/data/` in `.gitignore` must keep its leading slash.** Unanchored `data/`
  also matches `src/pnpl2026/data/` and silently drops the package from git.
  This has already happened once.

## Adding a model

Every new model idea, start to finish:

1. **Write it.** A new file in `src/pnpl2026/models/`, a plain `nn.Module`
   taking `n_channels` and `n_classes`, mapping `(B, 306, 250)` → `(B, 50)`.
2. **Register it.** One line in `models/__init__.py`:
   `REGISTRY = {"cnn_baseline": SimpleWordCNN, "my_model": MyModel}`
3. **Write a config.** `configs/my_model.yaml`, containing **only the lines
   that differ** from `base.yaml` — `model`, `run_name`, and whatever
   hyperparameters the idea actually changes.

Run it against the same session split as the baseline, or the comparison means
nothing. **Run the `shape-check` skill before the first training run** of any
new model — a silent shape bug trains happily to exactly chance.

## Data and storage

- **All bulk data, checkpoints, logs and submissions stay out of git** — under
  `data/`, `runs/`, `logs/`, `submissions/`, all gitignored.
- **Never hard-code an absolute path**, in code or in prose. Paths derive from
  `REPO_ROOT` in `config.py`; one machine's path is broken on the next.
- `cli.prepare` downloads the cache and **must run where there is outbound
  internet** — see `FORMAT.md` §5 for why that matters on the cluster.

## Compute

> ⚠️ **Do not use the HPC unless explicitly told to.** No `ssh`, no `qsub`, no
> `scp`, no starting a job on your own initiative. Shared allocation, real
> queue, other people's time. `FORMAT.md` §5 is reference for when the
> instruction comes.

- **State expected duration and load before launching anything long**, wherever
  it runs.
- **Never run two compute-heavy jobs concurrently** on the same machine.
- Job sizing, walltime and log-tailing on the cluster: `FORMAT.md` §5.

## Experimental discipline

- **Split by whole session, never randomly**, and size the split by *fraction
  of samples*, not session count.
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
