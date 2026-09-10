# Repo map

**The navigation bar for this repo.** Every top-level entry, one line each.
This file answers *"where do I go?"* — nothing else. It deliberately does not
explain what is inside a folder; open that folder's own `info/CONTEXT.md` or
`README.md` for that.

Keep this file current. A stale map is worse than none.

## Top level

| entry | what it is |
|---|---|
| `README.md` | front door for humans. Short; points inward. |
| `REPO.md` | this file — the map. |
| `CLAUDE.md` | the goal and the working rules. No content, no navigation. |
| `FORMAT.md` | the competition: metric, data, label space, submission, HPC access. |
| `LICENSE` | MIT. |
| `.claude/` | Claude's configuration — rules, skills, settings. |
| `handoff-docs/` | one document per feature, updated in place. Current state, not history. |
| `src/pnpl2026/` | the package. Shared machinery at its root; one subpackage per trial. |
| `configs/` | YAML run configs. `base.yaml` is defaults; a second file overrides it. |
| `pbs/` | PBS job scripts for the Imperial HPC. `train.pbs`, `sweep.pbs`. |
| `runs/` | checkpoints and per-run logs. Gitignored except `.gitkeep`. |
| `logs/` | PBS job output and live logs. Gitignored except `.gitkeep`. |
| `submissions/` | generated submission CSVs. Gitignored except `.gitkeep`. |
| `data/` | downloaded LibriBrain cache. Gitignored entirely; created by `cli.prepare`. |
| `tutorial.ipynb` | the competition's own walkthrough notebook, kept as reference. |
| `pyproject.toml`, `uv.lock` | dependencies. `uv` manages the environment. |
| `.gitignore`, `.gitattributes` | bulk-data exclusions; LF endings for `.pbs`/`.sh`. |

## Handoff docs

Two folders, split by feature size. A feature spanning many days or sessions
goes in `big/`; one that finishes inside a single session goes in `small/`.
`small/` should be much the rarer of the two.

## Trials

`src/pnpl2026/trial1/`, `trial2/`, … — one subpackage per complete attempt,
each with its own `data/`, `models/` and `cli/`. **Trials do not import from
one another**; code is copied rather than shared, so a finished trial stays
reproducible when the next changes the window, the rank or the sample rate.

Shared, trial-independent machinery — `metrics.py`, `config.py`, `data/vocab.py`
— stays at the package root. The folder name is `trial1`, not `trial-1`: a
hyphen is not importable.

## Not created yet

Planned homes, listed so the first person to need one puts it in the right
place rather than inventing a second.

| entry | for what |
|---|---|
| `tests/` | pytest suite. Nothing automated exists yet. |
| `info/` | per-folder prose, `CONTEXT.md` first. Add one to a folder when it earns prose. |
| `configs/sweep.txt` | one config path per line; read by `pbs/sweep.pbs`, which will fail without it. |
