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
| `CLAUDE.md` | the competition, the data, and the working rules — including that **Claude never commits**. |
| `handoff-docs/` | live handoff docs for long-running work. Not committed — local working notes. |
| `LICENSE` | MIT. |
| `.claude/` | Claude's configuration — rules, skills, settings. |
| `src/pnpl2026/` | the package. Shared machinery at its root; one subpackage per trial. |
| `configs/` | YAML run configs. `base.yaml` is defaults; a second file overrides it. `trial6/` to `trial17/` hold one file per ablation run; `trial12/` runs trial 10's code; `trial13/leaderboard.csv` records every leaderboard score, `trial14/ensembles.csv` every submitted ensemble's members. |
| `pbs/` | PBS job scripts for the Imperial HPC. `train.pbs`, `sweep.pbs`, `trial5.pbs`, `trial6.pbs`, `trial7_build.pbs`, `trial7.pbs`, `trial8.pbs`, `trial9.pbs`, `trial10_build.pbs`, `trial10_embed.pbs`, `trial10.pbs`, `trial11.pbs`, `trial12.pbs`, `trial13.pbs`, `trial13_calibrate.pbs`, `trial14.pbs`, `trial14_calibrate.pbs`, `trial15.pbs`, `trial16_embed.pbs`, `trial16.pbs`, `trial17.pbs`. |
| `runs/` | checkpoints and per-run logs. Gitignored except `.gitkeep`. |
| `logs/` | PBS job output and live logs. Gitignored except `.gitkeep`. |
| `submissions/` | generated submission CSVs. Gitignored except `.gitkeep`. |
| `data/` | downloaded LibriBrain cache. Gitignored entirely; created by `cli.prepare`. |
| `tutorial.ipynb` | the competition's own walkthrough notebook, kept as reference. |
| `pyproject.toml`, `uv.lock` | dependencies. `uv` manages the environment. |
| `.gitignore`, `.gitattributes` | bulk-data exclusions; LF endings for `.pbs`/`.sh`. |

## Trials

`src/pnpl2026/trial1/`, `trial2/`, … — one subpackage per complete attempt,
each with its own `data_processing/`, `models/`, `data/` and `cli/`. **Trials do not import from
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
| `configs/sweep.txt` | one config path per line; read by `pbs/sweep.pbs`, which will fail without it. |
