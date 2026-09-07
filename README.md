# PNPL 2026 — deep track

MEG word decoding for the PNPL 2026 competition: classify a single word from a
single trial over a fixed 50-word vocabulary. Scored on BAcc@10, so chance is
20%.

**Start with [REPO.md](REPO.md)** — it is the map of everything here.
[CLAUDE.md](CLAUDE.md) holds the goal and the working rules;
[FORMAT.md](FORMAT.md) holds the competition itself — metric, data, label
space, submission format, and how to get onto the HPC.

## Setup

```bash
uv sync                                       # environment from uv.lock
uv run python -m pnpl2026.cli.prepare         # download the data cache
```

`prepare` must run on a login node — HPC compute nodes have no outbound
internet.

## Train

```bash
uv run python -m pnpl2026.cli.train --config configs/base.yaml configs/deep.yaml
```

On the HPC, submit `pbs/train.pbs` instead; it tees a live log into `logs/`
that you can `tail -f` while the job runs.
