# Trial 12 — scaling F1 up, a seed-0 size ablation

Trial 10's **F1** is the adopted winner: 5.85M parameters, **58.02 on the
leaderboard**, against F0's 49.71 for the same pipeline at 204M. On pnpl's test
partition the two matched (71.07 vs 70.42), so the test columns did not say
which size wins on the leaderboard, and nothing between 5.85M and 204M is
measured. Trial 12 asks where along that path the score peaks, and **which
axis matters**: transformer width, transformer depth, or encoder width.

**It has no code of its own.** Every size is already a config field of trial
10's model, so trial 12 is a config queue (`configs/trial12/`) run with
`pnpl2026.trial10.cli.train`, on trial 10's cache and t5 table. Nothing is
rebuilt.

## The queue — `configs/trial12/runs.txt`, seed 0

Layers: `base.yaml → trial10/trial10.yaml → trial10/hpc.yaml →
trial12/trial12.yaml → <run>.yaml`. `trial12.yaml` makes the base F1; each run
file changes one size. Parameter counts are from trial 10's `build_model`.

| line | run | transformer | encoder hidden | total | transformer | answers |
|---|---|---|---|---|---|---|
| 1 | `s0_control` | 4 × 256, 8 heads | 160 | 5.85M | 3.16M | F1 rerun: the control, and one rerun sample beside 58.02 |
| 2 | `w512` | 4 × **512**, 8 heads | 160 | 15.65M | 12.61M | width ×2 |
| 3 | `w1024` | 4 × **1024**, 16 heads | 160 | 53.06M | 50.39M | width ×4: F0's width at F1's depth |
| 4 | `d8` | **8** × 256 | 160 | 9.01M | 6.32M | depth ×2 |
| 5 | `d16` | **16** × 256 | 160 | 15.33M | 12.64M | depth ×4: F0's depth at F1's width |
| 6 | `wd_8x512` | **8 × 512** | 160 | 28.26M | 25.22M | both ×2 |
| 7 | `e320` | 4 × 256 | **320** | 10.60M | 3.16M | encoder width, brainmagick's 320 |

Already measured: F1 5.85M → 58.02; F0 204.2M → 49.71. **`w512` and `d16` are
the same size** (15.65M vs 15.33M), so they compare width against depth at
fixed parameters.

Held fixed everywhere: lr 1e-4, batch 64 sentences, attention dropout 0.1, and
everything else F1's. A fixed lr is a known confound for the largest runs,
which often want a lower one; it is noted, not varied.

## Reading it

- **Rank on the leaderboard**, sentence mode, as F1 was. The reference-protocol
  test columns are logged and reported, but F0 against F1 showed they do not
  separate sizes on the leaderboard. The `holdout` columns are not used: they
  mispredicted trial 10's leaderboard by 20–30 points.
- **Noise floor.** One seed each; one leaderboard score carries roughly 1.5–2
  points of standard error (an estimate from ~19 scored words per class, not a
  measurement). `s0_control` against F1's 58.02 gives one rerun sample.
- **Adopt a larger size only if it beats `s0_control` by 3 points or more.**
  Anything smaller is "no measurable difference"; confirm with seeds 1–2
  before adopting.
- Record the result either way.

## Running it

From the repo root on CX3. Trial 10's inputs must already be on RDS
(`data/trial10_reference/`, `data/word_embeddings_t5_all.npz`); the preflight
names whichever is missing.

```bash
qsub -J 1-7 pbs/trial12.pbs                 # one array wave
qsub -v RUN=<n> pbs/trial12.pbs             # resubmit line n after a walltime kill; --resume continues it

uv run python -m pnpl2026.trial10.cli.collate \
    --runs-list configs/trial12/runs.txt --name trial12      # runs/trial12_results.md
```

⚠️ `--name trial12` is required: without it collate writes trial 10's file
names and overwrites `trial10_results.md`.

Submissions: trial 10's `predict` needs only the checkpoint. As a CPU job,
for every finished run:

```bash
qsub -l select=1:ncpus=8:mem=32gb -l walltime=02:00:00 -N predict-12 -j oe -o logs/ <<'EOF'
cd "$HOME/PNPL-2026"
source "$HOME/.local/bin/env"
export PYTHONPATH="$PWD/src" UV_OFFLINE=1 HF_HUB_OFFLINE=1
for run in s0 w512 w1024 d8 d16 wd e320; do
    ckpt=runs/trial12_${run}_s0/best.ckpt
    [ -f "$ckpt" ] || { echo "MISSING: $ckpt"; continue; }
    uv run --frozen python -m pnpl2026.trial10.cli.predict --ckpt "$ckpt" --holdout-dir data/holdout \
        || echo "FAILED: $run"
done
EOF
```

CSVs land in `submissions/trial12_<run>_s0_in_context.csv`; upload them by
hand on the competition site.

Cost: every run sits between F1 and F0 in size, and both trained on an RTX6000
inside trial 10's 12 h walltime. All seven fit one array wave.

## What was checked (2026-10-02)

- every config loads through `load_trial10_config` and builds the model sizes
  in the table above, all in `16-mixed`
- **shape-check skill**, real dimensions, fp16 autocast on GPU, all seven:
  full and truncated windows, a batch of one, backward with no disconnected or
  non-finite gradient, eval-mode padding invariance (difference exactly 0)
- `s0_control` and `w1024` smoke-trained 2 epochs on synthetic recordings in
  pnpl's layout, under the run names the PBS script gives them; both scored
  themselves at the end; `s0_control` resumed from last.ckpt for a third epoch
- `collate --name trial12` wrote `trial12_results.md` with a parameter column
  and left trial 10's files alone; with no `--name` it writes trial 10's files
  under trial 10's heading, as before
- `pbs/trial12.pbs` passes `bash -n` and differs from `trial10.pbs` only in
  its header, its runs list and the extra `trial12.yaml` layer

## Results

Pending. Paste `runs/trial12_results.md` and the leaderboard scores here.
