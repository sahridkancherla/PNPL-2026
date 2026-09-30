# Trial 2 run — STOPPED at epoch 10, 2026-09-25

**⚠️ LIVE DOC. Read this first after any compaction.** Lives in
`handoff-docs/`, not `info/` — repo-level prose does not go in an `info/`. Everything needed to
pick the run back up without the conversation that started it.

---

## 1. The vision, in Kamil's words

> "build a mix between a convnext and a transformer"
> "we will do until epoch 30"
> "go light on the training - i wont be there to monitor"

PNPL 2026 deep track: single-word decoding from single-trial MEG, `sub-0`,
50-word vocabulary, scored on **BAcc@10 where chance is 20%**.

---

## 2. Status: STOPPED

Training was stopped by Kamil at epoch 10 ("scrap the learning"). Overfitting
had set in and the laptop was about to run out of battery.

| | |
|---|---|
| Best checkpoint | `runs/trial2/best.ckpt` — **epoch 6, BAcc@10 0.3470** |
| Resume point | `runs/trial2/last.ckpt` — epoch 10 |
| Submission | `submissions/trial2.csv`, already generated from epoch 6 |
| Log | `logs/train_trial2.log` (earlier phases archived alongside) |

**To resume** (nothing is lost — `--resume` continues from epoch 10):
```bash
PYTHONPATH=src nohup .venv/bin/python -u -m pnpl2026.trial2.cli.train \
  --config configs/base.yaml configs/trial2.yaml configs/laptop.yaml --resume \
  >> logs/train_trial2.log 2>&1 &
```
⚠️ Write the new PID to `runs/trial2/train.pid`. Never check liveness with
`pgrep -f trial2.cli.train` — it matches the caller's own shell wrapper.

## 3. Results so far

Chance = 20.0%. 21st on the leaderboard = 64.5%, 1st = 74.7%.

| epoch | trial 1 | trial 2 |
|---|---|---|
| 0 | 24.9% | 26.7% |
| 1 | 27.9% | 29.4% |
| 2 | 28.6% | 29.9% |
| 3 | — | 33.1% |
| 4 | — | 33.2% |
| 5 | — | 32.9% |
| 6 | — | **34.7%** ← best |
| 7 | — | 34.2% |
| 8 | — | 34.1% |
| 9 | — | 34.6% |
| 10 | — | stopped mid-epoch |

**Epochs 6–9 sat inside the noise band of each other (±1.2 pts) — four epochs
with no measurable gain.** Meanwhile train loss kept falling (3.45 → 3.41 →
3.37) against a flat val loss of 3.69. That combination is overfitting, not
the step pattern: the run had genuinely stopped improving.

**Trial 2 beat trial 1 at every matched epoch.** Trial 1 is parked at epoch 2
(`runs/trial1_laptop/last.ckpt`) and is NOT a converged baseline.

⚠️ **Progress comes in steps, not smoothly**: +2.8, +0.5, +3.1, +0.2, −0.4,
+1.8. The noise band on a single epoch difference is **±1.2 points**, so flat
epochs are statistically zero, not a ceiling. *A plateau was called twice on
this run and was wrong both times.* Do not call one on fewer than 4 flat
epochs. BAcc@1 rose monotonically throughout (2.8% → 5.5%) and is the cleaner
progress signal.

---

## 4. Decisions, and why

| decision | why | rejected |
|---|---|---|
| Rank 73 whitening | The spectrum has **no cliff** — every ratio from index 2 to 247 is 1.0–1.3×. But 73 is where the shrinkage floor (0.0093) exceeds the eigenvalue, so past it whitening stops being whitening | rank 96/105 — no evidence, and it costs a rebuild |
| Trial 2 reads trial 1's cache | Preprocessing identical, so the comparison is clean and it saves 25 min + 17 GB | separate cache |
| duty 0.25, 4 workers | Apple Silicon cannot cap GPU use; duty cycle is the only lever. **8 workers made it HOTTER** (cut idle waiting → GPU duty 17%→21%) | more workers for throughput |
| No LR schedule | Constant 1e-4 matches trial 1, so the comparison is not confounded | cosine — *wanted, see Deferred* |
| `max_epochs: 30` | Kamil's instruction | 50 |

⚠️ **The thermal auto-backoff does not work.** It reads `pmset -g therm`,
which only records once macOS clamps clock speeds — long after fans are
audible. It has never fired. Kamil's ears are the real sensor.

---

## 5. Deferred — logged, not fixed

1. **Cosine LR schedule.** Highest-value change available; constant LR is why
   progress is steppy. Deferred to keep trial1/trial2 comparable.
0. ⚠️ **NEXT RUN NEEDS MORE REGULARISATION, NOT MORE EPOCHS.** Trial 2
   plateaued at ~34.7% by epoch 6 and overfit thereafter, with dropout 0.1,
   weight decay 0.01 and ±0.1 s jitter — all light for 1.9M params on this
   much noise. Raise those before spending another night on epochs.
2. **`CLAUDE.md` says the shrinkage floor is "~12% of the last kept
   eigenvalue". Measured: 72%.** Fix when next editing that file.
3. **Lightning wipes `metrics.csv` on every resume.** Epoch history survives
   only in the archived `logs/train_trial2*.log`. Set a versioned CSVLogger.
4. **Trial 1 is not converged** (epoch 2). Any "trial 2 wins" claim is only
   matched-epoch, not at convergence.
5. **Top-1 collapse**: only 38/50 classes ever predicted (was 25/50 at
   epoch 4). Improving, but a direct drag on a *balanced* metric.
6. **Validation has 49/50 classes**, so BAcc averages over 49 — not exactly
   the leaderboard quantity.
7. `trial1/models/block.py` still has a `spans` field that nothing populates.

---

## 6. Submission

`submissions/trial2.csv` — 960 rows × 50 classes, generated from the **epoch 6**
checkpoint. Regenerate after training:

```bash
PYTHONPATH=src .venv/bin/python -m pnpl2026.trial2.cli.predict \
  --ckpt runs/trial2/best.ckpt
```

⚠️ The transform chain was verified **bit-identical** to the cache
(max abs diff 0.0). Do not restructure `prepare_batch` casually.

⚠️ **Nothing here can submit to Kaggle and nothing should try.** `~/.kaggle/`
is empty, the API returns 401. Kamil submits manually via the web UI.

---

## 7. If it breaks

| symptom | do this |
|---|---|
| Process dead, checkpoint intact | Re-run the command in §2 with `--resume`. Loses at most one epoch |
| Machine hot | Lower `duty_cycle` in `configs/laptop.yaml`, restart with `--resume`. 0.2 → ~22.8 min/epoch |
| `SIGKILL` used | **Check for orphaned dataloader workers** — they survive the parent and hold memory: `ps -Ao pid,command \| grep .venv/bin/python` |
| Run dir refuses to start | Guard is working; it prevents overwriting checkpoints. Use `--resume`, never `--force`, unless discarding results deliberately |

---

## 8. Rules that bite here

- **Kamil commits. Claude does not.** Everything is currently uncommitted:
  `trial2/` entirely, the `data-processing` → `data_processing` rename,
  `configs/{trial1,trial2,laptop,trial1_laptop}.yaml`, `block.py` edits,
  `info/`, and `.venv/`.
- **Never `qsub`/`ssh` the HPC** unless explicitly told.
- **Never reorder `iter_windows()` output** — row alignment is by construction.
