# Training on all the data

Asked for on 2026-10-08, for the final runs: everything subject 0 recorded,
and every other subject LibriBrain100 has. Two switches, both off by
default, so every other run in trial 18 trains exactly as before.

| switch | adds | how much |
|---|---|---|
| `train_partitions: [train, validation, test]` | subject 0's 3 validation and 3 test recordings (Sherlock1 ses-11/12 run 2, MOCHATIMIT ses-2/3, TheMoth ses-29/30) | 6 of 162 recordings, ~4% |
| `others_cache_dir: data/trial18_others` | the 32 broad-track subjects, Sherlock1 ses-11 and ses-12 run 1 each | 64 recordings, ~23 h, against subject 0's ~80 h |

## What changes

- **The others' cache** (`cli/build_others.py`, `pbs/trial18_build_others.pbs`):
  the same steps as subject 0's (`data_processing/build_reference.py`,
  `build_others`). Only recordings pnpl's manifest lists; keys carry
  `sub-<id>`; subject 0's channel order is required and its positions
  copied. Subject 0's cache is not touched.
- **Data** (`data/dataset.py`): `ReferenceRecordings` takes several
  partitions and the others' records; each token carries its `subject`
  (0 for subject 0). The others' sentences are shaped as subject 0's are
  (G2: whole sentences, a tenth as lone 1 s words). Validation and scoring
  stay subject 0's own partitions.
- **Model** (`models/merger.py`): one subject layer per subject. Subject 0
  keeps `weight`, the single-subject parameter, so every earlier checkpoint
  loads strictly; the others' maps are `others` [n-1, 270, 270], identity at
  start. Prediction passes no subject: subject 0's map.
- **Selection** (`cli/train.py`): validation is not clean once it is in
  training, or once the others -- who heard those same two Sherlock
  sessions -- are. Such a run trains a fixed number of epochs (`epochs_from`:
  the mean best epoch count of earlier runs, read off their
  evaluation.json), no early stopping, and best.ckpt is the last epoch. Its
  evaluation.json says `"selection": "... NOT clean"`: only the board judges
  it.

## Why each part is built this way

- Others' sentences shaped like subject 0's, not served word by word: as
  single words they would outnumber subject 0's sentence items several to
  one in every batch. The cost: the transformer sees those two sessions'
  ~600 sentences 33 times an epoch and can learn them by heart -- on the
  holdout that text never comes back, so it is wasted capacity, not a
  leak; but it is the kind of training-text prior that has lost on the
  board before. If an all-data run loses, this is the first suspect.
- A per-subject map, not one shared: different heads sit differently in
  the helmet; the shared rest of the network sees one space.

## Running it

From the repo root on CX3, after `git pull`. Once, on a LOGIN node (downloads):

```bash
uv run python -m src.pnpl2026.cli.prepare --partition validation   # all subjects
uv run python -m src.pnpl2026.cli.prepare --partition test
qsub pbs/trial18_build_others.pbs                                  # CPU, ~1 h
```

Then G2 on everything, 3 seeds (`configs/trial18/all_g2.yaml`, runs.txt lines
13-15; the job stages both caches to the node):

```bash
qsub -J 13-15 pbs/trial18.pbs
```

For a final run of another recipe: copy `all_g2.yaml` onto it -- its three
switches and an `epochs_from` pointing at that recipe's own runs.

## What was checked (2026-10-08, laptop, short runs)

- two fake broad-track subjects (synthetic recordings under real manifest
  names, `sub-1`/`sub-2` Sherlock1 ses-11/12 run 1): the others' cache built
  4 recordings of 2 subjects, subject 0's recording skipped, pnpl's own
  partition kept beside "train", positions copied
- data: subject 0's train, validation and test plus both others in one
  training set; tokens labelled 0, 1 and 2 (5,740 / 1,037 / 1,080)
- subject layer, distinct primes: each subject's map applied to its own
  windows, gradients to all three maps and the input; no subject equals
  subject 0; a single-subject layer refuses subject 2
- compatibility: a trial 17 network's state loads strictly into trial 18's;
  into a three-subject one only `encoder.subject.others` is new
- smoke, all switches on: 2 others joined; `epochs_from` read [1, 3] -> 2
  epochs; the warning printed; training stopped at exactly 2 with no early
  stopping; best.ckpt is the last epoch with `n_subjects` 3 saved; the
  evaluation flagged; the checkpoint predicted the real holdout files (960
  rows)
- the job script's syntax (`bash -n`), with the others' cache checked in the
  preflight and staged beside subject 0's

## Results

12 of the 32 broad subjects are published (pnpl's repositories lack 13-32,
2026-10-09): 24 recordings, 9.7 h, 32,064 in-vocabulary words -- about +10%
on subject 0's own. G2 on everything (3 seeds): local columns not clean
(ref test 68.56, FRAG 32.18); on the holdout it agrees with G2 more than any
other recipe (top-1 63%, top-10 82%; its own seeds 81%).

**Board (2026-10-09): a clear loss -- dropped.** Alone, `trial18_all_g2_ens_ts_lone`
scored **54.79** against G2's 63.41 (-8.6). In G2's place in the 67.00 mix it
scored 67.003, no change. Not separable from here: the others' two sessions
repeated 13 times an epoch (the memorisation risk named above), the fixed
epoch count with no selection, or other brains in the shared encoder. Do not
use for the final runs.
