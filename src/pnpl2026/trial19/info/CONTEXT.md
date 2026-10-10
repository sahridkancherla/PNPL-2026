# Trial 19 — router

The continuous family: each item -- a sentence span or a 1 s lone word -- is
encoded ONCE, and every word is read out at its onset. Built 2026-10-10 from
the 10 Oct architecture note, on a copy of trial 18's code; the per-word
network and Variant M are gone from this copy.

| want | go to |
|---|---|
| why, the design, what each piece reuses, the note's checks, the decision rule, how to run, what was checked, results | `continuous.md` |
| the network: stem, markers, time transformer, readout, heads | `../models/continuous.py` |
| the word tower and its description table | `../models/tower.py`, `../cli/build_word_descriptions.py` |
| the merger's sensor-type offset | `../models/merger.py` (`ChannelMerger`) |
| items: spans, tails, markers, time masking, phoneme labels | `../data/dataset.py` (`ContinuousItems`, `collate_items`) |
| per-sample phoneme labels | `../cli/build_phoneme_labels.py` |
| the losses, the heads' combination, loading a checkpoint | `../lit.py` |
| the run settings | `../config.py` |
| a 100 Hz cache (wave 2) | `../cli/build_reference.py`, `pbs/trial19_build.pbs` |
| predicting the holdout as whole items | `../cli/predict.py` |
| configs and the queue | `configs/trial19/`, `pbs/trial19.pbs` |

Hub correction, lone-word tools and the screen are trial 17's, copied
unchanged; see `src/pnpl2026/trial17/info/CONTEXT.md`.
