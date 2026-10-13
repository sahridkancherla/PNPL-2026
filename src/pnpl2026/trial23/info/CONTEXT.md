# Trial 23 — router

The 13 Oct research note (`handoff-docs/cross-subject-note-2026-10-13.md`):
cross-subject information without trial 22's timing echo, every idea gated
on a labelled Sherlock testbed first. A copy of trial 22's code (both
families, prediction, ensembling, transduction, hub correction).

| want | go to |
|---|---|
| each proposal, its tool, gate, deviations, commands' order, checks, results | `research-note-13oct.md` |
| the testbed: shared audio check, labelled replica of the holdout | `../cli/testbed_check.py`, `../cli/testbed.py` |
| P1/SNF/P3/P8a: residual features, votes, holdout candidates | `../crossfeatures.py`, `../crossvotes.py`, `../cli/cross_graph.py` |
| P6: boundary features, fine-tune, diagnose and apply | `../boundary.py`, `../cli/finetune.py`, `../cli/boundary.py` |
| P7: lone-word TTA, the lone comparison | `../tta.py`, `../cli/predict.py`, `../cli/embed.py --lone-tta`, `../cli/lone_check.py` |
| P2: the row-identification encoder | `../row_encoder.py`, `../cli/row_encoder.py` |
| P5: the sequence prior (GPT-2 counts, EM, forward-backward) | `../sequence_prior.py`, `../cli/sequence_prior.py` |
| P4: MEG-XL frozen features | `../megxl.py`, `../cli/megxl_features.py` |
| P8b: session re-centering | `../recentre.py`, `../cli/recentre.py`, `--recentre` in predict and embed |
| dumps: chains38, isolated, onsets and table rows | `../cli/embed.py`, `chain_sequences` in `../data/sequences.py` |
| jobs | `pbs/trial23_{embed,finetune,predict,cpu,lm,megxl}.pbs` |
| lists | `configs/trial23/{members,fresh,lone,specialists}.txt` |

Trial 22's changes and results: `src/pnpl2026/trial22/info/route-to-73.md`.
