# Trial 23 — router

The 13 Oct research note (`handoff-docs/cross-subject-note-2026-10-13.md`):
cross-subject information without trial 22's timing echo, every idea gated
on a labelled Sherlock testbed first. A copy of trial 22's code (both
families, prediction, ensembling, transduction, hub correction).

| want | go to |
|---|---|
| each proposal, its tool, gate, deviations, commands' order, checks, results | `research-note-13oct.md` |
| the final nights: the 204M diagnosis, anneal, heads, tower, SAM, F0 fixed | `final-night-14oct.md`; where each member stands: `../cli/final_status.py` |
| anneal and head-only training, SAM, per-word scoring | `../cli/anneal.py`, `../perword/sam.py`, `../perword/evaluation.py` |
| another language model's target table; the tower weight | `../cli/build_lm_embeddings.py`; `--tower-weight` in predict and evaluate |
| night 2: SSL pretraining, TTA, the chained submit script | `../cli/pretrain_encoder.py`, `../cli/tent.py`, `pbs/night2.sh` |
| the testbed: shared audio check, labelled replica of the holdout | `../cli/testbed_check.py`, `../cli/testbed.py` |
| P1/SNF/P3/P8a: residual features, votes, holdout candidates | `../crossfeatures.py`, `../crossvotes.py`, `../cli/cross_graph.py` |
| P6: boundary features, fine-tune, diagnose and apply | `../boundary.py`, `../cli/finetune.py`, `../cli/boundary.py` |
| P7: lone-word TTA, the lone comparison | `../tta.py`, `../cli/predict.py`, `../cli/embed.py --lone-tta`, `../cli/lone_check.py` |
| P2: the row-identification encoder | `../row_encoder.py`, `../cli/row_encoder.py` |
| P5: the sequence prior (GPT-2 counts, EM, forward-backward) | `../sequence_prior.py`, `../cli/sequence_prior.py` |
| P4: MEG-XL frozen features | `../megxl.py`, `../cli/megxl_features.py` |
| P8b: session re-centering | `../recentre.py`, `../cli/recentre.py`, `--recentre` in predict and embed |
| dumps: chains38, isolated, onsets and table rows | `../cli/embed.py`, `chain_sequences` in `../data/sequences.py` |
| jobs | `pbs/trial23_{embed,finetune,predict,cpu,lm,megxl,anneal,lm_tables,tower,ssl,tent}.pbs`, `pbs/night2.sh` |
| lists | `configs/trial23/{members,fresh,lone,specialists,anneal,newmembers}.txt` |

Trial 22's changes and results: `src/pnpl2026/trial22/info/route-to-73.md`.
