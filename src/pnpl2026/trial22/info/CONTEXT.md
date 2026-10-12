# Trial 22 — router

The Route to 73 (`handoff-docs/route-to-73-2026-10-12.md`), built after the
organisers confirmed on 12 Oct that the holdout may be used in any way. All
39 other subjects heard subject 0's 156 holdout sentences (the repeat
check), so change 1 -- their recordings as extra votes -- is live. A copy of
trial 21's code (both families, prediction, ensembling, hub correction).

| want | go to |
|---|---|
| each change, its tool, the decision rules, commands, checks, results | `route-to-73.md` |
| the repeat check and the match table | `../cli/repeat_check.py` |
| another subject's files, its subject layer | `../subjects.py`, `../cli/predict.py --subject` |
| MCCA into subject 0's sensor space; the virtual holdout | `../crosssubject.py`, `../cli/align.py` |
| other subjects' votes added to subject 0's rows | `../cli/cross_votes.py` |
| transduction (TransCLIP-style) | `../transduction.py`, `../cli/transduce.py` |
| correction variants (hubw, emsk, hubpos, quota, qbnorm) | `../corrections.py`, `../cli/correct.py` |
| local checks on uneven draws of held-out dumps | `../local_checks.py` |
| jobs and lists | `pbs/trial22_align.pbs`, `pbs/trial22_predict.pbs`, `configs/trial22/` |

Trial 21's adaptation work (A0, A2, the hybrid B) and its results:
`src/pnpl2026/trial21/info/holdout-adaptation.md`.
