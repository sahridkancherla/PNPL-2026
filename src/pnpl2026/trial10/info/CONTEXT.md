# trial10/ — Variant F: the d'Ascoli reference, reproduced

The competition paper's 73.23% BAcc@10 reference rebuilt: 306 sensors at
50 Hz, a position-attention merger, a 10-layer conv encoder, a 16 × 1024
sentence transformer over every word, D-SigLIP onto t5-large. Self-contained;
it has its own cache and imports from no other trial.

## Router

| where | what |
|---|---|
| `variant-f.md` | **start here** — why 46% vs 73%, what was built, the queue, how to run it, what was checked |
| `trial12-scaling.md` | trial 12: F1 scaled up, seven seed-0 sizes, run with this code (`configs/trial12/`) |
| `../data_processing/build_reference.py` | the cache: all three pnpl partitions, per-recording scaler, positions, PCA |
| `../data/` | sentences as token sequences under two protocols, reference and holdout |
| `../models/` | `merger.py`, `encoder.py`, `transformer.py`, `net.py` |
| `../losses.py`, `../lit.py` | D-SigLIP; retrieval metrics; `load_trained` rebuilds from a checkpoint alone |
| `../config.py` | `Trial10Config`; every default is the paper's |
| `../cli/` | `build_reference`, `build_word_embeddings`, `train`, `evaluate`, `collate`, `predict` |
| `../../../../configs/trial10/` | `trial10.yaml` is F0; one file per run; `runs.txt` |
| `../../../../pbs/trial10*.pbs` | build (CPU), embed (CPU), the five-line array (RTX6000, fp16) |
| `../../trial7/info/variant-d.md` | where the holdout protocol's numbers were measured |

## The order

```
prepare validation + test -> trial10_build -> trial10_embed -> qsub -J 1-5 trial10 -> collate -> predict
```
