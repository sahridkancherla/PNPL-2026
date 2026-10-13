> The research note of 13 Oct 2026, as supplied; built as trial 23 -- tools, gates,
> deviations and results in `src/pnpl2026/trial23/info/research-note-13oct.md`.

# LibriBrain 2026 Deep Track: Which Cross-Subject and Transduction Ideas Could Plausibly Add 2+ Points (from 70.68 Toward 73)

The literature and your trial-22 results point the same way: the 39 other brains are unlikely to add 2+ points as raw votes or as a naive similarity graph, because what different brains share for the same audio is mostly acoustic envelope, silence and onset structure, which your mix already uses through timing. Your best remaining chances are (1) a timing-residualized, cross-fitted multi-subject graph or prototype vote, tested first on a labelled 13-subject testbed you already own (Sherlock book 1 sessions 11/12), and (2) fixing subject 0's sentence-boundary and session shift using subject 0's own short-sentence corpora (TIMIT and MOCHA-TIMIT).

## TL;DR

- **Build a labelled local testbed before spending submissions.** Subject 0's validation/test sessions are Sherlock book 1 chapters 11 and 12, and the Broad-track subjects heard the same Sherlock chapters (the competition paper fine-tunes Broad models on "half of the data from the Sherlock 1 Session 11 recording"). Use those shared recordings to replay the holdout setup with labels. Gate P1 and P3 on it. Without it, every cross-subject idea is a blind guess on a board where a new seed set alone moves the score by 1.4.
- **Most likely to give 2+ points:** P6, a boundary/short-sentence specialist built from subject 0's TIMIT and MOCHA data, with a local check on those test splits. Next is P1 or P3 (pick one; they use the same information), with timing removed from the features, same-sentence pairs dropped and cross-fitting. My estimates are +0.5 to +2 for P6 and +0.3 to +1.5 for P1/P3. P2 (row-identification encoder), P4 (MEG-XL as feature extractor) and P5 (test-fitted bigram) are lower value and higher risk.
- **Avoid** anything that lets other subjects' responses or decoders see sentence timing or sentence context: deep sentence-level decoders, windows of 1 s or more, features not residualized on timing, and EM loops that feed votes back into their own targets. Also avoid un-shuffling the 39 subjects' isolated-word files with the stored seeds until the organisers confirm it is allowed. Treat any public jump under about 1.5 points as noise.

## Key Findings

1. **Averaging across subjects helps much less than averaging across contexts.** d'Ascoli et al. (2025) compared both on datasets with many subjects and repeats: "decoding performance increases substantially more rapidly when averaging over contexts than over subjects." Averaging predictions in general is strong ("Most datasets show a two-fold improvement"; on the Armeni data "the top-10 accuracy reaches close to 80% after averaging only 8 predictions").\[1\] But your 39 extra recordings are the weaker kind of averaging: same context, other brains.

2. **The signal shared across brains for the same audio is mostly acoustics and timing, not word identity.** In a 2026 occlusion study of an MEG speech-segment decoder that retrieves the correct 3 s segment among 1,005 candidates with 70.4% top-10 accuracy (arXiv 2608.01481), the largest contributions came from silence (Δr = 75.62), high loudness (60.77), vowels (38.70) and strong acoustic onsets (36.19). Word onsets contributed only 1.02. This is exactly why your Route 1 and Route 2 votes correlated r = 0.78 with subject 0's picks. Any cross-subject feature must have this timing and envelope part removed, or it will just re-vote the timing prior.

3. **Cross-subject single-trial MEG information is real but small, and grows with pooling.**
   - Ravishankar et al. (2021; 8 subjects reading the same story once) predicted each subject's MEG from the others at a 20v40 accuracy of 0.884 (pairwise mapping) and 0.773 (shared response model), against a chance of 0.5.\[2\]\[3\]\[4\] Each test there pools 20 samples, so this is not single-trial accuracy.
   - Denoising raised the left-temporal N400m correlation with surprisal from 0.089 to 0.266.\[2\]
   - CL-SSTER (Shen et al., 19 subjects, audiobook EEG, 30 s windows) reached inter-subject correlations of only 0.050, against 0.017 for CorrCA and 0.022 for raw sensors.\[5\]
   - Your own number, r = 0.151 for MCCA-mapped others predicting subject 0, fits this picture: real but weak per trial.

4. **Alignment is required, and when it works it can match within-subject accuracy.**
   - In MEG, Zhang et al. (2017) found inter-subject classification after M-CCA "close to that of intra-subject classification". Inter-subject classification on raw sensors "never did much better than chance".\[6\]\[7\]
   - In the fMRI hyperalignment tutorial data (PyMVPA), between-subject accuracy rose from 0.42 (anatomical alignment) to 0.62 after hyperalignment, against 0.57 within-subject.\[8\]
   - Pooled templates can beat a subject's own template: in EEG (Malekmohammadi et al., 2025), a model built from other listeners identified 1 of 22 one-minute passages at 74.96%, against 36.57% for subject-specific models.\[9\] That study needed at least about 20 s of signal, though, and only reached above chance at 5 s.\[9\]

5. **Graph transduction gains scale with feature-cluster quality.** Applied to remote-sensing models (RS-TransCLIP), TransCLIP gave "average gains ranging from 9.9% up to 17.1%"; on histopathology models it gave "over 6% on Quilt, 7% on CONCH and 10% on PLIP". ZLaP (Stojnić et al., CVPR 2024) outperformed the latest related methods for zero-shot CLIP across 14 common datasets. Both run on strong image embeddings where nearest neighbours usually share a class. Your own-feature graph gave only +0.4 to +0.9, which tells you subject 0's single-trial neighbourhoods are weak. A 39-subject graph helps only if it makes neighbourhoods measurably purer, and the testbed can measure that.

6. **Your confirmation-bias failures match the literature.** Arazo et al. (2020) show that naive pseudo-labelling "overfits to incorrect pseudo-labels". Soft labels help, and they found "mixup augmentation and setting a minimum number of labeled samples per mini-batch" (16 per batch in their setup) effective against it. Your -3.6 self-training result and any multi-round EM over 868 rows face the same risk. Co-training needs views that are conditionally independent given the label.\[10\] Sentence timing, shared by all 40 files, breaks that assumption unless you remove it.

7. **Your references and the data structure.**
   - The organisers' Deep reference scores 73.23% BAcc@10 (25.60% BAcc@1) on their own test split. They describe the holdout as "drawn from sources not revealed to participants in advance".\[11\]
   - Subject 0 has 5.3 h of TIMIT (4.9 h training), 1.1 h of MOCHA-TIMIT and 6.0 h of Moth podcasts.\[12\] TIMIT has 630 speakers across 8 dialect groups.\[13\] These short, separate sentences with real silences and many speakers are the closest thing in your training data to the holdout.

## Details: Ranked Proposals

| Rank | Idea | Expected public gain (estimate) | Chance of ≥2 points | Cost | Fits in one day? |
|---|---|---|---|---|---|
| 0 | Labelled 13-subject Sherlock testbed (enabler) | 0 directly; stops blind submissions | n/a | Low (hours) | Yes |
| 1 | P6: boundary/short-sentence specialist from subject 0's TIMIT+MOCHA | +0.5 to +2 | ~25% | Medium | Yes, if fine-tuning starts today |
| 2 | P1 or P3: timing-residualized, cross-fitted 39-subject graph or prototype vote | +0.3 to +1.5 | ~15% | Medium | Yes |
| 3 | P8a: Bayesian product of per-subject likelihoods (P3 in a different form) | Same as P3 | Same | Low once P3 exists | Yes |
| 4 | P7: isolated-word improvements (private board only) | 0 on public; maybe +0.5 to +1 private | Unknown | Medium | Partly |
| 5 | P2: row-identification contrastive encoder | +0 to +1 | ~5–10% | High | Barely |
| 6 | P5: test-fitted bigram / vocabulary-restricted LLM prior | -1 to +1 | ~5% | Low–medium | Yes |
| 7 | P4: MEG-XL as frozen feature extractor for others | 0 to +0.5 | <5% | High | Barely |
| 8 | P8b: session re-centering (Euclidean alignment) of subject 0's holdout | -1 to +1 | ~5% | Low | Yes |

### Step 0 (do first): the labelled 13-subject testbed

- **What to build.**
  - Subject 0's validation and test sessions are Sherlock book 1 sessions 11 and 12, per the LibriBrain100 paper: "sessions 11 and 12—which correspond to chapters 11 and 12—from the first Sherlock book are held out for validation and testing".\[13\]
  - The Broad cohort heard "Sherlock (2 chapters)", and the competition paper fine-tunes Broad models on Sherlock 1 Session 11.\[11\]\[13\] Confirm by matching word onsets that subjects 1–12's released files are the same audio. Session 11 is explicitly confirmed; session 12 is likely.
  - Cut those chapters into short pseudo-sentences that look like the holdout: runs of 3–8 consecutive vocabulary words with natural gaps.
  - Run the full pipeline on them (mix → transduction v1 → hub correction), then add each cross-subject method using the 12 other subjects' recordings of the same audio.
- **Why it matters.** It is the only place where you can measure, with labels, whether a cross-subject graph or vote adds information beyond timing.
- **Scaling to 39 subjects.** Run with 3, 6 and 12 subjects. If the gain grows roughly with the square root of the subject count, a 12-subject gain of g suggests roughly g × √(39/12) ≈ 1.8g at 39. Treat this as an upper bound, because the holdout's short vocabulary-only sentences differ from narration.
- **Pitfall.** Narration has fewer silences and longer words than the holdout, so P6-type boundary effects will not show up here. Use subject 0's TIMIT and MOCHA test splits for those.

### Rank 1 — P6: boundary/short-sentence specialist (subject 0's own data)

- **Evidence.**
  - 36% of holdout rows are first or last words.
  - Silence and acoustic onsets are among the strongest drivers of MEG speech decoding (arXiv 2608.01481: silence Δr = 75.62, acoustic onset 36.19).\[14\]
  - Your Sherlock-heavy training set (67.3 h of 78.6 h) is continuous narration from one or two narrators with similar Southern British accents.\[13\] The holdout is short separate sentences, possibly from a different speaker.
  - The d'Ascoli sentence transformer gave "another 50% performance boost" from context,\[1\] but at sentence edges that context is truncated by silence.
- **Recipe.**
  1. **Diagnose.** On subject 0's TIMIT core test (192 utterances) and MOCHA test splits, restricted to vocabulary words, compute BAcc@10 separately for first, middle and last positions with your current mix.\[13\] Do the same on the holdout using label-free agreement between seeds as a proxy.
  2. **Build the specialist.** Fine-tune 3–5 of your best d'Ascoli-style members for a few epochs with TIMIT+MOCHA heavily upweighted (for example 50% of batches). Add explicit "first word / last word / gap before / gap after" tokens to the sentence transformer input. Keep the Sherlock data in the mix to avoid forgetting.
  3. **Apply it.** Route only first and last rows to the specialist, blended at a weight chosen on TIMIT/MOCHA validation. Middle rows keep the current mix.
- **Expected gain.** If boundary rows are 8–10 points worse than middle rows locally and the specialist closes about half of that, the average gain is about 0.36 × 4–5 ≈ +1.5 to +1.8. If there is no local gap, skip it.
- **Diagnostic.** The boundary-vs-middle gap on TIMIT/MOCHA test, and the specialist's gain on those rows with labels.
- **Failure mode.** TIMIT and MOCHA contain few tokens of your 50 vocabulary words, so local estimates are noisy. Pool validation and test, and use bootstrap confidence intervals.

### Rank 2 — P1 / P3: cross-subject information without the timing echo

P1 (affinity graph) and P3 (cross-fitted prototypes) use the same information: the stimulus-locked, word-specific part of the other brains' responses. Build the shared feature step once, compare both on the testbed, and submit the better one.

**Shared feature step (essential):**
- **Window.** Use -0.05 to +0.45 s from each word's own onset; your function words are short, so do not go longer. Band-pass 0.1–20 Hz and resample to 50 Hz (d'Ascoli used 0.1–40 Hz at 50 Hz).\[1\] Project each subject into your existing MCCA space and keep about 20–40 components. Use a short baseline from the first 50–100 ms.
- **Timing residualization.** Within each subject, regress every feature on a flexible timing design and keep the residuals. The design should include:
  - sentence position (one-hot), reversed position and sentence length;
  - log gap to the previous onset and log gap to the next onset, as splines;
  - an indicator for "previous gap was silence".
  
  These covariates are identical across all 40 files, so this removes exactly the part that encodes the timing prior. It also removes some real word information, such as duration, but your mix already has that.
- **Same-sentence exclusion.** Windows of neighbouring words overlap, so set within-sentence similarities to zero. Your graph already links only other sentences.

**P1 recipe (graph):**
- Per subject, compute the correlation between rows' residual vectors. Z-score each subject's 866×866 matrix.
- Average across the 39 others, weighting each subject by its split-half reliability. Do not include subject 0, so the graph stays independent of subject 0's decoders.
  - With 39 similar views, a reliability-weighted average is near-optimal for independent noise.
  - Similarity Network Fusion (Wang et al. 2014, cross-diffusion over K-nearest-neighbour graphs) and co-regularized spectral clustering (Kumar, Rai and Daumé 2011) are alternatives if subjects differ a lot in quality.\[15\]\[16\]\[17\]
- Keep the top k = 10 neighbours, symmetrize, and sharpen weights with a power of 3, as ZLaP does.
- Run label propagation F ← αSF + (1−α)Y with α in 0.7–0.9, seeded with the mix's soft posteriors Y before hub correction.
- Blend at a weight chosen on the testbed. Then apply hub correction as now.

**P3 recipe (cross-fitted prototypes):**
- Per other subject, split the 156 sentences into 5 folds, grouped by sentence.
- On 4 folds, fit soft class means μ_w = Σ q_iw x_i / Σ q_iw from the mix posteriors q, with shrinkage covariance (Ledoit–Wolf).
- Score the held-out fold by LDA log-likelihood.
- Sum the log-likelihoods over the 39 subjects with a temperature of about 1/√39 or tuned on the testbed. This is the "Bayesian joint inference" version (P8a).
- Add the result to the mix at a small weight.
- Do one pass only. A second EM round leaks each row's pseudo-label back into it through the other rows' prototypes.

**Label-free diagnostics on the holdout:**
1. **Split-half graph reliability.** Split the 39 subjects into two halves, build both graphs, and measure neighbour overlap against chance. Spearman–Brown gives the full-set reliability as 2r/(1+r). If overlap is near chance after residualization, stop.
2. **Echo check.** The new vote should correlate much less than 0.78 with subject 0's picks. It should correlate near zero with a timing-only decoder (your onset-gap voter, which reaches 31–41 BAcc@10).
3. **Neighbour-purity proxy.** On the testbed with labels, compute k-nearest-neighbour purity (the share of neighbours with the same word) against the base rate. Require at least 2× the base rate before submitting.

**Expected gain and failure mode.** +0.3 to +1.5. The main failure is that after timing removal the remaining word-specific signal per subject is too small. That is the outcome the d'Ascoli "subjects < contexts" result and your r = 0.151 make most likely.

### Rank 4 — P7: isolated words (private board only)

- The 92 isolated 1 s words make up about a fifth of each word's private rows, and the public board never sees them.
- **Low-risk option.** Ensemble more seeds of your isolated-word specialist. Average predictions over small onset jitters (±20 ms, the jitter range a speech-detection model used for robustness) and over sensor dropout.\[18\]
- **Rule risk.** The 39 other subjects' isolated-word files are shuffled per file with a stored seed. That shuffle looks designed to stop exactly the cross-subject alignment you want. Ask the organisers before using the seeds.
- **Matching without seeds is not realistic.** Cross-subject identification of naturalistic speech needed about 5–20 s windows even for 22 candidates (Malekmohammadi et al. 2025).\[9\] 1 s isolated words will not match reliably across subjects by response alone.

### Rank 5 — P2: row-identification / CL-SSTER-style encoder

- **Evidence.**
  - Segments are identifiable across subjects: Défossez et al. (2023) found the right 3 s speech segment with 41.3% top-1 and 70.7% top-10 accuracy among 1,363 candidates.
  - But that identity comes mostly from acoustics over the whole window, and CL-SSTER's absolute inter-subject correlations stayed small (0.050 on speech).\[5\]
  - An 866-way instance task with 40 examples each pushes rows of the same word apart, the opposite of what you need. It will also learn neighbour-word acoustics.
- **If you try it.**
  - Use a 0.5 s window and timing-residualized inputs.
  - Use other subjects' recordings of the same row as positives.
  - Exclude from the negatives any rows that share the mix's top-3 words; this is a mild, controlled echo.
  - Train on the 40 holdout files plus the 13-subject Sherlock chapters, so you can check word purity with labels.
- **Gate.** Use it only if it beats raw residual features on testbed purity.

### Rank 6 — P5: test-fitted bigram or vocabulary-restricted LLM prior

- A 50×50 bigram fitted by EM on about 700 transitions, from posteriors that are mostly wrong at top-1 (organisers' reference BAcc@1 is 25.60%), will mostly learn your own errors. Arazo et al. warn about exactly this.\[11\]\[19\]
- A general LLM prior restricted to the 50 words is more defensible than Sherlock text, but your three failed language-prior trials and the balanced metric (priors favour frequent words) argue for a very low weight.
- There is also a rules grey zone if the LLM has memorised the stimulus list.
- **Only submit** if a forward–backward rescoring at weight ≤0.1 improves the TIMIT/MOCHA test split locally.

### Rank 7 — P4: MEG-XL as a feature extractor

- MEG-XL is strong when fine-tuned: 57.3 BAcc@10 on LibriBrain with 13% of the training data and 63.0 with all of it.\[20\] d'Ascoli's model overtakes it after about 2.5 h of subject data.\[21\]
- Its input is 50 word windows of 3 s each, concatenated into 150 s.\[21\] That carries the whole sentence and timing structure, so it is a strong timing-echo risk.
- Its frozen masked-prediction features were not trained for word identity, and its 42.96 Broad reference needed fine-tuning.
- Use it only as an alternative feature set inside the residualized P1/P3 pipeline, and only if it wins on testbed purity.

### Rank 8 — P8b: session shift of subject 0 (re-centering)

- In Junqueira et al.'s systematic evaluation (J. Neural Eng. 2024), Euclidean alignment (re-centering each session's covariance) gave shared EEG deep models "an absolute improvement of 5.55% ... for the offline case and 4.33% in the pseudo-online" case across subjects.
- An online test-time adaptation study found it fell below the unadapted baseline in a cross-session setting.\[22\] Your BatchNorm re-estimation already damaged seed agreement.
- Test it only on subject 0's TIMIT and MOCHA test sessions (different days, different speakers) before any submission.

## Suggested Order for the 13 Remaining Submissions

Ground rules:
- One flipped row is worth about 0.2 points on common words and up to 2 points on rare ones. A new seed set alone moves the score by 1.4. So the public board can only confirm swings of about 1.5 points or more.
- Choose methods on the testbed and on TIMIT/MOCHA, and use submissions to confirm.

**Today (3 submissions; build the testbed first):**
1. P6 boundary specialist, routed to first and last rows only, if the local boundary gap is real.
2. P1 or P3 (whichever wins on the testbed) on top of 70.68, at the weight the testbed chose.
3. The 70.68 pipeline unchanged but with a fresh seed set. This measures the noise floor, so you can judge 1 and 2 against it.

**Tomorrow (10 submissions):**

4–5. P6 + P1/P3 combined, with two different seed sets. If both beat their single-method counterparts, the gain is likely real.

6. The P8a product-of-experts variant (if P3 was used) or the SNF variant (if P1 was used): one alternative only.\[23\]\[24\]

7. P7 isolated-word ensemble upgrade added to the best pipeline. It is invisible on public; this submission only checks that nothing broke.

8. P5 at weight ≤0.1, only if it passed locally; otherwise use the slot for P2 if P2 passed testbed purity.

9–11. Re-runs of the best combined pipeline with different seed sets, averaged into one larger ensemble. Averaging seeds reduces real prediction variance, not just board noise.

12–13. Final candidates. Pick the one with the best local evidence, not the single highest public score; the highest public score among many noisy tries is biased upward and will not carry over to the private board. If the platform allows two final selections, make one the safest locally validated pipeline.

## Things to Avoid

- **Anything that shows other subjects' data together with sentence context or timing.** That includes deep sentence-transformer decoders on other subjects (Route 1), "virtual subject" inputs to subject 0's sentence models (Route 2), windows of 1 s or more, MEG-XL's 150 s context, and features not residualized on position and gaps. All of these re-vote the timing prior. Your r = 0.78 echo is the warning.
- **Multi-round EM or self-training on 868 rows** where votes feed their own targets: confirmation bias (Arazo et al. 2020; your -3.6 result).\[19\] Use one cross-fitted pass.
- **Treating 39 votes as independent judges** (Dawid–Skene-style weighting) when they share stimulus timing. Correlated errors break that model.
- **Linking rows within the same sentence**, which overlap in time, and **position-driven links between first words**, whose onset responses look alike regardless of the word.
- **Inferring labels** from leaderboard probing or from looking up the sentence list. Also avoid un-shuffling the isolated-word files with the stored seeds without the organisers' explicit approval.
- **Re-trying approaches that already failed:** BatchNorm re-estimation, per-window normalization, bigger models, Sherlock-text bigrams, kNN memory over training instances, and the six per-word correction variants.

## Caveats

- **The gain estimates are my judgement, not measurements.** No published study tests multi-subject transduction for word classification with 50 classes, about 17 items per class and 39 views.
- **Several numbers come from summaries, not full texts.** Ravishankar et al.'s single-trial encoding gains appear only in a figure.\[2\] Michalke et al.'s 2023 M-CCA accuracies could not be retrieved. Zhang et al. (2017) report accuracies only in figures, on trial-averaged data.\[6\]
- **The testbed match is partly an assumption.** That subjects 1–12 heard exactly subject 0's session-12 audio is an inference from the published split description; check it by matching onsets.
- **The public board is a small, noisy sample.** Reaching 73 there could happen by luck without a real improvement, and a real +1 improvement could fail to show. The private board, which includes the isolated words, decides the ranking.

## Sources

1. [Towards decoding individual words from non-invasive brain recordings - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12658044)
2. [Frontiers | Single-Trial MEG Data Can Be Denoised Through Cross-Subject Predictive Modeling](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2021.737324/full)
3. [Single-Trial MEG Data Can Be Denoised Through Cross-Subject Predictive Modeling - PubMed](https://pubmed.ncbi.nlm.nih.gov/34858157/)
4. [Single-Trial MEG Data Can Be Denoised Through Cross-Subject Predictive Modeling - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8632362/)
5. <https://arxiv.org/abs/2402.14213>
6. [Inter-Subject Alignment of MEG Datasets at the Neural Representational Space](https://www.biorxiv.org/content/10.1101/096040v1.full)
7. [Inter‐subject alignment of MEG datasets in a common representational space - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6866831/)
8. [Hyperalignment for between-subject analysis — PyMVPA 2.6.5.dev1 documentation](http://www.pymvpa.org/examples/hyperalignment.html)
9. [Speech prediction of a listener via EEG-based classification through subject-independent phase dissimilarity model - Scientific Reports](https://www.nature.com/articles/s41598-025-12135-y)
10. [\[1304.5634\] A Survey on Multi-view Learning](https://ar5iv.labs.arxiv.org/html/1304.5634)
11. [The 2026 PNPL Competition: Word Classification and Efficient Cross-Subject Generalisation in LibriBrain100](https://arxiv.org/html/2609.03231v1)
12. [PNPL Competition](https://libribrain.com/)
13. [LibriBrain100: One Hundred Hours of Broad and Deep MEG Data for Neural Speech Decoding at Scale](https://arxiv.org/html/2608.25204)
14. [Interpretable MEG Decoding of Perceived Speech: Cortical Sources and the Stimulus Features That Drive Retrieval](https://arxiv.org/pdf/2608.01481)
15. [Community Detection in Multimodal Data: A Similarity Network Perspective](https://arxiv.org/pdf/2502.15993)
16. [\[PDF\] Co-regularized Multi-view Spectral Clustering](https://www.semanticscholar.org/paper/Co-regularized-Multi-view-Spectral-Clustering-Kumar-Rai/ba9233dd7c81765b7afb2cc1a6e5e9a075518d8c)
17. [GitHub - grisaitis/similarity-network-fusion: A Python implementation of similarity network fusion by Wang et al, 2014 (https://doi.org/10.1038/nmeth.2810) · GitHub](https://github.com/grisaitis/similarity-network-fusion)
18. [MEBM-Speech: Multi-scale Enhanced BrainMagic for Robust MEG Speech Detection](https://arxiv.org/html/2603.02255)
19. [\[PDF\] Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning](https://www.semanticscholar.org/paper/Pseudo-Labeling-and-Confirmation-Bias-in-Deep-Arazo-Ortego/35e8312d8bdcffb8e0c956d20d5a581cad1c1b8a)
20. [\[2602.02494\] MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](https://arxiv.org/abs/2602.02494)
21. [MEG-XL: Data-Efficient Brain-to-Text via Long-Context Pre-Training](https://arxiv.org/html/2602.02494v1)
22. [Calibration-free online test-time adaptation for ... - arXiv](https://arxiv.org/pdf/2311.18520)
23. [Similarity network fusion for aggregating data types on a genomic scale](https://www.nature.com/articles/nmeth.2810)
24. [\[PDF\] Similarity network fusion for aggregating data types on a genomic scale](https://www.semanticscholar.org/paper/Similarity-network-fusion-for-aggregating-data-on-a-Wang-Mezlini/bc5071d839c7f06d4cb5af9df9f116595b38755a)
