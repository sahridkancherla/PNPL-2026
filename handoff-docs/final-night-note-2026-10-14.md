# LibriBrain 2026 Deep Track, Final Night: Which Optimiser, Model-Size and Architecture Changes Are Worth Running

Spend tonight on three low-risk things: (1) a short low-learning-rate "anneal" of your best checkpoint on holdout-shaped fragments, with weight averaging; (2) one or two new output heads on a different language model's middle layer, added as new ensemble members; and (3) at most one SAM run. Do not try L-BFGS, a smaller main model, a Conformer or state-space backbone, or a CTC or autoregressive decoder. The published evidence and your own board history both point the same way: what you learn from the target (heads, data shape, the final phase of training) transfers, and backbone changes do not.

## TL;DR
- **Q1 (optimiser and schedule):** No L-BFGS or second-order finish. Full-batch and second-order methods have a documented generalisation problem in deep nets. Yes to a short cooldown and anneal (decay to 0 over about 10–20% of the original run, on a mix weighted toward holdout-shaped fragments), followed by averaging the last few checkpoints, with BatchNorm statistics recomputed on *training* data only. SAM is the only other training-level change with large published gains on small data, and it costs 2× compute. Run it as a diversity member, not a replacement.
- **Q2 (smaller model):** Not as your main bet. On deep single-subject LibriBrain, the roughly 200M-parameter d'Ascoli pipeline beats 8.4M BrainOmni and 20M MEG-XL (both 63.0 BAcc@10) once there are more than about 2.5 h of training data. The organisers' 16×1024 reference scores 73.23, and your own widening of the sentence transformer gave +3.3. Your 204M loss is more likely an optimisation or recipe failure than too much capacity. A smaller model is worth at most one cheap diversity member, and only if GPU time is left over.
- **Q3 (architecture):** Only head-level changes get a yes. These are (a) the learned word tower plus a training-free WiSE-style blend with the frozen t5 anchor, and (b) extra heads on the middle layers of a different language model (GPT-2 or Llama-3.2-1B), which brain-encoding work consistently finds align best. Conformer or SSM backbones, CTC, autoregressive decoders and rerankers have no word-level BAcc@10 evidence, and they are backbone changes, which have lost for you every time.

## Key Findings

### Q1. Optimiser and schedule

**(a) L-BFGS or quasi-Newton finish. Verdict: no.**
- Bollapragada et al. (ICML 2018) say plainly that "since small batch sizes give rise to faster algorithms with better generalization properties, L-BFGS is currently not considered an algorithm of choice for large-scale machine learning". Full-batch quasi-Newton runs into "generalization problems", and the stochastic setting "makes quasi-Newton updating difficult".\[1\]
- Keskar et al. (ICLR 2017) found that large-batch methods "tend to converge to sharp minimizers", while small-batch methods "consistently converge to flat minimizers", and attribute this to gradient noise.\[2\] L-BFGS needs large, stable batches, which pushes it into exactly that regime.
- Buffelli et al. (arXiv 2411.07979) ran exact Gauss-Newton in reversible deep networks and found poor generalization. The networks stayed in a "lazy" regime in which "the neural tangent kernel (NTK) changes very little" and representations barely moved.\[3\]
- Three practical problems are specific to your model:
  - BatchNorm in train mode, dropout and in-batch-negative D-SigLIP all make the loss stochastic. That breaks L-BFGS line searches and curvature pairs.
  - A deterministic loss would mean switching BatchNorm to eval mode, turning dropout off and fixing negatives. That is a different objective from the one you trained on.
  - Your board already showed that further optimisation of the best model hurts ("more training" lost 3.8).

**(b) Learning-rate schedules and annealing. Verdict: yes to a short decay-to-zero anneal on target-like data.**
- Hägele, Bakouch, Kosson, Ben Allal, Von Werra & Jaggi (EPFL/Hugging Face, NeurIPS 2024, arXiv 2405.18392) found that a constant learning rate followed by a short cooldown "scales predictably and reliably similar to cosine". A cooldown of under 20% of total steps can match cosine, and a 1−sqrt cooldown shape works best. The paper also reports that stochastic weight averaging "yields improved performance along the training trajectory, without additional training costs".\[4\]
- MiniCPM (Hu et al. 2024, Tsinghua University and ModelBest, arXiv 2404.06395) introduced the warmup-stable-decay scheduler, "conducive to continuous training and domain adaptation". It observed that the loss "decreases dramatically" in a decay stage of about 10% of tokens, and it mixed high-quality, task-like data into that decay.
- Llama 3: annealing on a small amount of high-quality data "improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%". Gains on 405B were "negligible". Their data-value test annealed linearly to 0 with 30% new data and 70% of the default mix.\[5\]
- A 2026 small-model report (Index SLM, arXiv 2607.09885) found that WSD plus curated data in the final 10% gave the best score (38.10), beating either change alone. Cosine plus curated data scored slightly *below* plain cosine. The authors conjecture that the cosine tail leaves too little learning rate for the model to adapt to the new data.\[6\]
- How this applies to you:
  - You already measured +5 from holdout-shaped fragments, so you have real "target-like data". The decay phase is where such data has the most leverage.
  - Your −3.8 from "more training" was not an anneal. An anneal is short, starts from the early-stopped checkpoint, decays to zero, and is averaged.
  - LLM gains do not transfer one-for-one. Expect fractions of a point to a few points, not +24%.

**(c) Weight averaging (SWA, EMA, last-k). Verdict: yes, combined with the anneal, using BatchNorm recompute on training data only.**
- Izmailov et al. (UAI 2018) average the weights at the end of each of the last K epochs, then recompute BatchNorm statistics with one forward pass over the training data. Averaging weights approximates averaging predictions when the averaged models are close together.\[7\]\[8\] That holds for checkpoints taken from one low-learning-rate tail, not for independent runs.
- Busbridge et al. (NeurIPS 2023) is mainly a rule for scaling EMA momentum with batch size: ρ̂ = ρ^κ when the batch size changes by κ. In their supervised ImageNet runs the model EMA was only slightly better than the target model, about +0.2–0.3 top-1.\[9\]\[10\] Treat EMA as a small, stable gain.
- Your BatchNorm caveat: re-estimating BatchNorm on *test* data cost 4.5–8.5. That is a covariate-shift effect, not a reason to avoid averaging. Recompute on training windows that use exactly the training-time preprocessing. Alternatively, freeze BatchNorm statistics during the anneal so that the averaged weights sit next to compatible statistics.

**(d) Other training-level changes.**
- **SAM: yes, as one diversity run if it fits.**
  - Foret et al. (ICLR 2021) report CIFAR-100 error for WRN-28-10 of 16.5 vs 18.8 with SGD, and PyramidNet 15.4 vs 19.7.\[11\]
  - Chen et al. (arXiv 2106.01548) report ViT-B/16 on ImageNet going from 74.6 to 79.9. The gain is "proportional to the number of training images" in reverse: on 1/4 of ImageNet, SAM boosts ViT-B/16 by 14.4%. In contrastive training, ViT-B/16 went from 77.4 to 80.0.\[12\]
  - The catch is cost: each step takes twice the compute.\[13\] There is also no MEG evidence.
- **Soft targets or label smoothing for the contrastive loss: possible diversity member, low priority.**
  - SoftCLIP (AAAI 2024) gave +6.8 and +7.2 zero-shot ImageNet top-1. In its ablation, plain label smoothing gave +1.8 and intra-modal soft targets +4.0.\[14\]\[15\]
  - Your loss is D-SigLIP, which in d'Ascoli's preprint beat CLIP and SigLIP: top-10 accuracy 18.2 ± 0.76 vs 15.7 ± 1.89 vs 14.7 ± 0.09 (Fig. 8 of arXiv 2412.17829 v1).\[16\] Changing the loss is therefore not free.
- **Mixup: no.** I found no evidence for it in MEG or EEG word decoding.
- **Gradient clipping: keep it at 1.0, as MEG-XL does.** It is a stabiliser, not a source of gains.
- **Batch size: do not increase it.** d'Ascoli used 64\[17\] and MEG-XL used 50 words.\[17\]\[18\] Keskar's results argue against large batches, and SigLIP-style losses do not need large batches the way softmax CLIP does.

### Q2. Is a slightly smaller model worth trying?

**Verdict: no as the main bet. At most one cheap diversity member, last in the queue.**

| Evidence | Numbers | What it implies |
|---|---|---|
| MEG-XL paper (arXiv 2602.02494), LibriBrain, 100% of data, top-50 BAcc@10 | BrainOmni 8.4M **63.0 ± 0.1**; MEG-XL 20M **63.0 ± 0.4**; BBL 15M 49.9; LaBraM 5.8M 47.7; BIOT 3.2M 45.6; BioCodec 1.0M 41.9\[18\] | Among small pretrained models, size is not what separates them; pretraining fit is. |
| Same paper, compared with the d'Ascoli supervised model | On LibriBrain "both methods perform similarly until approximately 2.5 hours… after which their supervised method pulls ahead"; the authors describe the comparison as "20M vs 200M" | With about 50 h from one subject, the big supervised model wins.\[18\] |
| Organisers' reference | 16 layers × 1024, conv 270/320, lr 1e-5 cosine, patience 5: **73.23**; your shrunken version 69.3 | Capacity in the sentence transformer pays. |
| Your board | Sentence transformer 256→1024: **+3.3**; 204M "scale everything": −8 | The sentence transformer benefits from width. Scaling the whole backbone without retuning the learning rate failed. |
| d'Ascoli 2025 (Nature Communications) | Adding the sentence transformer gave a "50% performance boost" over the BrainModule CNN; performance grows log-linearly with data, with "no clear signs of diminishing returns"\[17\] | The bottleneck is signal and data, not a lack of regularisation from a smaller model. |
| Défossez 2023 ablations (arXiv 2208.12266) | Removing spatial attention cost 2.5 points; removing the subject layer cost 17% (in multi-subject training)\[19\]\[20\] | Removing components hurts; nothing there suggests narrowing helps. |
| Double descent (Nakkiran et al.) | Test-error peaks sit near the interpolation threshold and are "accentuated with label noise"\[21\] | Single-trial MEG is effectively very noisy-labelled. Shrinking toward the threshold can make things *worse*, not safer. |

Interpretation: your 204M loss is better explained by the recipe than by overparameterisation. The organisers train their large model at lr 1e-5 with patience 5, and d'Ascoli trains about 200M parameters at 1e-4 with batch 64.\[17\] A smaller model would mostly add diversity, and seeds of your existing recipe already provide that at known quality.

### Q3. Architectural changes

| Option | Type | Evidence | Verdict |
|---|---|---|---|
| (a) Learned word tower or prompt learning, with frozen anchor | Head | CoOp (frozen CLIP, learned context): around +15% average over hand-crafted prompts at 16 shots across 11 datasets (Zhou et al., arXiv 2109.01134). WiSE-FT (Wortsman et al., arXiv 2109.01903), interpolating zero-shot and fine-tuned weights: +4 to 6 pp under distribution shift while raising ImageNet accuracy by 1.6 pp, 2–23 pp on six further shifts, and +0.8 to 3.3 pp over standard fine-tuning on seven transfer datasets. Your board: +0.7 in the mix. | **Yes.** Add a training-free WiSE-style score blend between tower and anchor. |
| (b) Second head on a different language model | Head | d'Ascoli: the target model "has a mild effect on decoding performance".\[17\] Caucheteux & King 2022: middle layers (n/2 to 3n/4) best map onto both MEG and fMRI, in an inverted-U pattern. Antonello, Vaidya & Huth (NeurIPS 2023): fMRI encoding scales log-linearly with LM size (125M→30B, ∼15% across 3 subjects); OPT peaks about 3/4 of the way through, LLaMA earlier. Toneva & Wehbe 2019: middle layers best. Your board: another t5 layer gave +1.5–1.7 in the ensemble. | **Yes, top priority among the architecture options.** Expect similar single-model scores; the value is diversity. |
| (c) Conformer or SSM encoder | Backbone | MEGConformer won LibriBrain 2025 *phoneme* classification (65.8 F1-macro) and scored 88.9 on speech detection; neither is a word task. Its key fix was instance normalisation, which cost you 6–7. MEGState: phoneme decoding only, and the abstract gives no numbers. Brain2Qwerty v2 (*typing*): Conformer async encoder CER 0.25 vs Transformer *sync* encoder 0.23. The 0.28 in your brief is the Encoder-alone CER inside the full pipeline, not a Transformer baseline.\[22\]\[23\]\[24\] | **No.** Wrong task, wrong direction, and a backbone change. |
| (d) CTC, autoregressive decoder, reranker | Decoder | d'Ascoli suggest CTC only to remove the need for word timings, which you already have.\[17\] Brain2Qwerty v2's LLM cut WER (0.39 vs 0.55) but *raised* CER (0.31 vs 0.28) by producing fluent but wrong sentences. Jayalath 2025 LLM rescoring improved sentence BLEU, not per-word top-10 recall.\[17\]\[22\]\[25\] | **No.** These do not target BAcc@10. |
| (e) Readout variants | Head or near-backbone | d'Ascoli use a single-head attention pool.\[17\]\[26\] I found no published evidence for multi-head pooling or multiple windows, and your "longer readout" lost. | **Low priority.** Multi-head pooling is acceptable only as a spare-GPU diversity member. |

## Overnight recipes

**Shared protocol for any new member.**
- Use a fixed local check on holdout-shaped fragments only as a sanity filter (no worse than about 1 point below the parent), not for ranking. You have already seen that local validation does not rank recipes the way the board does.
- Add each new model to the 46-member ensemble as member 47 (or 48). Z-score its 50-way similarity scores per window exactly as for the existing members. Give it the same weight as one existing member (1/47), or the weight your previous best head received.
- Submit A/B: current ensemble vs ensemble + new member(s). Seed-to-seed board noise is 1.7 points, so only trust differences of about 1 point or more. Batch several new members into one submission rather than testing each alone.

**Recipe 1: anneal plus average (highest priority; run on both G2 and the 1024-width transformer variant).**
- Start from the *early-stopped* best checkpoint, not the over-trained one.
- Optimiser: AdamW with the same betas and weight decay. Learning rate starts at 0.1× your peak and decays to 0 with a 1−sqrt shape. Use no warmup.
- Length: about 10–20% of the original number of training steps (Hägele: under 20% is enough).
- Data mix: about 50% holdout-shaped fragments and 50% normal sentence batches. Llama 3 used 30/70, so stay within 30–50%.
- BatchNorm: put BatchNorm layers in eval mode (frozen running statistics) for the anneal. Keep dropout on.
- Save 5 evenly spaced checkpoints. Average the last 3 (SWA-style).
- Decide BatchNorm statistics in this order:
  1. Keep the frozen BatchNorm statistics.
  2. If the local check drops, recompute BatchNorm statistics with one train-mode pass over about 50k *training* windows using identical preprocessing.
  3. Never re-estimate on test data.
- Also keep the final un-averaged checkpoint as a fallback member.

**Recipe 2: new language-model target heads (high priority).**
- Targets: Llama-3.2-1B hidden state at the middle layer (about layer 8 of 16), and GPT-2-medium layer 12 of 24. Feed each word in isolation, as d'Ascoli did with t5, and mean-pool its sub-tokens.\[17\]
- Projection: a linear map to your embedding width with the same D-SigLIP loss.
- Training: use exactly the procedure behind your +1.5–1.7 second t5 head (same backbone recipe, same epochs, same early stopping).
- If time is short, train only the new head on the frozen G2 backbone for 3–5 epochs at learning rate 1e-3 (MEG-XL used 1e-3 for a fresh head and 1e-5 for the body).\[18\]
- Add each head as a separate member scored against its own vocabulary embeddings.

**Recipe 3: WiSE-style word-tower blend (free, no training).**
- For the continuous and word-tower model, score = α·sim(tower prototypes) + (1−α)·sim(frozen t5 anchor).
- Make α ∈ {0.5, 0.7} two members, or pick one at 0.5, the WiSE-FT default.\[27\]

**Recipe 4: SAM diversity member (only if GPUs remain).**
- Apply SAM to the G2 recipe with ρ = 0.05 (Foret's default) on top of AdamW (the base-optimiser form).
- Use the same epochs if the night allows, otherwise SAM only during a Recipe-1-style anneal.
- Compute the SAM perturbation with BatchNorm statistics frozen to avoid double-counting.
- Add as a separate member.

## Priority order for one night
First launch Recipe 1 on G2 and on the 1024-width variant in parallel. These are short runs on the evidence-backed lever (an anneal on target-shaped data plus averaging), and they cost hours, not a full retrain. On the remaining GPUs, train Recipe 2's Llama-3.2-1B middle-layer head and GPT-2 middle-layer head, because target diversity is the change that has already paid +1.5–1.7 for you. Compute Recipe 3's blends on CPU while those run. Start Recipe 4 (SAM) only if a GPU is idle. Skip L-BFGS, smaller main models, Conformer or SSM backbones, CTC and autoregressive decoders. For the board tomorrow, submit ensemble + all new members first, then the leave-one-group-out variants. Treat any difference under about 1 point as noise.

## Caveats
- None of the schedule, SAM or averaging evidence comes from MEG. The LLM annealing gains (+24% on GSM8k) come from a very different regime and will be much smaller here.
- I could not get d'Ascoli's Supplementary Fig. 3 (target-model comparison) with numbers; only the main-text statement of a "mild effect" was available.\[17\] The D-SigLIP vs CLIP vs SigLIP numbers come from the arXiv v1 preprint (Fig. 8) and may differ slightly in the published version.\[16\]
- The Brain2Qwerty v2 comparison in your brief (Conformer 0.25 vs Transformer 0.28) does not match the paper. The 0.25 vs 0.23 comparison mixes asynchronous and synchronous decoding on a typing task, and the 0.28 is the Encoder-alone CER inside the full pipeline.\[22\]
- MEGConformer's winning fix (instance normalisation) contradicts your board result (−6–7).\[24\] This is a further reason not to import that backbone.

## Sources

1. [A Progressive Batching L-BFGS Method for Machine Learning](https://arxiv.org/pdf/1802.05374)
2. [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/pdf/1609.04836)
3. [Exact, Tractable Gauss-Newton Optimization in Deep Reversible Architectures Reveal Poor Generalization](https://arxiv.org/pdf/2411.07979)
4. [\[PDF\] Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations](https://www.semanticscholar.org/paper/Scaling-Laws-and-Compute-Optimal-Training-Beyond-H%C3%A4gele-Bakouch/71990b0af0783c3c6656aa13697eac27a3c89ffa)
5. [The Llama 3 Herd of Models](https://arxiv.org/pdf/2407.21783)
6. [Index SLM Technical Report](https://arxiv.org/pdf/2607.09885)
7. [Averaging Weights Leads to Wider Optima and Better Generalization](https://www.alphaxiv.org/abs/1803.05407v3)
8. [Averaging Weights Leads to Wider Optima and Better Generalization](https://izmailovpavel.github.io/files/swa/poster.pdf)
9. [How to Scale Your EMA Dan Busbridge∗ Jason Ramapuram∗ Pierre Ablin∗](https://proceedings.neurips.cc/paper_files/paper/2023/file/e7681dd6fe16052433ab68cd1555bdc9-Paper-Conference.pdf)
10. [How to Scale Your EMA \[Quick Review\]](https://liner.com/review/how-to-scale-your-ema)
11. [Sharpness-Aware Minimization for Efficiently Improving ...](https://arxiv.org/pdf/2010.01412)
12. [when vision transformers outperform resnets without pre- ...](https://arxiv.org/pdf/2106.01548)
13. [Surrogate Gap Minimization Improves Sharpness-Aware Training](https://arxiv.org/pdf/2203.08065)
14. [SoftCLIP: Softer Cross-modal Alignment Makes CLIP Stronger Yuting Gao1\*](https://arxiv.org/pdf/2303.17561)
15. [(PDF) SoftCLIP: Softer Cross-modal Alignment Makes CLIP Stronger](https://www.researchgate.net/publication/369655531_SoftCLIP_Softer_Cross-modal_Alignment_Makes_CLIP_Stronger)
16. [\[2412.17829\] Decoding individual words from non-invasive brain recordings across 723 participants](https://ar5iv.labs.arxiv.org/html/2412.17829)
17. [Towards decoding individual words from non-invasive brain recordings - Nature Communications](https://www.nature.com/articles/s41467-025-65499-0)
18. <https://arxiv.org/pdf/2602.02494v2>
19. [Decoding speech perception from non-invasive brain recordings](https://arxiv.org/pdf/2208.12266)
20. [Decoding speech perception from non-invasive brain recordings](https://arxiv.org/html/2208.12266)
21. [Deep double descent: where bigger models and more data hurt - IOPscience](https://iopscience.iop.org/article/10.1088/1742-5468/ac3a74)
22. [Accurate Decoding of Natural Sentences from Non-Invasive Brain Recordings](https://arxiv.org/html/2608.18114v1)
23. [MEGState: Phoneme Decoding from Magnetoencephalography Signals](https://arxiv.org/abs/2512.17978)
24. [MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification](https://arxiv.org/abs/2512.01443)
25. [Unlocking Non-Invasive Brain-to-Text](https://arxiv.org/abs/2505.13446)
26. [Decoding individual words from non-invasive brain recordings across 723 participants](https://arxiv.org/html/2412.17829v1)
27. [Benchmarking Low-Shot Robustness to Natural Distribution Shifts](https://arxiv.org/pdf/2304.11263)
