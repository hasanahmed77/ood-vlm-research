# Research Plan: OOD Detection for Generative Vision-Language Models

> **One-line pitch:** TANL and all prior OOD detection work assumes discriminative CLIP-style models
> with a closed-set softmax — but the world has moved to generative VLMs (LLaVA, InstructBLIP),
> where that entire framework breaks. We propose the first activation-inspired, training-free
> OOD detection method for generative VLMs.

---

## Problem Statement

Existing VLM-based OOD detection methods (NegLabel, TANL, CSP, AdaNeg) all share a critical
assumption: the model is a **discriminative CLIP encoder** that produces a softmax probability
over a fixed label set. This enables the "negative label" paradigm — mine text labels that
activate strongly on OOD images, detect OOD via similarity scores.

**This breaks completely for generative VLMs** (LLaVA, InstructBLIP, Qwen-VL) because:
1. No fixed label-set softmax exists
2. Activation scores cannot be computed the same way
3. The model generates free-form text, not classification logits

Nobody has solved OOD detection for generative VLMs in a training-free, test-time adaptive way.
That is our paper.

---

## Research Hypothesis

> Token-level log-likelihoods and hidden-state uncertainty in generative VLMs carry
> activation-equivalent signals that can be leveraged for training-free OOD detection —
> analogous to TANL's activation metric but adapted for autoregressive generation.

---

## Key Papers to Read (In Order)

| Priority | Paper | Venue | Why Read |
|----------|-------|-------|----------|
| 1 | NegLabel | ICLR 2024 | Foundation of negative label OOD framework |
| 2 | TANL (this repo) | CVPR 2025 | Our direct jumping-off point |
| 3 | MCM | NeurIPS 2022 | Original CLIP OOD baseline |
| 4 | AdaNeg | NeurIPS 2024 | Best adaptive baseline to beat |
| 5 | LLaVA 1.5 | NeurIPS 2023 | Target generative VLM |
| 6 | OpenOOD v1.5 | arXiv 2023 | Benchmark framework we'll use |
| 7 | VisTa | CVPR 2025 | Know this competitor (object-level OOD) |
| 8 | RUNA | AAAI 2025 | Know this competitor (object-level OOD) |
| 9 | ANTS | arXiv 2025 | Know this competitor (MLLM negative labels) |

---

## What We Are NOT Doing (Avoid These Dead Ends)

| Direction | Why Dead |
|-----------|----------|
| LLM-augmented negative label corpus | ANTS (arXiv 2025) + EOE (2024) already did this |
| Object/region-level OOD with CLIP | VisTa (CVPR 2025) + RUNA (AAAI 2025) already did this |
| Medical imaging OOD | Low novelty, skeptical A* reviewers, declining funding |
| Video OOD | Too high compute for M2 Air + Colab setup |

---

## Methodology (Working Plan)

### Step 1 — Reproduce TANL baseline
- Run OpenOOD-VLM codebase
- Reproduce Table 1: FPR95 = 9.81% on ImageNet with ViT-B/16 CLIP
- Reproduce NegLabel baseline: FPR95 = 25.40% (sanity check)

### Step 2 — Demonstrate failure on generative VLMs (Motivation)
- Apply TANL's score function to LLaVA-1.5-7B
- Show it produces poor OOD separation on ImageNet vs iNaturalist/Places
- Document exactly WHY it fails (no closed-set softmax, no label probability vector)
- This broken experiment = paper motivation section

### Step 3 — Propose activation-equivalent metric for generative VLMs
Candidate approaches (to be explored):
- **Token log-likelihood proxy:** Prompt the VLM with "This is a photo of [label]" and
  measure the log-probability of the label tokens as an activation signal
- **Hidden state norm/entropy:** Use last-layer hidden state statistics as uncertainty signal
- **Contrastive generation:** Prompt with ID vs. negative concept descriptions,
  measure relative generation probability
- **Vocabulary probing:** Extract the implicit probability distribution over the vocabulary
  at the final token position as a soft label distribution

### Step 4 — Test-time adaptation for generative VLMs
- Adapt TANL's FIFO queue concept: cache high-confidence positive/negative image embeddings
- Dynamically update which "negative concepts" activate strongly on the evolving test distribution
- Keep it training-free and test-efficient (key selling point)

---

## Experiments Plan

### Datasets
| Dataset | Role |
|---------|------|
| ImageNet-1k | In-distribution (ID) |
| iNaturalist | OOD (semantic shift) |
| Places365 | OOD (scene shift) |
| SUN | OOD |
| Textures | OOD (texture shift) |
| OpenOOD benchmark | Standardized evaluation |

### Models
| Model | Role |
|-------|------|
| CLIP ViT-B/16 | Discriminative baseline (reproduce TANL) |
| LLaVA-1.5-7B | Primary generative VLM target |
| InstructBLIP-7B | Secondary generative VLM |
| Qwen-VL (optional) | Generalization experiment |

### Baselines to Beat
- MCM (training-free, CLIP)
- NegLabel (training-free, CLIP)
- TANL (training-free, CLIP) ← our paper shows this fails on generative VLMs
- AdaNeg (adaptive, CLIP)

### Key Metrics
- FPR95 (False Positive Rate at 95% TPR) ← primary
- AUROC
- ID Classification Accuracy (should not degrade)

---

## Compute Setup

| Task | Tool |
|------|------|
| Reading, coding, debugging | MacBook Air M2 8GB (local) |
| CLIP feature extraction, small ablations | Colab T4 (free tier) |
| LLaVA-7B inference, full ImageNet eval | Colab A100 (Pro+, $50/month) |
| Backup GPU compute | Kaggle (30h/week free) |

**Rule:** Never run large experiments locally. Develop + debug locally on small data subsets,
then scale on Colab.

---

## Publication Strategy: Workshop → Main Conference

### Why This Strategy Works
1. Workshop paper (4 pages) forces you to crystallize the core idea early
2. Workshop feedback from reviewers shapes the full paper
3. Workshop acceptance gives you confidence and a citation anchor
4. Extended version to main conference adds: more baselines, more datasets, theory, ablations

### Target Timeline

```
2026
├── Jun  (Now)     Set up repo, read all 9 papers
├── Jul            Reproduce TANL on Colab, understand codebase deeply
├── Aug            Reproduce failure on LLaVA — write motivation section (2 pages)
├── Sep            Propose method v1, initial experiments on ImageNet
├── Oct            Workshop paper draft (4 pages)
├── Nov ──────────► SUBMIT to NeurIPS 2026 Workshop on Distribution Shifts
│                  (or similar workshop, deadline usually Oct-Nov)
├── Dec            Workshop feedback + notification

2027
├── Jan            Begin full paper extension
├── Feb            Full experiments (all datasets, all baselines, ablations)
├── Mar ──────────► SUBMIT to ICML 2027 or CVPR 2027
│                  (deadlines: ICML ~Jan, CVPR ~Nov 2026*)
├── Apr-Jun        Revision period
└── Jul            Camera-ready / presentation

* Note: For CVPR 2027, submission deadline is likely Nov 2026 —
  meaning the workshop and main conference run in parallel after Oct.
  Plan accordingly: workshop paper IS the early draft of the main paper.
```

### Venue Priority
| Venue | Type | Why |
|-------|------|-----|
| NeurIPS Workshop on Distribution Shifts | Workshop (first target) | Directly on-topic, active community |
| ICLR Workshop on Trustworthy ML | Workshop (alternative) | Good for safety/reliability angle |
| NeurIPS 2027 main | Main conference (primary) | OOD = NeurIPS-friendly topic |
| ICML 2027 main | Main conference (alternative) | Strong if theory angle develops |
| CVPR 2027 main | Main conference (alternative) | If visual experiments dominate |

---

## Success Criteria

### Workshop Paper (Minimum Bar)
- [ ] TANL failure on generative VLMs clearly demonstrated
- [ ] One working proxy metric for generative VLM activation
- [ ] Results on ImageNet + 2 OOD datasets
- [ ] Beats MCM and NegLabel on at least one generative VLM

### Main Conference Paper (Full Bar)
- [ ] Method works across LLaVA-1.5 AND InstructBLIP (generalization)
- [ ] Beats all CLIP-based baselines adapted naively to generative setting
- [ ] OpenOOD benchmark full evaluation
- [ ] Ablation study on each component
- [ ] Theoretical justification (even if lightweight)
- [ ] Training-free, test-efficient (same constraints as TANL)

---

## Repository Structure (To Build)

```
ood-vlm/
├── PLAN.md                  ← This file
├── TANL.pdf                 ← Reference paper
├── TANL-appendix.pdf        ← Reference appendix
├── papers/                  ← PDFs of all 9 papers to read
├── reproductions/
│   ├── tanl_baseline/       ← Reproduce TANL on ImageNet
│   └── failure_analysis/    ← Show TANL fails on generative VLMs
├── src/
│   ├── models/              ← CLIP, LLaVA, InstructBLIP wrappers
│   ├── scores/              ← Score functions (TANL, MCM, ours)
│   ├── datasets/            ← ImageNet, iNaturalist, Places loaders
│   └── eval/                ← FPR95, AUROC evaluation
├── experiments/
│   ├── configs/             ← Experiment configs
│   └── results/             ← Saved results, plots
├── notebooks/               ← Colab notebooks for experiments
└── paper/                   ← LaTeX source (workshop + full)
```

---

## Weekly Habit

- **Every Friday:** Write 3 bullet points — what you ran, what broke, what it means
- **Every 2 weeks:** Re-read this PLAN.md and update it
- Keep a `NOTES.md` for raw observations during experiments — reviewers love when
  you can trace a design decision back to a specific empirical observation

---

*Last updated: June 2026*
*Target: NeurIPS 2026 Workshop → NeurIPS/ICML/CVPR 2027 Main Conference*
