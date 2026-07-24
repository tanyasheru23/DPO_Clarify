# Experiment 1 – Supervised Fine-Tuning (QLoRA)

## Objective

Fine-tune Qwen2.5-7B-Instruct to generate simpler, more beginner-friendly
technical explanations while preserving correctness.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2.5-7B-Instruct |
| Fine-tuning | QLoRA |
| Dataset | 942 training samples |
| Epochs | 2 |
| Learning Rate | 5e-5 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| LoRA Rank | 32 |

---

# Readability Metrics

| Metric | Base | SFT | Δ |
|--------|------:|------:|------:|
| Flesch Reading Ease | 45.20 | **65.42** | +20.22 |
| Reading Grade Level | 11.33 | **8.43** | -2.90 |
| Average Word Count | 110.58 | 99.48 | -11.10 |
| Jargon Density | 0.003 | **0.001** | -0.002 |
| Example Signals | 0.74 | **1.20** | +0.46 |
| Analogy Signals | 0.22 | **0.40** | +0.18 |
| BERTScore F1 | **0.724** | 0.718 | -0.006 |

---

# GPT-4 Judge Evaluation

| Metric | Base | SFT | Δ |
|--------|------:|------:|------:|
| Overall | 4.46 | **4.68** | +0.22 |
| Clarity | 4.58 | **4.60** | +0.02 |
| Beginner Friendliness | 4.56 | **4.84** | +0.28 |
| Use of Examples | 3.88 | **4.58** | +0.70 |
| Jargon Handling | 4.48 | **4.64** | +0.16 |
| Logical Flow | **4.80** | 4.74 | -0.06 |

---

# Category-wise GPT Judge Scores

| Category | Base | SFT |
|----------|------:|------:|
| AI | 3.80 | **5.00** |
| Computer Science | 4.49 | **4.51** |
| Statistics | 4.27 | **4.80** |
| Mathematics | 4.47 | **4.76** |
| Physics | 4.50 | **4.70** |
| Biology | 4.45 | **4.60** |
| Chemistry | **4.60** | 4.47 |
| Software Engineering | 4.80 | **5.00** |
| General Reasoning | 5.00 | 5.00 |
| Machine Learning | 4.11 | **4.63** |
| Economics | 4.60 | **4.73** |

---

# Observations

- Readability improved substantially, with Flesch Reading Ease increasing from **45.2 → 65.4**.
- Average reading grade level decreased from **11.3 → 8.4**, making responses easier to understand.
- GPT-4 Judge rated the fine-tuned model higher in **overall quality**, **beginner friendliness**, **use of examples**, and **jargon handling**.
- The largest qualitative improvement was observed in explanations requiring educational simplification rather than factual recall.
- A small reduction in logical flow score was observed, suggesting an area for further optimization during DPO.

---

# Limitations

- Approximately 942 training samples.
- Two SFT epochs.
- Evaluation performed using GPT-4 as an automatic judge.
- Response length constrained during training.