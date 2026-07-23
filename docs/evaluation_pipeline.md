# Evaluation Pipeline

## Overview

The goal of this project is not only to fine-tune an LLM using Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO), but also to quantitatively measure whether the fine-tuned models produce better educational explanations than the baseline model.

Unlike traditional text generation projects that rely solely on BLEU or ROUGE, this project evaluates responses across multiple complementary dimensions:

- Readability
- Semantic relevance
- Instruction-following quality
- Educational effectiveness
- Output consistency
- Generation efficiency

The complete evaluation pipeline is fully automated and can be executed for any compatible model checkpoint.

---

# Evaluation Workflow

```
                 Evaluation Prompts
                         │
                         ▼
              Generate Model Responses
                         │
                         ▼
              results/base_responses.json
                         │
                         ▼
                 evaluation.py
                         │
         ┌───────────────┼────────────────┐
         │               │                │
         ▼               ▼                ▼
 Readability      Semantic Metrics   LLM-as-Judge
         │               │                │
         └───────────────┼────────────────┘
                         ▼
             Category-wise Aggregation
                         │
                         ▼
               Final Evaluation Report
```

---

# Evaluation Dataset

The benchmark contains **50 educational prompts** covering multiple domains.

| Category | Examples |
|----------|----------|
| AI | Gradient Descent, Overfitting |
| Machine Learning | Attention, Embeddings |
| Computer Science | Hash Maps, Recursion |
| Mathematics | Bayes Theorem, Derivatives |
| Statistics | Correlation, Confidence Intervals |
| Physics | Entropy, Electricity |
| Biology | Vaccines, Sleep |
| Chemistry | Solubility |
| Economics | Inflation |
| Software Engineering | Unit Testing |
| General Reasoning | Opportunity Cost |

The prompts were manually curated to evaluate explanatory ability rather than factual memorization.

---

# Generation Configuration

Responses are generated using deterministic decoding.

```python
do_sample=False
temperature=1.0
```

Greedy decoding ensures reproducible evaluation across Base, SFT and DPO models.

Each prompt is processed independently and the following metadata is stored:

- Prompt
- Generated response
- Latency
- Number of generated tokens
- Model name

---

# Evaluation Metrics

## 1. Readability Metrics

These metrics estimate how accessible each explanation is for beginners.

### Flesch Reading Ease

Measures overall reading difficulty.

Higher score = easier to understand.

Typical interpretation:

| Score | Interpretation |
|--------|---------------|
| 90-100 | Very Easy |
| 60-70 | Ideal |
| 30-50 | Difficult |

---

### Grade Level

Estimated education level required to understand the response.

Lower values generally indicate more beginner-friendly explanations.

---

### Average Word Count

Measures average response length.

Used to compare verbosity across different models.

---

### Jargon Density

Measures the proportion of domain-specific technical terminology.

Lower jargon density generally improves accessibility.

---

### Example Signals

Counts explanatory examples such as:

- "for example"
- "imagine"
- "suppose"

Educational explanations benefit from concrete examples.

---

### Analogy Signals

Counts analogy-based explanations including phrases like:

- "like"
- "similar to"
- "think of"

Analogies often improve conceptual understanding.

---

# 2. Semantic Quality

## BERTScore

Traditional metrics such as BLEU and ROUGE reward exact word overlap.

Educational explanations often have many equally correct phrasings.

BERTScore measures semantic similarity using contextual embeddings, making it more suitable for evaluating explanatory responses.

Higher scores indicate greater semantic relevance.

---

# 3. LLM-as-a-Judge

An LLM evaluates every generated response according to a fixed rubric.

Each criterion is scored from **1 (Poor)** to **5 (Excellent)**.

The evaluation rubric includes:

- Clarity
- Beginner Friendliness
- Use of Examples
- Jargon Handling
- Logical Flow
- Overall Quality

These metrics capture qualitative aspects that cannot be measured using lexical similarity alone. Modern LLM evaluation commonly combines automatic metrics with model-based judging for more comprehensive assessment. :contentReference[oaicite:0]{index=0}

---

# 4. Reliability Metrics

## Non-English Ratio

Measures the proportion of responses containing non-English content.

Examples include:

- Chinese prefixes
- Mixed-language generations
- Unexpected multilingual artifacts

Lower values indicate more consistent instruction following.

---

## Formatting Error Rate

Measures malformed generations such as:

- Unexpected prefixes
- Dataset artifacts
- Corrupted formatting
- Incomplete responses

Lower values indicate cleaner outputs.

---

# Category-wise Analysis

Scores are aggregated separately for every subject domain.

Example:

```
Physics
Mathematics
Computer Science
Economics
Biology
```

This makes it possible to identify subject-specific strengths and weaknesses rather than relying on a single overall score.

---

# Why Not BLEU or ROUGE?

BLEU and ROUGE primarily measure lexical overlap.

For educational explanation tasks, multiple valid answers may use completely different wording while conveying identical meaning.

Example:

Reference:

> Overfitting memorizes noise in the training data.

Generated:

> Overfitting learns patterns that do not generalize to unseen examples.

Despite being semantically equivalent, BLEU may assign a relatively low score due to limited word overlap.

Therefore, this project emphasizes:

- Readability
- Semantic similarity (BERTScore)
- LLM-as-a-Judge
- Reliability metrics

instead of relying exclusively on token-overlap metrics.

---

# Output

The evaluation script produces:

```
Readability Metrics
-------------------
Flesch Reading Ease
Grade Level
Average Word Count
Jargon Density
Example Signals
Analogy Signals

Semantic Metrics
----------------
BERTScore

Reliability Metrics
-------------------
Formatting Error Rate
Non-English Ratio

LLM-as-Judge
------------
Clarity
Logical Flow
Examples
Overall Score

Category Breakdown
------------------
AI
Machine Learning
Computer Science
Mathematics
Physics
Biology
Chemistry
Economics
...
```

---

## Completed

- ✅ Designed a benchmark consisting of 50 educational prompts spanning multiple academic domains.
- ✅ Implemented an automated response generation pipeline using Qwen2.5-7B-Instruct.
- ✅ Configured deterministic inference (do_sample=False) for reproducible evaluation.
- ✅ Recorded generation metadata
- ✅ Implemented automated evaluation metrics
- ✅ Generated the baseline evaluation for **Qwen2.5-7B-Instruct**.

### Baseline Results

| Metric | Result |
|---------|-------:|
| Flesch Reading Ease | **55.29** |
| Grade Level | **10.22** |
| Average Word Count | **146.8** |
| BERTScore | **0.713** |
| Overall LLM-as-Judge | **4.83 / 5** |
| Formatting Error Rate | **0.0%** |
| Non-English Ratio | **0.1%** |

## Observations

The corrected evaluation pipeline demonstrates that the baseline model already produces clear, well-structured educational explanations with strong instruction-following performance.

Key observations include:

- High overall educational quality (**4.83/5**) according to the LLM-as-a-Judge rubric.
- Responses consistently include examples and beginner-friendly explanations.
- Proper chat template formatting eliminated malformed outputs and multilingual artifacts observed during early development.
- The baseline establishes a reliable reference point for evaluating improvements introduced by SFT and DPO.

## Next Milestones

- ⏳ Supervised Fine-Tuning (SFT)
- ⏳ Evaluate SFT model
- ⏳ Direct Preference Optimization (DPO)
- ⏳ Evaluate DPO model
- ⏳ Final comparison (Base vs SFT vs DPO)

---

# Summary

The evaluation framework combines traditional readability metrics, semantic similarity, model-based judging and reliability analysis to provide a holistic assessment of educational explanation quality.

Rather than measuring only lexical overlap, the framework focuses on the qualities most important for instructional language models:

- Correctness
- Clarity
- Accessibility
- Consistency
- Educational effectiveness


## Engineering Notes

During development, the evaluation pipeline was validated to ensure responses were generated using the tokenizer's official chat template.

An incorrectly specified `add_generation_prompt` argument initially prevented the assistant generation marker from being appended, leading to malformed outputs.

After correcting the prompt formatting, response quality and instruction-following behavior improved substantially. All reported baseline metrics were generated using the corrected evaluation pipeline.