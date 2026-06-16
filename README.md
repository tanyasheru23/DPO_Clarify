# DPOClarify

Train a **Mistral-7B** model using **Direct Preference Optimization (DPO)** and **QLoRA** to prefer explanations that improve understanding rather than merely being factually correct.

The dataset consists of question-answer preference pairs where **both answers are correct**, but one explanation is substantially clearer than the other.

---

## Motivation

Most LLMs default to textbook-style explanations:

* definitions before intuition
* heavy jargon
* few analogies
* limited concrete examples

These responses are often accurate but don't always help users build genuine understanding.

The key idea behind this project:

> **Clarity is a preference, not a correctness label.**

Traditional Supervised Fine-Tuning (SFT) only learns *what is correct*.

Direct Preference Optimization (DPO) can instead learn:

> "Given two factually correct answers, which one better helps a person understand?"

---

## Current Status

* [x] Dataset collection
* [x] Dataset validation
* [x] CI pipeline (lint + tests)
* [ ] Mistral-7B + QLoRA training
* [ ] DPO training
* [ ] Evaluation

---

## Dataset

**1,047 preference pairs** collected from three sources:

| Source             | Pairs | Quality Signal           |
| ------------------ | ----: | ------------------------ |
| StackExchange      |   600 | Upvote score gap         |
| Reddit ELI5        |   300 | Community upvotes        |
| Synthetic (GPT-4o) |   147 | Rubric-guided generation |

Each example contains:

* `prompt` → a natural user question
* `chosen` → a clear explanation (analogy-driven, concrete examples first)
* `rejected` → a confusing explanation (definition-first, jargon-heavy)

Both `chosen` and `rejected` are **factually correct**.

The only difference is **clarity**.

---

## Project Structure

```text
DPO_Clarify_Dataset/

├── .github/
│   └── workflows/
│       └── ci.yml

├── src/
│   ├── stack_exchange_pairs.py
│   ├── eli5_pairs.py
│   ├── synthetic_pairs.py
│   ├── save_dataset.py
│   └── utils.py

├── tests/
│   └── test_pipeline.py

├── build_dataset.py
├── config.py
├── requirements.txt
└── .env.example
```

---

## Installation

```bash
git clone https://github.com/<your-username>/DPO_Clarify_Dataset.git

cd DPO_Clarify_Dataset

pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Add your OpenAI API key:

```text
OPENAI_API_KEY=your_key_here
```

---

## Build the Dataset

```bash
python build_dataset.py
```

Intermediate outputs are saved after each stage to avoid losing progress if a job fails.

Generated files:

```text
data/
├── parts/
│   ├── se_pairs.json
│   ├── eli5_pairs.json
│   └── synth_pairs.json
│
├── dataset.jsonl
└── dataset_hf/
```

---

## Continuous Integration (CI)

GitHub Actions runs automatically on every push and pull request.

Checks include:

1. **Linting** with Ruff
2. **Formatting checks** with Ruff
3. **Unit tests** with Pytest
4. **Coverage reporting**

A pull request must pass all checks before merging.

---

## Next Steps

* Train Mistral-7B using QLoRA
* Run DPO training
* Evaluate explanation quality
* Compare against SFT baselines
