# Training

## Base Model

Model:
Qwen2.5-7B-Instruct

Fine-tuning method:
QLoRA

Frameworks:

- Transformers
- PEFT
- TRL
- BitsAndBytes

---

# Quantization

4-bit NF4 quantization

Double Quantization enabled

Optimizer:

Paged AdamW 8-bit

---

# LoRA Configuration

Rank: 32

Alpha: 64

Dropout: 0.05

Target Modules

- q_proj
- k_proj
- v_proj
- o_proj
- gate_proj
- up_proj

Bias:

none

---

# SFT Hyperparameters

Epochs

2

Learning Rate

5e-5

Scheduler

Cosine

Batch Size

1

Gradient Accumulation

8

Maximum Sequence Length

1024

Gradient Checkpointing

Enabled

---

# Dataset

Source

Stack Exchange

Training Samples

942

Evaluation Samples

105

Data consists of

Prompt

Chosen Response

Rejected Response

The SFT stage trains only on the chosen responses.

---

# Checkpoint Strategy

Model checkpoints are saved every 25 optimizer steps.

Saved artifacts include:

- LoRA Adapter
- Optimizer State
- Scheduler State
- Trainer State
- Tokenizer
- RNG State

This allows interrupted training to resume without restarting.

---

# Training Outputs

The following artifacts are produced:

models/

- sft_output
- sft_merged

training/logs/

- trainer_state.json
- checkpoints

results/

- generated responses
- evaluation reports

---

# Evaluation Pipeline

Base

↓

Generate Responses

↓

Automatic Metrics

↓

GPT Judge

↓

Comparison Report

---

# Lessons Learned

## Training

Colab free tier may disconnect during long QLoRA runs.

Checkpointing every 25 steps significantly improves fault tolerance.

---

## Evaluation

Training and evaluation must use identical system prompts.

A mismatch between prompts can invalidate comparisons.

---

## Reproducibility

Training is reproducible through

- saved adapters
- checkpoints
- evaluation reports
- fixed hyperparameters