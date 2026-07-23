MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

MODELS_DIR = "./models"

"""
training_config.py
===================
Single source of truth for everything training-related: LoRA config,
SFT/DPO hyperparameters, and derived paths.

Pulls BASE_MODEL / SFT_MODEL_DIR / DPO_MODEL_DIR from the root config.py
so paths aren't duplicated in two places. Only training-specific values
live here.

"""

import sys
from pathlib import Path

# Make root config.py importable when this file is run from training/
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import BASE_MODEL, SFT_MODEL_DIR, DPO_MODEL_DIR, OUTPUT_HF_DIR  # noqa: E402

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATASET_HF_DIR = OUTPUT_HF_DIR                     # data/dataset_hf (train/test splits)
SFT_MERGED_DIR = Path("models/sft-merged")          # full merged SFT weights (base + SFT adapter)
LOGS_DIR = Path("training/logs")

for p in (SFT_MODEL_DIR, DPO_MODEL_DIR, SFT_MERGED_DIR, LOGS_DIR):
    Path(p).mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# CHAT FORMAT — must match evaluation/generate.py exactly, or SFT/DPO
# training distribution won't match what generate.py feeds the model
# at eval time.
# ─────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an educational AI assistant."
    "Answer the given question clearly, simply(easy to understand) and accurately."
    "Limit your response to about 120-150 words."
)

# ─────────────────────────────────────────────
# QUANTIZATION (shared by SFT + DPO loading)
# ─────────────────────────────────────────────
BNB_CONFIG = dict(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    # bnb_4bit_compute_dtype set to torch.bfloat16 in adapter_utils.py
    # (kept out of this dict since torch import is heavy for a config file)
)

# ─────────────────────────────────────────────
# LoRA — SFT and DPO use the same architecture-level config (r/alpha/
# dropout/target_modules) per the finalized hyperparameters. Kept as two
# separate objects (not one shared instance) since DPO's adapter is a
# fresh one trained on top of the merged SFT model, not a continuation.
# ─────────────────────────────────────────────
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]

SFT_LORA_CONFIG = dict(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

DPO_LORA_CONFIG = dict(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=LORA_TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)

# ─────────────────────────────────────────────
# SFT TRAINING ARGS
# ─────────────────────────────────────────────
SFT_ARGS = dict(
    output_dir=str(LOGS_DIR / "sft-run"),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,
    report_to=[],
)

MAX_SEQ_LENGTH = 1024  # max total length (prompt + completion)

# ─────────────────────────────────────────────
# DPO TRAINING ARGS
# ─────────────────────────────────────────────
DPO_BETA = 0.1
DPO_MAX_PROMPT_LENGTH = 512
DPO_MAX_LENGTH = 1024

DPO_ARGS = dict(
    output_dir=str(LOGS_DIR / "dpo-run"),
    num_train_epochs=2,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=25,
    report_to=[],
)

# ─────────────────────────────────────────────
# GENERATION (for inference.py smoke tests only — the real eval pipeline
# is evaluation/generate.py + evaluation/evaluate.py)
# ─────────────────────────────────────────────
MAX_NEW_TOKENS = 200