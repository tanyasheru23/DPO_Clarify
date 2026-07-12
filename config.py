from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_JSONL = DATA_DIR / "dataset.jsonl"
OUTPUT_HF_DIR = DATA_DIR / "dataset_hf"
PARTS_DIR = DATA_DIR / "parts"
EVAL_PROMPTS_PATH  = DATA_DIR / "eval_prompts.json"
RESULTS_DIR = Path("results")

# Create directories once
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HF_DIR.mkdir(parents=True, exist_ok=True)
PARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SE_PAIRS = 600
TARGET_ELI5_PAIRS = 300
TARGET_SYNTH_PAIRS = 150

MIN_SCORE_GAP = 3
MIN_ANSWER_LENGTH = 80

OPENAI_MODEL = "gpt-4o"
JUDGE_MODEL = "gpt-4o"

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
BASE_MODEL         = Path("models/Qwen2.5-7B-Instruct")
DPO_MODEL_DIR      = Path("models/dpo-output")
SFT_MODEL_DIR      = Path("models/sft-output")