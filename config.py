from pathlib import Path

# Paths
DATA_DIR = Path("data")
OUTPUT_JSONL = DATA_DIR / "dataset.jsonl"
OUTPUT_HF_DIR = DATA_DIR / "dataset_hf"
PARTS_DIR = DATA_DIR / "parts"

TARGET_SE_PAIRS = 600  # from StackExchange
TARGET_ELI5_PAIRS = 300  # from ELI5
TARGET_SYNTH_PAIRS = 150  # synthetic via OpenAI

MIN_SCORE_GAP = 10  # chosen score must beat rejected by at least this
MIN_ANSWER_LENGTH = 80  # characters — filter out one-liners

OPENAI_MODEL = "gpt-4o"
