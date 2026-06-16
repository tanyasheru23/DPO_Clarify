from config import OUTPUT_HF_DIR, OUTPUT_JSONL, MIN_ANSWER_LENGTH
from datasets import Dataset
import random
import json


def validate_pair(pair: dict) -> bool:
    """Basic sanity checks before including a pair."""
    prompt = pair.get("prompt", "")
    chosen = pair.get("chosen", "")
    rejected = pair.get("rejected", "")

    if not prompt or not chosen or not rejected:
        return False
    if len(chosen) < MIN_ANSWER_LENGTH or len(rejected) < MIN_ANSWER_LENGTH:
        return False
    if chosen == rejected:
        return False
    # Chosen should generally be longer (more thorough explanations)
    # but don't enforce strictly — just flag very lopsided cases
    return True


def save_dataset(pairs: list[dict]):
    """Save as JSONL and HuggingFace Dataset."""
    valid = [p for p in pairs if validate_pair(p)]
    invalid = len(pairs) - len(valid)
    print(f"\n{len(valid)} valid pairs  ({invalid} dropped)")

    # Shuffle
    random.shuffle(valid)

    # Split 90/10
    split_idx = int(len(valid) * 0.9)
    train_pairs = valid[:split_idx]
    eval_pairs = valid[split_idx:]

    # Save JSONL
    with open(OUTPUT_JSONL, "w") as f:
        for pair in valid:
            f.write(
                json.dumps(
                    {
                        "prompt": pair["prompt"],
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                        "source": pair.get("source", "unknown"),
                    }
                )
                + "\n"
            )
    print(f"Saved {OUTPUT_JSONL}  ({len(valid)} rows)")

    # Save HuggingFace Dataset format (what TRL DPOTrainer expects)
    def to_hf(subset):
        return Dataset.from_list(
            [
                {
                    "prompt": p["prompt"],
                    "chosen": p["chosen"],
                    "rejected": p["rejected"],
                }
                for p in subset
            ]
        )

    from datasets import DatasetDict

    hf_dataset = DatasetDict({"train": to_hf(train_pairs), "test": to_hf(eval_pairs)})
    hf_dataset.save_to_disk(str(OUTPUT_HF_DIR))
    print(f"Saved HF dataset → {OUTPUT_HF_DIR}/")
    print(f"Train: {len(train_pairs)} | Eval: {len(eval_pairs)}")

    # Source breakdown
    from collections import Counter

    sources = Counter(p.get("source") for p in valid)
    print("\nSource breakdown:")
    for src, count in sources.items():
        print(f"{src:15s} {count}")
