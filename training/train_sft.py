"""
train_sft.py
============
QLoRA SFT on `chosen` responses only, using TRL's SFTTrainer.

Usage (from project root):
    python -m training.train_sft
"""

import json
import sys
from pathlib import Path

from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_MODEL, SFT_MODEL_DIR, EVAL_PROMPTS_PATH  # noqa: E402
from training.training_config import (  # noqa: E402
    DATASET_HF_DIR,
    MAX_SEQ_LENGTH,
    SFT_ARGS,
    SFT_LORA_CONFIG,
    SYSTEM_PROMPT,
)
from training.adapter_utils import (  # noqa: E402
    attach_lora,
    load_quantized_base,
    load_tokenizer,
    save_adapter,
)


def check_no_eval_overlap(train_dataset):
    """Guard against silent train/eval prompt overlap before burning GPU hours.
    Must run BEFORE build_prompt_completion reformats the 'prompt' column,
    since eval_prompts.json stores raw (non-templated) prompt strings."""
    eval_prompts = {
        p["prompt"]
        for p in json.loads(Path(EVAL_PROMPTS_PATH).read_text(encoding="utf-8"))
    }
    train_prompts = set(train_dataset["prompt"])
    overlap = eval_prompts & train_prompts
    if overlap:
        raise ValueError(
            f"{len(overlap)} eval prompts found in SFT training data — "
            "this will invalidate SFT vs baseline comparisons. Aborting."
        )
    print("✓ No overlap between eval prompts and training data")


def build_prompt_completion(example, tokenizer):
    """
    Reformat into the columns SFTConfig(completion_only_loss=True) expects:
      prompt     -> chat-templated string ending right where the assistant
                    turn should start (add_generation_prompt=True) — matches
                    evaluation/generate.py's formatting
      completion -> raw chosen response text (loss is computed on this only;
                    the prompt tokens are automatically masked out)
    """
    example["prompt"] = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["prompt"]},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    example["completion"] = example["chosen"]
    return example


def main():
    print(f"Loading base model (4-bit): {BASE_MODEL}")
    tokenizer = load_tokenizer(BASE_MODEL)
    base = load_quantized_base(BASE_MODEL)
    model = attach_lora(base, SFT_LORA_CONFIG)

    print(f"Loading dataset: {DATASET_HF_DIR}")
    dataset = load_from_disk(str(DATASET_HF_DIR))
    train_dataset = dataset["train"].remove_columns(
        [c for c in dataset["train"].column_names if c not in ("prompt", "chosen")]
    )

    # Check overlap BEFORE reformatting 'prompt' into its chat-templated form
    check_no_eval_overlap(train_dataset)

    train_dataset = train_dataset.map(lambda ex: build_prompt_completion(ex, tokenizer))

    sft_config = SFTConfig(
        **SFT_ARGS,
        max_length=MAX_SEQ_LENGTH,
        completion_only_loss=True,  # loss computed on 'completion' only, prompt tokens masked
        packing=False,  # packing is incompatible with completion-only loss
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    save_adapter(model, tokenizer, SFT_MODEL_DIR)
    print("✓ SFT training complete")


if __name__ == "__main__":
    main()
