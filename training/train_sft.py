"""
train_sft.py
============
QLoRA SFT on `chosen` responses only, using TRL's SFTTrainer.

Usage (from project root):
    python -m training.train_sft
"""

import sys
from pathlib import Path

from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
from trl.trainer import DataCollatorForCompletionOnlyLM

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_MODEL, SFT_MODEL_DIR  # noqa: E402
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

# Must match the assistant-turn marker in Qwen's ChatML template exactly,
# so DataCollatorForCompletionOnlyLM knows where the loss-relevant tokens
# start (loss is masked on everything before this, i.e. system + user turns).
RESPONSE_TEMPLATE = "<|im_start|>assistant\n"


def build_text(example, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["chosen"]},
    ]
    example["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example


def main():
    print(f"Loading base model (4-bit): {BASE_MODEL}")
    tokenizer = load_tokenizer(BASE_MODEL)
    base = load_quantized_base(BASE_MODEL)
    model = attach_lora(base, SFT_LORA_CONFIG)

    print(f"Loading dataset: {DATASET_HF_DIR}")
    dataset = load_from_disk(str(DATASET_HF_DIR))
    train_dataset = dataset["train"].map(
        lambda ex: build_text(ex, tokenizer),
        remove_columns=[c for c in dataset["train"].column_names if c not in ("prompt", "chosen")],
    )

    # Sanity check: none of the eval prompts should be in the training set.
    # Cheap guard against silent train/eval overlap before burning GPU hours.
    import json
    from config import EVAL_PROMPTS_PATH

    eval_prompts = {p["prompt"] for p in json.loads(Path(EVAL_PROMPTS_PATH).read_text(encoding="utf-8"))}
    train_prompts = set(train_dataset["prompt"])
    overlap = eval_prompts & train_prompts
    if overlap:
        raise ValueError(
            f"{len(overlap)} eval prompts found in SFT training data — "
            "this will invalidate SFT vs baseline comparisons. Aborting."
        )
    print("✓ No overlap between eval prompts and training data")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=RESPONSE_TEMPLATE,
        tokenizer=tokenizer,
    )

    sft_config = SFTConfig(
        **SFT_ARGS,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        packing=False,  # packing + completion-only loss masking don't mix well
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("Starting SFT training...")
    trainer.train()

    save_adapter(model, tokenizer, SFT_MODEL_DIR)
    print("✓ SFT training complete")


if __name__ == "__main__":
    main()