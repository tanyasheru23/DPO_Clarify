"""
train_dpo.py
============
QLoRA DPO on chosen/rejected pairs, trained on top of the SFT model.

Flow:
  1. Merge the SFT LoRA adapter into the base model (once — skipped if
     already merged). This produces `models/sft-merged/`.
  2. Load the merged model in 4-bit and attach a *fresh* LoRA adapter.
  3. Train with DPOTrainer, ref_model=None — TRL disables the new
     adapter internally to compute reference logits, which correctly
     gives you the SFT model's logits as the reference (since disabling
     the adapter reduces the merged model back to plain SFT weights).

Usage (from project root):
    python -m training.train_dpo
"""

import sys
from pathlib import Path

from datasets import load_from_disk
from trl import DPOConfig, DPOTrainer

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_MODEL, SFT_MODEL_DIR, DPO_MODEL_DIR  # noqa: E402
from training.training_config import (  # noqa: E402
    DATASET_HF_DIR,
    DPO_ARGS,
    DPO_BETA,
    DPO_LORA_CONFIG,
    DPO_MAX_LENGTH,
    DPO_MAX_PROMPT_LENGTH,
    SFT_MERGED_DIR,
    SYSTEM_PROMPT,
)
from training.adapter_utils import (  # noqa: E402
    attach_lora,
    load_quantized_base,
    load_tokenizer,
    merge_adapter_and_save,
    save_adapter,
)


def build_preference_example(example, tokenizer):
    """
    Format into the columns DPOTrainer expects:
      prompt   -> fully chat-templated string, ending right where the
                  assistant turn should start (add_generation_prompt=True)
      chosen   -> raw assistant text + eos token
      rejected -> raw assistant text + eos token

    Pre-applying the template to `prompt` (rather than leaving it as a
    conversational list) keeps this consistent with how
    evaluation/generate.py builds prompts at inference time.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["prompt"]},
    ]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return {
        "prompt": formatted_prompt,
        "chosen": example["chosen"] + tokenizer.eos_token,
        "rejected": example["rejected"] + tokenizer.eos_token,
    }


def main():
    # Step 1 — merge SFT adapter into base, once
    merge_adapter_and_save(BASE_MODEL, SFT_MODEL_DIR, SFT_MERGED_DIR)

    # Step 2 — load merged SFT model in 4-bit, attach a fresh adapter for DPO
    print(f"Loading merged SFT model (4-bit): {SFT_MERGED_DIR}")
    tokenizer = load_tokenizer(SFT_MERGED_DIR)
    base = load_quantized_base(SFT_MERGED_DIR)
    model = attach_lora(base, DPO_LORA_CONFIG)

    print(f"Loading dataset: {DATASET_HF_DIR}")
    dataset = load_from_disk(str(DATASET_HF_DIR))
    train_dataset = dataset["train"].map(
        lambda ex: build_preference_example(ex, tokenizer),
        remove_columns=[c for c in dataset["train"].column_names if c not in ("prompt", "chosen", "rejected")],
    )

    dpo_config = DPOConfig(
        **DPO_ARGS,
        beta=DPO_BETA,
        max_prompt_length=DPO_MAX_PROMPT_LENGTH,
        max_length=DPO_MAX_LENGTH,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # uses the disable-adapter trick -> reference = merged SFT weights
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )

    print("Starting DPO training...")
    trainer.train()

    save_adapter(model, tokenizer, DPO_MODEL_DIR)
    print("✓ DPO training complete")


if __name__ == "__main__":
    main()