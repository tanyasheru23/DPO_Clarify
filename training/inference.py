"""
inference.py
============
Quick manual smoke test for the base/SFT/DPO models — NOT the eval
pipeline (use evaluation/generate.py + evaluate.py for that). This is
just for eyeballing a response or two right after training, before
committing to a full 50-prompt eval run.

Usage (from project root):
    python -m training.inference --model base --prompt "What is entropy?"
    python -m training.inference --model sft  --prompt "What is entropy?"
    python -m training.inference --model dpo  --prompt "What is entropy?"
"""

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import BASE_MODEL, SFT_MODEL_DIR, DPO_MODEL_DIR  # noqa: E402
from training.training_config import MAX_NEW_TOKENS, SFT_MERGED_DIR, SYSTEM_PROMPT  # noqa: E402
from training.adapter_utils import load_quantized_base, load_tokenizer  # noqa: E402


def load_for_inference(model_choice: str):
    """
    base -> quantized base model, no adapter
    sft  -> quantized base + SFT adapter
    dpo  -> quantized merged-SFT model + DPO adapter
             (mirrors exactly what training produced — DPO adapter was
             trained on top of the merged SFT weights, not raw base)
    """
    if model_choice == "base":
        tokenizer = load_tokenizer(BASE_MODEL)
        model = load_quantized_base(BASE_MODEL)

    elif model_choice == "sft":
        if not Path(SFT_MODEL_DIR).exists():
            raise FileNotFoundError(
                f"No SFT adapter found at {SFT_MODEL_DIR} — run train_sft.py first"
            )
        tokenizer = load_tokenizer(BASE_MODEL)
        base = load_quantized_base(BASE_MODEL)
        model = PeftModel.from_pretrained(base, str(SFT_MODEL_DIR))

    elif model_choice == "dpo":
        if not Path(DPO_MODEL_DIR).exists():
            raise FileNotFoundError(
                f"No DPO adapter found at {DPO_MODEL_DIR} — run train_dpo.py first"
            )
        if not Path(SFT_MERGED_DIR).exists():
            raise FileNotFoundError(
                f"No merged SFT model found at {SFT_MERGED_DIR} — "
                "train_dpo.py creates this automatically, run that first"
            )
        tokenizer = load_tokenizer(SFT_MERGED_DIR)
        base = load_quantized_base(SFT_MERGED_DIR)
        model = PeftModel.from_pretrained(base, str(DPO_MODEL_DIR))

    else:
        raise ValueError(f"Unknown model choice: {model_choice}")

    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # note: fixed vs. the typo in evaluation/generate.py
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(
        model.device
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "sft", "dpo"], required=True)
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    model, tokenizer = load_for_inference(args.model)
    response = generate(model, tokenizer, args.prompt)

    print(f"\n[{args.model}] {args.prompt}\n")
    print(response)
