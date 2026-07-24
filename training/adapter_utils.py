"""
adapter_utils.py
=================
Shared helpers used by train_sft.py, train_dpo.py, and inference.py:
  - loading the base model (quantized or full precision)
  - attaching a fresh LoRA adapter
  - saving an adapter
  - merging an adapter into its base and saving the merged weights

Keeping this in one place means the SFT and DPO scripts can't drift out
of sync on how the model is loaded/quantized.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

sys.path.append(str(Path(__file__).resolve().parent.parent))
from training.training_config import BNB_CONFIG  # noqa: E402


def get_bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        **BNB_CONFIG,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_tokenizer(model_path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = (
        "right"  # right-padding for training (left-padding is a generation-time thing)
    )
    return tokenizer


def load_quantized_base(model_path):
    """Load a model in 4-bit NF4 for QLoRA training or inference."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        quantization_config=get_bnb_config(),
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
    )
    return model


def load_full_precision_base(model_path):
    """Load a model in bf16, no quantization. Only used for merging LoRA
    weights into the base — merging into 4-bit weights isn't reliable."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=True,
    )
    return model


def attach_lora(model, lora_cfg: dict):
    """Prepare a quantized model for k-bit training and attach a fresh LoRA adapter."""
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(**lora_cfg)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def save_adapter(model, tokenizer, output_dir):
    """Save adapter weights only (not merged) — standard for QLoRA outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✓ Adapter saved to {output_dir}")


def merge_adapter_and_save(base_model_path, adapter_path, output_dir):
    """
    Merge a LoRA adapter into its base model and save full weights.

    Used once, after SFT, to produce a merged model that DPO training
    treats as its new base. This lets DPOTrainer's built-in
    disable-adapters trick serve as the reference model (= the SFT
    model) without holding two full models in memory at once.

    Loads in bf16 (not 4-bit) since merging into quantized weights is
    unreliable — needs more VRAM than training, but only briefly.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"✓ Merged model already exists at {output_dir}, skipping merge")
        return

    print(f"Loading base in bf16 for merge: {base_model_path}")
    base = load_full_precision_base(base_model_path)
    tokenizer = load_tokenizer(base_model_path)

    print(f"Loading adapter to merge: {adapter_path}")
    merged = PeftModel.from_pretrained(base, str(adapter_path))
    merged = merged.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"✓ Merged model saved to {output_dir}")

    # Free VRAM before the caller loads the quantized version for training
    del base, merged
    torch.cuda.empty_cache()
