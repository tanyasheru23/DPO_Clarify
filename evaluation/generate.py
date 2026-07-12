"""
generate.py
===========
Loads a model and generates responses for all eval prompts.
Saves responses to results/{model_name}_responses.json
 
Usage:
    python -m evaluation.generate --model base
    python -m evaluation.generate --model ./sft-output
    python -m evaluation.generate --model ./dpo-output
"""

import json
import argparse
import torch
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from time import sleep
from config import BASE_MODEL, RESULTS_DIR, EVAL_PROMPTS_PATH

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# LOAD MODEL
 
def load_model(model_arg: str):
    """
    Load base model or finetuned adapter.
    model_arg = "base" → loads BASE_MODEL from HuggingFace
    model_arg = "./path" → loads base + LoRA adapter from path
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Loading the model from local directory
     
    print(f"Loading base model: {BASE_MODEL}")
    base = AutoModelForCausalLM.from_pretrained(
        str(BASE_MODEL),
        quantization_config=bnb_config,
        device_map={"":0}, #Map to GPU
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(BASE_MODEL),
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
 
    adapter_path = Path(model_arg)

    if adapter_path.exists():
        print(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base, adapter_path)
    else:
        model = base
 
    model.eval()
    return model, tokenizer

# GENERATE RESPONSES

def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
) -> str:

    messages = [
        {
            "role": "system",
            "content": (
                "You are an educational AI assistant."
                "Answer the given question clearly, simply(easy to understand) and accurately."
                "Limit your response to about 120-150 words."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_promt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    ).to(model.device)

    start = time.time()

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=1.0,        # required when do_sample=False
            do_sample=False,        # greedy — deterministic, reproducible
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    latency = time.time() - start

    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    return {
        "response": response,
        "latency": round(latency, 3),
        "tokens_generated": len(generated_tokens),
    }

def generate_all(model_arg: str):
    """Generate responses for all prompts and save to results/."""
    # Load prompts
    prompts = json.loads(Path(EVAL_PROMPTS_PATH).read_text(encoding="utf-8"))
    print(f"Loaded {len(prompts)} prompts")
 
    # Load model
    model, tokenizer = load_model(model_arg)
     
    # Model name for output file
    model_name = "base" if model_arg == "base" else Path(model_arg).name
 
    responses = []

    for item in tqdm(prompts, desc="Generating"):

        try:

            result = generate_response(
                model,
                tokenizer,
                item["prompt"],
            )

            tqdm.write(
                f"{item['id']:02d} | "
                f"{result['latency']:.2f}s | "
                f"{result['tokens_generated']} tokens"
            )

            responses.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "response": result["response"],
                    "latency": result["latency"],
                    "tokens_generated": result["tokens_generated"],
                    "model": model_name,
                }
            )

        except Exception as e:

            print(f"\nError on prompt {item['id']}")

            print(e)

            responses.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "prompt": item["prompt"],
                    "response": "",
                    "latency": None,
                    "tokens_generated": 0,
                    "model": model_name,
                    "error": str(e),
                }
            )

    output_path = RESULTS_DIR / f"{model_name}_responses.json"

    output_path.write_text(
        json.dumps(
            responses,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\n✓ Saved {len(responses)} responses")
    print(f"✓ Output: {output_path}")

    return responses

# MAIN
 
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="base",
        help="'base' for baseline model, or path to finetuned adapter"
    )

    args = parser.parse_args()
    
    generate_all(args.model)