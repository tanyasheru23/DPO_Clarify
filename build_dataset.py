"""
DPO Clarify Dataset Pipeline
=============================
Builds preference pairs (chosen=clear, rejected=confusing) from:
  1. StackExchange (score-based signal)
  2. ELI5 Reddit (upvote-based signal)
  3. Synthetic pairs via OpenAI API (concept coverage)

Output: data/dpo_clarify_dataset.jsonl  (prompt / chosen / rejected columns)
        data/dpo_clarify_dataset_hf/    (HuggingFace Dataset format for TRL)
"""

import json
import random
from src.stack_exchange_pairs import build_stackexchange_pairs
from src.eli5_pairs import build_eli5_pairs
from src.synthetic_pairs import build_synthetic_pairs
from src.save_dataset import save_dataset
from pathlib import Path
from config import TARGET_SE_PAIRS, TARGET_ELI5_PAIRS, TARGET_SYNTH_PAIRS, PARTS_DIR

if __name__ == "__main__":
    random.seed(42)
    all_pairs = []

    # se_pairs    = build_stackexchange_pairs(TARGET_SE_PAIRS)
    # eli5_pairs  = build_eli5_pairs(TARGET_ELI5_PAIRS)
    # synth_pairs = build_synthetic_pairs(TARGET_SYNTH_PAIRS)

    # all_pairs = se_pairs + eli5_pairs + synth_pairs
    # save_dataset(all_pairs)


    # Run Part 1
    se_pairs = build_stackexchange_pairs(TARGET_SE_PAIRS)

    # Save immediately — don't wait for parts 2 and 3
    with open(PARTS_DIR/"se_pairs.json", "w") as f:
        json.dump(se_pairs, f)
    print(f"✓ {PARTS_DIR}/se_pairs.json saved ({len(se_pairs)} pairs)")

    # Run Part 2
    eli5_pairs = build_eli5_pairs(TARGET_ELI5_PAIRS)

    with open(PARTS_DIR/"eli5_pairs.json", "w") as f:
        json.dump(eli5_pairs, f)
    print(f"✓ {PARTS_DIR}/eli5_pairs.json saved ({len(eli5_pairs)} pairs)")


    # Run Part 3
    synth_pairs = build_synthetic_pairs(TARGET_SYNTH_PAIRS)

    with open(PARTS_DIR/"synth_pairs.json", "w") as f:
        json.dump(synth_pairs, f)
    print(f"✓ {PARTS_DIR}/synth_pairs.json saved ({len(synth_pairs)} pairs)")

    Combine only after all three are done
    se_pairs    = json.load(open(PARTS_DIR/"se_pairs.json"))
    eli5_pairs  = json.load(open(PARTS_DIR/"eli5_pairs.json"))
    synth_pairs = json.load(open(PARTS_DIR/"synth_pairs.json"))

    all_pairs = se_pairs + eli5_pairs + synth_pairs
    save_dataset(all_pairs)

    print("\n✅ Dataset ready. Next step: run train_dpo.py")
