import os
import json
import time
from config import OPENAI_MODEL
from openai import OpenAI  # pip install openai
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)  # uses OPENAI_API_KEY env var

# Concepts to cover — chosen to fill gaps in SE/ELI5 coverage
# CONCEPTS = [
#     # Math
#     "the chain rule in calculus",
#     "eigenvectors and eigenvalues",
#     "Bayes' theorem",
#     "gradient descent",
#     "the Central Limit Theorem",
#     "convexity in optimization",
#     "what a p-value actually means",
#     "the intuition behind logarithms",
#     # CS / ML
#     "how attention mechanisms work in transformers",
#     "why neural networks need non-linear activation functions",
#     "what overfitting is and why it happens",
#     "how backpropagation works",
#     "the difference between precision and recall",
#     "why we need regularization",
#     "what an embedding is",
#     "how a hash map works internally",
#     "recursion vs iteration — when to use which",
#     "what Big O notation actually measures",
#     # Physics / Science
#     "why entropy always increases",
#     "how quantum superposition works",
#     "what causes electric resistance",
#     "why the sky is blue",
#     "how vaccines train the immune system",
#     "what DNA actually does in a cell",
#     "how compound interest works",
#     "what causes inflation",
#     "the difference between correlation and causation",
#     "why randomized controlled trials matter",
#     "what a confidence interval means"
# ]

PROMPT_TEMPLATES = [
    "Explain {concept} to a curious student.",
    "How would you explain {concept} to someone hearing it for the first time?",
    "Break down {concept} in a way that actually makes sense.",
    "What is {concept}? Explain it clearly.",
    "I keep hearing about {concept} but don't get it. Can you explain?",
    "Give me a clear explanation of {concept}.",
    "Explain {concept} without jargon.",
    "Help me understand {concept}.",
    "What's the intuition behind {concept}?",
    "Explain {concept} like I'm smart but new to this topic.",
]

templates_str = "\n".join(f"- {t}" for t in PROMPT_TEMPLATES)

SYNTH_SYSTEM_PROMPT = f"""You are a dataset generator for training AI models to give clearer explanations.

Generate a preference pair for a random educational concept from any domain (math, physics, CS, biology, economics, etc). 

Pick a concept that is:
- genuinely tricky to understand for a beginner
- commonly misunderstood or poorly explained online
- not too niche, not too broad
- specific enough to explain in one response (not "machine learning" or "physics" or "biology")

Then generate:
- prompt: a natural question someone might actually ask about that concept. If required, use these prompt templates as reference: {templates_str}
- chosen: a clear explanation using analogy, concrete example first, no unexplained jargon
- rejected: a technically accurate but confusing explanation, definition-first, jargon-heavy

CRITICAL:
1. Both chosen and rejected must be FACTUALLY CORRECT
2. Vary the prompt phrasing naturally — don't always say "explain X"
3. Don't repeat concepts across calls

Return ONLY valid JSON:
{{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "concept": "..."
}}"""

def generate_synthetic_pair(retries: int = 3) -> dict | None:
    """Call OpenAI to generate one chosen/rejected pair for a concept."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=1000,
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
                    {"role": "user",   "content": f"Generate one QnA preference pair"}
                ],
                response_format={"type": "json_object"}  # enforces JSON output
            )
            raw = response.choices[0].message.content.strip()
            pair = json.loads(raw)
            pair["source"] = "synthetic"
            return pair
        except (json.JSONDecodeError, Exception) as e:
            print(f"!!! Attempt {attempt+1} failed: {e}")
            time.sleep(2)
    return None

def build_synthetic_pairs(target: int) -> list[dict]:
    """Generate synthetic pairs across all concepts, cycling if needed."""
    print("\n[3/3] Generating {target} synthetic pairs via OpenAI ({OPENAI_MODEL})...")
    pairs = []
    # concepts_cycled = CONCEPTS * ((target // len(CONCEPTS)) + 2)
    # random.shuffle(concepts_cycled)

    # for concept in concepts_cycled:
    #     if len(pairs) >= target:
    #         break
    #     print(f"-->{concept}")
    #     prompt = random.choice(PROMPT_TEMPLATES).format(concept=concept)
    #     pair = generate_synthetic_pair(prompt, concept)
    #     if pair:
    #         pairs.append(pair)
    #     time.sleep(0.5)  # gentle rate limiting

    seen_concepts = set()

    attempts = 0
    max_attempts = target * 2  # allow extra attempts for duplicates

    while len(pairs) < target and attempts < max_attempts:
        pair = generate_synthetic_pair()
        attempts += 1
        if pair and pair.get("concept", "").lower() not in seen_concepts:
            seen_concepts.add(pair["concept"])
            pairs.append(pair)
            print(f"[{len(pairs)}/{target}] || {pair.get('concept', 'unknown')} || {pair.get('prompt')}")
        time.sleep(0.5)

    print(f"{len(pairs)} synthetic pairs generated")
    return pairs
