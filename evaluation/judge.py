"""
judge.py
========
LLM-as-a-Judge using GPT-4o.
Scores each response on 5 dimensions, returns structured JSON.

Usage:
    from evaluation.judge import judge_response
    scores = judge_response(prompt, response)
"""

import json
import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing how well an AI explains concepts to a curious 15-year-old student with no prior knowledge of the topic.

You will receive a question and an AI-generated explanation. Score the explanation on 5 dimensions.

Return ONLY valid JSON with this exact structure:
{
  "clarity": <1-5>,
  "beginner_friendliness": <1-5>,
  "use_of_examples": <1-5>,
  "jargon_handling": <1-5>,
  "logical_flow": <1-5>,
  "reasoning": "<one sentence explaining your scores>"
}

Scoring guide:
- clarity: 1=very confusing, 5=crystal clear
- beginner_friendliness: 1=assumes expert knowledge, 5=anyone could understand
- use_of_examples: 1=no examples, 5=excellent concrete examples or analogies
- jargon_handling: 1=unexplained jargon everywhere, 5=all terms explained simply
- logical_flow: 1=jumps around confusingly, 5=builds understanding step by step"""

JUDGE_USER_TEMPLATE = """Question: {prompt}

Explanation: {response}

Score this explanation."""


def judge_response(prompt: str, response: str, retries: int = 3) -> dict:
    """
    Ask GPT-4o to score a single response.
    Returns dict with 5 scores + reasoning.
    Returns None if all retries fail.
    """
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=200,
                temperature=0,      # deterministic judging
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user",   "content": JUDGE_USER_TEMPLATE.format(
                        prompt=prompt,
                        response=response
                    )}
                ],
                response_format={"type": "json_object"}
            )
            raw    = completion.choices[0].message.content.strip()
            scores = json.loads(raw)

            # Validate expected keys exist
            expected = {"clarity", "beginner_friendliness", "use_of_examples",
                        "jargon_handling", "logical_flow", "reasoning"}
            if not expected.issubset(scores.keys()):
                raise ValueError(f"Missing keys in judge response: {scores.keys()}")

            # Compute overall score as average of 5 numeric dimensions
            numeric_keys = ["clarity", "beginner_friendliness", "use_of_examples",
                            "jargon_handling", "logical_flow"]
            scores["overall"] = round(
                sum(scores[k] for k in numeric_keys) / len(numeric_keys), 2
            )
            return scores

        except Exception as e:
            print(f"  ⚠ Judge attempt {attempt+1} failed: {e}")
            time.sleep(2)

    return None


def judge_all(responses: list[dict]) -> list[dict]:
    """
    Run judge on a list of response dicts.
    Each dict must have 'prompt' and 'response' keys.
    Adds 'judge_scores' key to each dict.
    """
    print(f"\nRunning LLM-as-judge on {len(responses)} responses...")
    for i, item in enumerate(responses):
        print(f"  [{i+1}/{len(responses)}] judging: {item['prompt'][:60]}...")
        scores = judge_response(item["prompt"], item["response"])
        item["judge_scores"] = scores
        time.sleep(0.5)    # gentle rate limiting

    return responses