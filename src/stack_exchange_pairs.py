from config import MIN_ANSWER_LENGTH, MIN_SCORE_GAP
from src.utils import is_relevant, clean_html
from datasets import load_dataset

def build_stackexchange_pairs(target: int) -> list[dict]:
    """
    Load HuggingFaceH4/stack-exchange-preferences.
    Each row has a question + list of answers with scores.
    We sort answers by score and take top as chosen, bottom as rejected.
    """
    print("\n[1/3] Loading StackExchange dataset...")
    ds = load_dataset(
        "HuggingFaceH4/stack-exchange-preferences",
        split="train",
        streaming=True,
    )

    pairs = []
    skipped = 0

    for row in ds:
        if len(pairs) >= target:
            break

        # Filter by keywords
        title = row.get("question")
        if not is_relevant(title):
            skipped += 1
            # print(f"skipped: {skipped}")
            continue

        answers = row.get("answers", [])
        if len(answers) < 2:
            skipped += 1
            continue

        # Sort by score descending
        answers_sorted = sorted(answers, key=lambda x: x.get("pm_score", 0), reverse=True)
        best   = answers_sorted[0]
        worst  = answers_sorted[-1]

        best_score  = best.get("pm_score", 0)
        worst_score = worst.get("pm_score", 0)

        # Quality gates
        if best_score - worst_score < MIN_SCORE_GAP:
            skipped += 1
            continue

        chosen_text   = clean_html(best.get("text", ""))
        rejected_text = clean_html(worst.get("text", ""))

        if len(chosen_text) < MIN_ANSWER_LENGTH or len(rejected_text) < MIN_ANSWER_LENGTH:
            skipped += 1
            continue

        # Build the prompt from the question title + body
        question_title = row.get("title", "").strip()
        question_body  = clean_html(row.get("question", ""))[:400]
        prompt = f"{question_title}\n\n{question_body}".strip()

        pairs.append({
            "prompt":   prompt,
            "chosen":   chosen_text,
            "rejected": rejected_text,
            "source":   "stackexchange",
            "score_gap": best_score - worst_score
        })
        print(f"pairs: {len(pairs)}")

    print(f"{len(pairs)} SE pairs built  (skipped {skipped})")
    return pairs
