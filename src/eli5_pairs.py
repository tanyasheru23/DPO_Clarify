from config import MIN_ANSWER_LENGTH
from datasets import load_dataset

def build_eli5_pairs(target: int) -> list[dict]:
    """
    ELI5: every question has multiple answers with upvote scores.
    Chosen = highest scored answer, Rejected = lowest scored.
    """
    print("\n[2/3] Loading ELI5 dataset...")
    ds = load_dataset("dany0407/eli5_category", split="train")

    pairs = []
    skipped = 0

    for row in ds:
        if len(pairs) >= target:
            break

        answers      = row.get("answers", {})
        answer_texts = answers.get("text", [])
        answer_scores = answers.get("score", [])

        if len(answer_texts) < 2:
            skipped += 1
            continue

        # Zip and sort
        zipped = list(zip(answer_scores, answer_texts))
        zipped.sort(reverse=True)

        best_score,  chosen_text   = zipped[0]
        worst_score, rejected_text = zipped[-1]

        if best_score - worst_score < 5:   # ELI5 scores are lower than SE
            skipped += 1
            continue

        if len(chosen_text) < MIN_ANSWER_LENGTH or len(rejected_text) < MIN_ANSWER_LENGTH:
            skipped += 1
            continue

        question = row.get("title", "").strip()
        prompt   = f"Explain this in simple terms: {question}"

        pairs.append({
            "prompt":    prompt,
            "chosen":    chosen_text,
            "rejected":  rejected_text,
            "source":    "eli5",
            "score_gap": best_score - worst_score
        })
        # print(f"skipped: {skipped}")
        print(f"pairs: {len(pairs)}")

    print(f"{len(pairs)} ELI5 pairs built  (skipped {skipped})")
    return pairs
