import json
import re
from pathlib import Path

from textstat import flesch_reading_ease


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("data/datasetv1.jsonl")
CLEAN_FILE = Path("data/cleaned.jsonl")
FLAGGED_FILE = Path("data/flagged_for_review.jsonl")

MIN_WORDS = 20
MAX_WORDS = 500

# If one answer is much longer than the other, flag it.
MAX_LENGTH_RATIO = 4.0

# Difference in readability that is worth investigating.
READABILITY_DIFF_THRESHOLD = 20


# ============================================================
# BASIC UTILITIES
# ============================================================

def word_count(text):
    return len(text.split())


def normalize(text):
    """Normalize text for duplicate detection."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def jargon_density(text):
    """
    Very simple heuristic:
    words containing technical-looking patterns.
    
    This is only a flag, NOT a quality judgment.
    """
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]+\b", text)

    if not words:
        return 0.0

    jargon_patterns = [
        r"^[a-z]+ology$",
        r"^[a-z]+ization$",
        r"^[a-z]+tion$",
        r"^[a-z]+ity$",
        r"^[a-z]+ance$",
        r"^[a-z]+ence$",
    ]

    jargon_count = 0

    for word in words:
        if len(word) >= 10:
            jargon_count += 1
            continue

        for pattern in jargon_patterns:
            if re.match(pattern, word.lower()):
                jargon_count += 1
                break

    return jargon_count / len(words)


def example_signal(text):
    """
    Detect simple signals that an explanation contains
    an example or analogy.
    """
    patterns = [
        r"\bfor example\b",
        r"\bfor instance\b",
        r"\bsuch as\b",
        r"\bimagine\b",
        r"\bthink of\b",
        r"\blike\b",
        r"\bsuppose\b",
        r"\bconsider\b",
    ]

    text_lower = text.lower()

    return sum(
        bool(re.search(pattern, text_lower))
        for pattern in patterns
    )


def contains_non_english(text):
    """
    Rough check for non-ASCII-heavy text.

    This is deliberately conservative.
    It only flags text with a substantial amount
    of non-ASCII content.
    """
    if not text:
        return True

    non_ascii = sum(ord(c) > 127 for c in text)

    return (non_ascii / len(text)) > 0.10


# ============================================================
# PAIR ANALYSIS
# ============================================================

def analyze_pair(example):
    prompt = str(example.get("prompt", "")).strip()
    chosen = str(example.get("chosen", "")).strip()
    rejected = str(example.get("rejected", "")).strip()

    flags = []

    # -----------------------------
    # Required fields
    # -----------------------------

    if not prompt:
        flags.append("empty_prompt")

    if not chosen:
        flags.append("empty_chosen")

    if not rejected:
        flags.append("empty_rejected")

    if not chosen or not rejected:
        return flags

    # -----------------------------
    # Identical answers
    # -----------------------------

    if normalize(chosen) == normalize(rejected):
        flags.append("identical_chosen_rejected")

    # -----------------------------
    # Word counts
    # -----------------------------

    chosen_words = word_count(chosen)
    rejected_words = word_count(rejected)

    if chosen_words < MIN_WORDS:
        flags.append("chosen_too_short")

    if rejected_words < MIN_WORDS:
        flags.append("rejected_too_short")

    if chosen_words > MAX_WORDS:
        flags.append("chosen_too_long")

    if rejected_words > MAX_WORDS:
        flags.append("rejected_too_long")

    # -----------------------------
    # Length imbalance
    # -----------------------------

    smaller = max(min(chosen_words, rejected_words), 1)
    larger = max(chosen_words, rejected_words)

    if larger / smaller > MAX_LENGTH_RATIO:
        flags.append("large_length_imbalance")

    # -----------------------------
    # Readability
    # -----------------------------

    try:
        chosen_readability = flesch_reading_ease(chosen)
        rejected_readability = flesch_reading_ease(rejected)

        readability_diff = (
            chosen_readability - rejected_readability
        )

        # Chosen being substantially harder to read
        # is suspicious for our objective.
        if readability_diff < -READABILITY_DIFF_THRESHOLD:
            flags.append("chosen_much_harder_to_read")

    except Exception:
        flags.append("readability_error")

    # -----------------------------
    # Jargon
    # -----------------------------

    chosen_jargon = jargon_density(chosen)
    rejected_jargon = jargon_density(rejected)

    if chosen_jargon > rejected_jargon * 2:
        flags.append("chosen_much_more_jargon")

    # -----------------------------
    # Examples / analogies
    # -----------------------------

    chosen_examples = example_signal(chosen)
    rejected_examples = example_signal(rejected)

    if (
        rejected_examples > chosen_examples
        and rejected_examples >= 2
    ):
        flags.append("rejected_has_more_example_signals")

    # -----------------------------
    # Non-English / unusual text
    # -----------------------------

    if contains_non_english(chosen):
        flags.append("chosen_non_english_content")

    if contains_non_english(rejected):
        flags.append("rejected_non_english_content")

    return flags


# ============================================================
# MAIN
# ============================================================

def main():

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)

    clean_examples = []
    flagged_examples = []

    seen_prompts = set()
    seen_pairs = set()

    total = 0

    with INPUT_FILE.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            total += 1

            # -----------------------------
            # JSON validation
            # -----------------------------

            try:
                example = json.loads(line)

            except json.JSONDecodeError:
                print(
                    f"[SKIP] Line {line_number}: invalid JSON"
                )
                continue

            # -----------------------------
            # Duplicate prompt
            # -----------------------------

            prompt = normalize(
                str(example.get("prompt", ""))
            )

            pair_key = (
                prompt,
                normalize(str(example.get("chosen", ""))),
                normalize(str(example.get("rejected", ""))),
            )

            if pair_key in seen_pairs:
                continue

            seen_pairs.add(pair_key)

            flags = analyze_pair(example)

            # -----------------------------
            # Duplicate prompt detection
            # -----------------------------

            if prompt in seen_prompts:
                flags.append("duplicate_prompt")

            seen_prompts.add(prompt)

            # -----------------------------
            # Separate clean vs flagged
            # -----------------------------

            if flags:
                example["_cleaning_flags"] = flags
                flagged_examples.append(example)

            else:
                clean_examples.append(example)

    # ========================================================
    # SAVE
    # ========================================================

    with CLEAN_FILE.open("w", encoding="utf-8") as f:

        for example in clean_examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    with FLAGGED_FILE.open("w", encoding="utf-8") as f:

        for example in flagged_examples:

            f.write(
                json.dumps(
                    example,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("DATASET CLEANING SUMMARY")
    print("=" * 60)

    print(f"Original examples:  {total}")
    print(f"Clean examples:     {len(clean_examples)}")
    print(f"Flagged examples:   {len(flagged_examples)}")

    if total:
        print(
            f"Flagged percentage: "
            f"{len(flagged_examples) / total * 100:.2f}%"
        )

    print("\nOutput:")
    print(f"  Clean:   {CLEAN_FILE}")
    print(f"  Flagged: {FLAGGED_FILE}")

    print("\nFlag distribution:")

    flag_counts = {}

    for example in flagged_examples:

        for flag in example["_cleaning_flags"]:

            flag_counts[flag] = (
                flag_counts.get(flag, 0) + 1
            )

    for flag, count in sorted(
        flag_counts.items(),
        key=lambda x: -x[1]
    ):
        print(f"  {flag}: {count}")


if __name__ == "__main__":
    main()