"""
metrics.py
==========
Computes objective + heuristic metrics for a single response.
No API calls — all local computation.

Metrics:
  Readability (textstat):
    - flesch_reading_ease     higher = easier (60-70 is ideal)
    - flesch_kincaid_grade    grade level (7 = 7th grade)
    - smog_index              grade level estimate
    - coleman_liau_index      grade level estimate
    - automated_readability   grade level estimate

  Heuristic (custom):
    - word_count              total words in response
    - avg_sentence_length     words per sentence
    - jargon_density          fraction of words that are jargon
    - example_score           count of example/analogy signals
    - analogy_score           count of analogy signals
    - definition_first        1 if response starts with a formal definition
    - has_steps               1 if response uses numbered steps

  Semantic (bert_score):
    - bertscore_precision     token-level precision vs prompt
    - bertscore_recall        token-level recall vs prompt
    - bertscore_f1            harmonic mean — main number to report

  Formatting:
    - is_non_english          1 if response contains non-English characters
    - non_english_ratio       fraction of characters that are non-ASCII
    - has_repeated_prompt     1 if model just repeated the question back
    - formatting_error_rate   combined score (0=clean, 1=completely broken)

Install: pip install textstat bert-score
"""

import re
import textstat

# ─────────────────────────────────────────────
# JARGON LIST
# ─────────────────────────────────────────────

JARGON_WORDS = {
    # Math
    "bijective",
    "surjective",
    "injective",
    "isomorphism",
    "homeomorphism",
    "eigendecomposition",
    "stochastic",
    "deterministic",
    "heuristic",
    "asymptotic",
    "convergence",
    "divergence",
    "monotonic",
    # CS
    "polymorphism",
    "encapsulation",
    "instantiation",
    "idempotent",
    "memoization",
    "serialization",
    "deserialization",
    "instantiate",
    "immutable",
    "concurrency",
    "parallelism",
    "asynchronous",
    # ML
    "hyperparameter",
    "regularization",
    "backpropagation",
    "stochastic",
    "gradient",
    "optimization",
    "convergence",
    "epoch",
    "minibatch",
    "softmax",
    "relu",
    "sigmoid",
    "perceptron",
    "autoencoder",
    # Physics
    "thermodynamic",
    "electromagnetic",
    "eigenstate",
    "superposition",
    "wavefunction",
    "quantization",
    "relativistic",
    # Stats
    "heteroscedasticity",
    "autocorrelation",
    "multicollinearity",
    "frequentist",
    "bayesian",
    "posterior",
    "likelihood",
}

EXAMPLE_SIGNALS = [
    "for example",
    "for instance",
    "such as",
    "e.g.",
    "let's say",
    "imagine",
    "suppose",
    "consider",
    "think of",
    "picture this",
    "like when",
    "just like",
    "as an example",
]

ANALOGY_SIGNALS = [
    "like a",
    "like an",
    "similar to",
    "it's like",
    "think of it as",
    "imagine it as",
    "just like",
    "same as",
    "analogous to",
    "is similar to",
    "works like",
    "acts like",
]

DEFINITION_PATTERNS = [
    r"^[A-Z][a-z]+ is defined as",
    r"^[A-Z][a-z]+ refers to",
    r"^In mathematics,",
    r"^In computer science,",
    r"^[A-Z][a-z]+ is a (mathematical|statistical|computational|theoretical)",
]


# ─────────────────────────────────────────────
# READABILITY
# ─────────────────────────────────────────────


def compute_readability(text: str) -> dict:
    """Run all textstat readability metrics."""
    return {
        "flesch_reading_ease": round(textstat.flesch_reading_ease(text), 2),
        "flesch_kincaid_grade": round(textstat.flesch_kincaid_grade(text), 2),
        "smog_index": round(textstat.smog_index(text), 2),
        "coleman_liau_index": round(textstat.coleman_liau_index(text), 2),
        "automated_readability": round(textstat.automated_readability_index(text), 2),
    }


# ─────────────────────────────────────────────
# HEURISTICS
# ─────────────────────────────────────────────


def compute_heuristics(text: str) -> dict:
    """Compute custom heuristic metrics."""
    text_lower = text.lower()
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    avg_sentence_length = round(word_count / len(sentences) if sentences else 0, 2)
    jargon_count = sum(1 for w in words if w.lower().strip(".,;:") in JARGON_WORDS)
    jargon_density = round(jargon_count / word_count if word_count > 0 else 0, 4)
    example_score = sum(1 for signal in EXAMPLE_SIGNALS if signal in text_lower)
    analogy_score = sum(1 for signal in ANALOGY_SIGNALS if signal in text_lower)
    definition_first = int(
        any(re.match(pattern, text) for pattern in DEFINITION_PATTERNS)
    )
    has_steps = int(bool(re.search(r"\b(1\.|step 1|first,|firstly)", text_lower)))

    return {
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "jargon_density": jargon_density,
        "example_score": example_score,
        "analogy_score": analogy_score,
        "definition_first": definition_first,
        "has_steps": has_steps,
    }


# ─────────────────────────────────────────────
# BERTSCORE
# Measures semantic similarity between the response
# and the prompt — checks if the model actually
# answered the question or went off-topic
# ─────────────────────────────────────────────


def compute_bertscore(prompt: str, response: str) -> dict:
    """
    Compute BERTScore between prompt and response.
    Uses distilbert-base-uncased (small, fast, no GPU needed).
    High F1 = response is semantically relevant to the prompt.
    Low F1 = model went off-topic (or answered in wrong language).
    """
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            cands=[response],
            refs=[prompt],
            model_type="distilbert-base-uncased",  # small model, fast
            lang="en",
            verbose=False,
        )
        return {
            "bertscore_precision": round(P[0].item(), 4),
            "bertscore_recall": round(R[0].item(), 4),
            "bertscore_f1": round(F1[0].item(), 4),
        }
    except Exception as e:
        print(f"  ⚠ BERTScore failed: {e}")
        return {
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
        }


# ─────────────────────────────────────────────
# FORMATTING ERROR RATE
# Catches: Chinese/non-English output, prompt
# repetition, empty responses
# ─────────────────────────────────────────────


def compute_formatting(prompt: str, response: str) -> dict:
    """
    Detect formatting and language errors in response.

    is_non_english:      1 if Chinese or other non-Latin characters detected
    non_english_ratio:   fraction of chars that are non-ASCII (0=clean, 1=all foreign)
    has_repeated_prompt: 1 if response starts by repeating the question back
    formatting_error_rate: combined score — fraction of errors present (0=clean)
    """
    # Non-English detection — check for CJK characters (Chinese/Japanese/Korean)
    cjk_chars = sum(
        1 for c in response if "\u4e00" <= c <= "\u9fff" or "\u3040" <= c <= "\u30ff"
    )
    is_non_english = int(cjk_chars > 5)  # allow up to 5 stray chars

    # Non-ASCII ratio — broader than CJK, catches Arabic, Cyrillic etc
    non_ascii = sum(1 for c in response if ord(c) > 127)
    non_english_ratio = round(non_ascii / len(response) if response else 0, 4)

    # Prompt repetition — did model just echo the question?
    prompt_words = set(prompt.lower().split())
    response_start = set(response[: len(prompt)].lower().split())
    overlap = (
        len(prompt_words & response_start) / len(prompt_words) if prompt_words else 0
    )
    has_repeated_prompt = int(overlap > 0.7)  # >70% word overlap = repetition

    # Combined formatting error rate
    # Each issue contributes equally — average of binary flags + severity of ratio
    errors = [is_non_english, has_repeated_prompt, int(non_english_ratio > 0.3)]
    formatting_error_rate = round(sum(errors) / 3, 4)

    return {
        "is_non_english": is_non_english,
        "non_english_ratio": non_english_ratio,
        "has_repeated_prompt": has_repeated_prompt,
        "formatting_error_rate": formatting_error_rate,
    }


# ─────────────────────────────────────────────
# COMBINED
# ─────────────────────────────────────────────


def compute_all_metrics(prompt: str, text: str) -> dict:
    """
    Compute all metrics for a response.
    Now takes prompt too — needed for BERTScore and formatting checks.
    """
    return {
        **compute_readability(text),
        **compute_heuristics(text),
        **compute_bertscore(prompt, text),
        **compute_formatting(prompt, text),
    }
