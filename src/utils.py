import re

# Few key words to filter the dataset for our specific choice
TOPIC_KEYWORDS = {
    "math": [
        "calculus",
        "derivative",
        "integral",
        "algebra",
        "matrix",
        "probability",
        "statistics",
    ],
    "cs": [
        "algorithm",
        "recursion",
        "complexity",
        "binary",
        "graph",
        "sorting",
        "dynamic programming",
    ],
    "physics": ["force", "energy", "quantum", "velocity", "thermodynamics", "entropy"],
    "ml": [
        "gradient",
        "neural",
        "loss function",
        "backprop",
        "overfitting",
        "embedding",
    ],
}

# Flatten into one set
ALL_KEYWORDS = {kw for words in TOPIC_KEYWORDS.values() for kw in words}


def is_relevant(title: str) -> bool:
    title_lower = title.lower()
    return any(kw in title_lower for kw in ALL_KEYWORDS)


def clean_html(text: str) -> str:
    """Strip basic HTML tags that appear in SE answers."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
