# import json
# from pathlib import Path
# from unittest.mock import patch

# import pytest
from src.utils import clean_html
from evaluation.metrics import compute_formatting, compute_heuristics, compute_readability


##############################
# utils.py
##############################


def test_clean_html():

    html = "<p>Hello</p><b>World</b>"

    result = clean_html(html)

    assert result == "Hello World"

class TestComputeFormatting:
    def test_clean_english_response(self):
        result = compute_formatting("What is recursion?", "Recursion is when a function calls itself.")
        assert result["is_non_english"] == 0
        assert result["formatting_error_rate"] == 0.0

    def test_detects_chinese(self):
        result = compute_formatting("What is recursion?", "递归是一种函数调用自身的编程技术。这是一个很好的例子。")
        assert result["is_non_english"] == 1
        assert result["formatting_error_rate"] > 0

    def test_detects_repeated_prompt(self):
        prompt = "What is recursion and how does it work in programming"
        # response starts by repeating the prompt almost verbatim
        response = "What is recursion and how does it work in programming? Well let me explain..."
        result = compute_formatting(prompt, response)
        assert result["has_repeated_prompt"] == 1

    def test_non_english_ratio_clean(self):
        result = compute_formatting("prompt", "This is a clean English response.")
        assert result["non_english_ratio"] == 0.0

    def test_empty_response(self):
        result = compute_formatting("prompt", "")
        assert result["non_english_ratio"] == 0.0
        assert result["is_non_english"] == 0


class TestComputeHeuristics:
    def test_counts_example_signals(self):
        text = "For example, imagine you are climbing a hill. Think of it like gravity."
        result = compute_heuristics(text)
        assert result["example_score"] >= 1
        assert result["analogy_score"] >= 1

    def test_detects_definition_first(self):
        text = "In mathematics, a derivative is defined as the limit of the difference quotient."
        result = compute_heuristics(text)
        assert result["definition_first"] == 1

    def test_word_count(self):
        text = "one two three four five"
        result = compute_heuristics(text)
        assert result["word_count"] == 5

    def test_no_jargon_in_simple_text(self):
        text = "A dog is an animal that likes to run and play outside every day."
        result = compute_heuristics(text)
        assert result["jargon_density"] == 0.0

    def test_has_steps_detected(self):
        text = "Step 1: open the file. Step 2: read the contents."
        result = compute_heuristics(text)
        assert result["has_steps"] == 1


class TestComputeReadability:
    def test_returns_all_keys(self):
        text = "The cat sat on the mat. It was a very happy cat indeed."
        result = compute_readability(text)
        assert "flesch_reading_ease" in result
        assert "flesch_kincaid_grade" in result
        assert "smog_index" in result
        assert "coleman_liau_index" in result
        assert "automated_readability" in result

    def test_simple_text_high_reading_ease(self):
        # Very simple text should score high on reading ease
        text = "The dog ran. The cat sat. The sun is hot. I like pie."
        result = compute_readability(text)
        assert result["flesch_reading_ease"] > 70

    def test_complex_text_lower_reading_ease(self):
        # Jargon-heavy text should score lower
        text = """The eigendecomposition of a stochastic matrix demonstrates 
        asymptotic convergence properties inherent to thermodynamic equilibrium 
        states in heteroscedastic multivariate distributions."""
        result = compute_readability(text)
        assert result["flesch_reading_ease"] < 30
