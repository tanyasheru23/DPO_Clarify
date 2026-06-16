# import json
# from pathlib import Path
# from unittest.mock import patch

# import pytest
from src.utils import clean_html, is_relevant
# from src.save_dataset import validate_pair
# from src.synthetic_pairs import build_synthetic_pairs
# from src.eli5_pairs import build_eli5_pairs
# from src.stack_exchange_pairs import build_stackexchange_pairs


##############################
# utils.py
##############################


def test_clean_html():
    html = "<p>Hello</p><b>World</b>"
    result = clean_html(html)
    assert result == "Hello World"


def test_is_relevant_true():
    assert is_relevant("How does gradient descent work?")


def test_is_relevant_false():
    assert not is_relevant("What is my favourite movie?")
