# import json
# from pathlib import Path
# from unittest.mock import patch

# import pytest
from src.utils import clean_html, get_se_domain, ALLOWED_DOMAINS
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


def test_get_se_domain_list():

    metadata = [
        "https://math.stackexchange.com/questions/123",
        "https://math.stackexchange.com",
        "https://math.stackexchange.com/users/1"
    ]

    assert get_se_domain(metadata) == "math"


def test_get_se_domain_json_string():

    metadata = """
    [
      "https://physics.stackexchange.com/questions/456",
      "https://physics.stackexchange.com",
      "https://physics.stackexchange.com/users/2"
    ]
    """

    assert get_se_domain(metadata) == "physics"


def test_get_se_domain_invalid_input():

    metadata = []

    assert get_se_domain(metadata) is None


def test_get_se_domain_none():

    assert get_se_domain(None) is None

def test_allowed_domains():

    assert "ai" in ALLOWED_DOMAINS

    assert "datascience" in ALLOWED_DOMAINS

    assert "softwareengineering" in ALLOWED_DOMAINS
