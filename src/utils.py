import re

ALLOWED_DOMAINS = [
    "ai",
    "biology",
    "chemistry",
    "cs",
    "datascience",
    "economics",
    "softwareengineering",
    "stats",
    "physics",
]


def get_se_domain(metadata: str | list) -> str | None:
    """Extract subdomain from SE URL e.g. 'math' from 'math.stackexchange.com'"""
    try:
        # metadata comes as a list of URLs, first one is the question URL
        if isinstance(metadata, str):
            import json

            metadata = json.loads(metadata)
        url = metadata[0]  # e.g. "https://math.stackexchange.com/questions/123"
        # extract the part before .stackexchange
        domain = url.split("//")[1].split(".stackexchange")[0]
        return domain
    except (IndexError, AttributeError):
        return None


def clean_html(text: str) -> str:
    """Strip basic HTML tags that appear in SE answers."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
