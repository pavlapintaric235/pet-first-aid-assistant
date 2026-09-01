from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup, Tag


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CATALOG_PATH = PROJECT_ROOT / "data" / "source_catalog.json"
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

REQUEST_TIMEOUT_SECONDS = 30

REQUEST_HEADERS = {
    "User-Agent": (
        "PetFirstAidAssistant/0.1 "
        "(educational LLM Zoomcamp project; source attribution preserved)"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REMOVABLE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "form",
    "button",
    "nav",
    "footer",
    "aside",
}

CONTENT_TAGS = {
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "li",
    "blockquote",
}


def load_source_catalog(
    catalog_path: Path = SOURCE_CATALOG_PATH,
) -> list[dict[str, Any]]:
    """Load and validate the veterinary source catalogue."""

    if not catalog_path.exists():
        raise FileNotFoundError(
            f"Source catalogue was not found at {catalog_path}"
        )

    with catalog_path.open("r", encoding="utf-8") as file:
        sources = json.load(file)

    if not isinstance(sources, list):
        raise ValueError("The source catalogue must contain a JSON list.")

    required_fields = {
        "source_id",
        "publisher",
        "title",
        "url",
        "source_type",
        "species",
        "topics",
    }

    for position, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(
                f"Source at position {position} must be a JSON object."
            )

        missing_fields = required_fields - source.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Source {position} is missing required fields: {missing}"
            )

    return sources


def find_source(
    sources: list[dict[str, Any]],
    source_id: str,
) -> dict[str, Any]:
    """Find one source in the catalogue by its stable ID."""

    for source in sources:
        if source["source_id"] == source_id:
            return source

    available_ids = ", ".join(
        sorted(source["source_id"] for source in sources)
    )

    raise ValueError(
        f"Unknown source ID: {source_id}. "
        f"Available source IDs: {available_ids}"
    )


def create_http_session() -> requests.Session:
    """Create an HTTP session with the project's identifying headers."""

    session = requests.Session()
    session.headers.update(REQUEST_HEADERS)

    return session


def download_html(
    source: dict[str, Any],
    session: requests.Session | None = None,
) -> tuple[str, str]:
    """Download a source page and return its HTML and final URL."""

    active_session = session or create_http_session()

    response = active_session.get(
        source["url"],
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "").lower()

    if "text/html" not in content_type:
        raise ValueError(
            f"Expected an HTML page for {source['source_id']}, "
            f"but received Content-Type: {content_type or 'unknown'}"
        )

    return response.text, response.url


def remove_unwanted_elements(soup: BeautifulSoup) -> None:
    """Remove elements that should not become retrieval documents."""

    for tag_name in REMOVABLE_TAGS:
        for element in soup.find_all(tag_name):
            element.decompose()

    unwanted_selectors = [
        "[aria-hidden='true']",
        "[role='navigation']",
        "[role='banner']",
        "[role='contentinfo']",
        ".cookie",
        ".cookies",
        ".cookie-banner",
        ".breadcrumb",
        ".breadcrumbs",
        ".newsletter",
        ".social-share",
        ".related-content",
        ".advertisement",
        ".advertising",
        "#cookie-banner",
    ]

    for selector in unwanted_selectors:
        for element in soup.select(selector):
            element.decompose()


def find_main_container(soup: BeautifulSoup) -> Tag:
    """Find the element most likely to contain the article content."""

    selectors = [
        "main article",
        "article",
        "main",
        "[role='main']",
        ".article-content",
        ".content-body",
        ".page-content",
        "#main-content",
        "#content",
    ]

    for selector in selectors:
        container = soup.select_one(selector)

        if isinstance(container, Tag):
            return container

    if soup.body is None:
        raise ValueError("The downloaded page does not contain a body.")

    return soup.body


def normalize_text(value: str) -> str:
    """Normalize whitespace while keeping the original wording."""

    return re.sub(r"\s+", " ", value).strip()


def extract_main_text(html: str) -> str:
    """Extract readable main content from an HTML document."""

    soup = BeautifulSoup(html, "lxml")
    remove_unwanted_elements(soup)

    container = find_main_container(soup)

    text_blocks: list[str] = []
    previously_seen: set[str] = set()

    for element in container.find_all(CONTENT_TAGS):
        text = normalize_text(
            element.get_text(separator=" ", strip=True)
        )

        if not text:
            continue

        if text in previously_seen:
            continue

        previously_seen.add(text)
        text_blocks.append(text)

    extracted_text = "\n\n".join(text_blocks).strip()

    if len(extracted_text) < 500:
        raise ValueError(
            "The extracted source content is unexpectedly short. "
            "The page structure may require a source-specific extractor."
        )

    return extracted_text


def calculate_content_hash(content: str) -> str:
    """Calculate a stable SHA-256 hash for change detection."""

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def create_raw_record(
    source: dict[str, Any],
    final_url: str,
    content: str,
) -> dict[str, Any]:
    """Create the structured representation saved to raw data."""

    retrieved_at = datetime.now(timezone.utc).isoformat()

    return {
        "source_id": source["source_id"],
        "publisher": source["publisher"],
        "title": source["title"],
        "original_url": source["url"],
        "final_url": final_url,
        "source_type": source["source_type"],
        "authority_level": source.get("authority_level"),
        "species": source["species"],
        "topics": source["topics"],
        "language": source.get("language", "en"),
        "retrieved_at": retrieved_at,
        "content_hash": calculate_content_hash(content),
        "content_length": len(content),
        "content": content,
    }


def save_raw_record(
    record: dict[str, Any],
    output_directory: Path = RAW_DATA_DIRECTORY,
) -> Path:
    """Save one raw source record as formatted JSON."""

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory / f"{record['source_id']}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def fetch_source(source_id: str) -> Path:
    """Download, extract, structure, and save one source."""

    sources = load_source_catalog()
    source = find_source(sources, source_id)

    html, final_url = download_html(source)
    content = extract_main_text(html)

    record = create_raw_record(
        source=source,
        final_url=final_url,
        content=content,
    )

    return save_raw_record(record)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Download one approved veterinary source and save "
            "its extracted content."
        )
    )

    parser.add_argument(
        "--source-id",
        required=True,
        help="Stable source_id from data/source_catalog.json",
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line ingestion operation."""

    arguments = parse_arguments()
    output_path = fetch_source(arguments.source_id)

    relative_output_path = output_path.relative_to(
        PROJECT_ROOT
    )

    print(
        f"Source downloaded successfully: "
        f"{relative_output_path}"
    )


if __name__ == "__main__":
    main()