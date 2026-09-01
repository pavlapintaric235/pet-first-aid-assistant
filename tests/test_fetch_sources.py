import json

import pytest

from src.ingestion.fetch_sources import (
    calculate_content_hash,
    extract_main_text,
    find_source,
    load_source_catalog,
)


def test_load_source_catalog(tmp_path):
    catalog_path = tmp_path / "sources.json"

    sources = [
        {
            "source_id": "example_source",
            "publisher": "Example Publisher",
            "title": "Example Veterinary Article",
            "url": "https://example.com/article",
            "source_type": "web_page",
            "species": ["dog"],
            "topics": ["first_aid"],
        }
    ]

    catalog_path.write_text(
        json.dumps(sources),
        encoding="utf-8",
    )

    loaded_sources = load_source_catalog(catalog_path)

    assert loaded_sources == sources


def test_load_source_catalog_rejects_missing_fields(
    tmp_path,
):
    catalog_path = tmp_path / "sources.json"

    catalog_path.write_text(
        json.dumps(
            [
                {
                    "source_id": "incomplete_source",
                    "title": "Incomplete Source",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required fields"):
        load_source_catalog(catalog_path)


def test_find_source_returns_matching_source():
    sources = [
        {
            "source_id": "first",
            "title": "First source",
        },
        {
            "source_id": "second",
            "title": "Second source",
        },
    ]

    result = find_source(sources, "second")

    assert result["title"] == "Second source"


def test_find_source_rejects_unknown_id():
    sources = [
        {
            "source_id": "known_source",
        }
    ]

    with pytest.raises(ValueError, match="Unknown source ID"):
        find_source(sources, "missing_source")


def test_extract_main_text_removes_navigation_and_scripts():
    repeated_paragraph = (
        "This paragraph contains veterinary first-aid "
        "information for an injured animal. "
    ) * 8

    html = f"""
    <html>
        <head>
            <title>Test page</title>
            <script>console.log("not article content")</script>
        </head>
        <body>
            <nav>Navigation should not be included</nav>
            <main>
                <article>
                    <h1>Emergency First Aid</h1>
                    <p>{repeated_paragraph}</p>
                    <p>Contact a veterinarian immediately.</p>
                </article>
            </main>
            <footer>Footer should not be included</footer>
        </body>
    </html>
    """

    result = extract_main_text(html)

    assert "Emergency First Aid" in result
    assert "Contact a veterinarian immediately." in result
    assert "Navigation should not be included" not in result
    assert "Footer should not be included" not in result
    assert "console.log" not in result


def test_content_hash_is_stable():
    content = "Veterinary first-aid content"

    first_hash = calculate_content_hash(content)
    second_hash = calculate_content_hash(content)

    assert first_hash == second_hash
    assert len(first_hash) == 64