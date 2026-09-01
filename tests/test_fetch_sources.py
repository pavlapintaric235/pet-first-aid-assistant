import json

import pytest

from src.ingestion.fetch_sources import (
    calculate_content_hash,
    extract_main_text,
    extract_structured_content,
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

    loaded_sources = load_source_catalog(
        catalog_path
    )

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

    with pytest.raises(
        ValueError,
        match="missing required fields",
    ):
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

    result = find_source(
        sources,
        "second",
    )

    assert result["title"] == "Second source"


def test_find_source_rejects_unknown_id():
    sources = [
        {
            "source_id": "known_source",
        }
    ]

    with pytest.raises(
        ValueError,
        match="Unknown source ID",
    ):
        find_source(
            sources,
            "missing_source",
        )


def make_long_test_html() -> str:
    first_paragraph = (
        "This paragraph contains veterinary first-aid "
        "information for an injured animal. "
    ) * 8

    second_paragraph = (
        "Apply safe first aid while arranging immediate "
        "professional veterinary care. "
    ) * 8

    third_paragraph = (
        "Transport the animal carefully and minimize "
        "unnecessary movement during the emergency. "
    ) * 8

    return f"""
    <html>
        <head>
            <title>Test page</title>
            <script>
                console.log("not article content")
            </script>
        </head>
        <body>
            <nav>
                Navigation should not be included
            </nav>

            <main>
                <article>
                    <h1>Emergency First Aid</h1>
                    <p>{first_paragraph}</p>

                    <h2>Bleeding</h2>
                    <p>{second_paragraph}</p>

                    <h2>Safe Transport</h2>
                    <p>{third_paragraph}</p>
                </article>
            </main>

            <footer>
                Footer should not be included
            </footer>
        </body>
    </html>
    """


def test_extract_main_text_removes_unwanted_content():
    html = make_long_test_html()

    result = extract_main_text(html)

    assert "Emergency First Aid" in result
    assert "Bleeding" in result
    assert "Safe Transport" in result

    assert (
        "Navigation should not be included"
        not in result
    )
    assert (
        "Footer should not be included"
        not in result
    )
    assert "console.log" not in result


def test_extract_structured_content_preserves_sections():
    html = make_long_test_html()

    content, sections = extract_structured_content(
        html
    )

    assert len(sections) == 3

    assert sections[0]["heading"] == (
        "Emergency First Aid"
    )
    assert sections[0]["heading_level"] == 1
    assert sections[0]["heading_path"] == [
        "Emergency First Aid"
    ]

    assert sections[1]["heading"] == "Bleeding"
    assert sections[1]["heading_level"] == 2
    assert sections[1]["heading_path"] == [
        "Emergency First Aid",
        "Bleeding",
    ]

    assert sections[2]["heading"] == (
        "Safe Transport"
    )
    assert sections[2]["section_index"] == 2

    assert sections[1]["word_count"] > 0
    assert "Bleeding" in content


def test_extract_structured_content_uses_introduction():
    paragraph = (
        "Emergency veterinary information should remain "
        "connected to its surrounding safety instructions. "
    ) * 15

    html = f"""
    <html>
        <body>
            <main>
                <p>{paragraph}</p>
            </main>
        </body>
    </html>
    """

    _, sections = extract_structured_content(
        html
    )

    assert len(sections) == 1
    assert sections[0]["heading"] == "Introduction"
    assert sections[0]["heading_level"] == 0
    assert sections[0]["heading_path"] == [
        "Introduction"
    ]


def test_extract_structured_content_rejects_short_page():
    html = """
    <html>
        <body>
            <main>
                <h1>Short page</h1>
                <p>Not enough usable veterinary content.</p>
            </main>
        </body>
    </html>
    """

    with pytest.raises(
        ValueError,
        match="unexpectedly short",
    ):
        extract_structured_content(html)


def test_content_hash_is_stable():
    content = "Veterinary first-aid content"

    first_hash = calculate_content_hash(
        content
    )
    second_hash = calculate_content_hash(
        content
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64