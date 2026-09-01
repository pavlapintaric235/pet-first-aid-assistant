import json

import pytest

from src.retrieval.keyword_search import (
    KeywordSearch,
    build_index_text,
    load_processed_documents,
    tokenize,
)


def make_document(
    document_id,
    content,
    species,
    topics,
    title,
):
    return {
        "id": document_id,
        "source_id": f"source_{document_id}",
        "publisher": "Test Veterinary Publisher",
        "title": title,
        "url": "https://example.com/article",
        "species": species,
        "topics": topics,
        "content": content,
    }


@pytest.fixture
def sample_documents():
    return [
        make_document(
            document_id="bleeding",
            title="Severe Bleeding First Aid",
            content=(
                "Apply direct pressure to a severely bleeding "
                "wound using clean gauze and contact an emergency "
                "veterinarian immediately."
            ),
            species=["dog", "cat"],
            topics=["bleeding", "wounds"],
        ),
        make_document(
            document_id="heatstroke",
            title="Heatstroke in Dogs",
            content=(
                "Move the overheated dog into a cooler area and "
                "begin controlled cooling while traveling to an "
                "emergency veterinary hospital."
            ),
            species=["dog"],
            topics=["heatstroke", "temperature"],
        ),
        make_document(
            document_id="cat_transport",
            title="Transporting an Injured Cat",
            content=(
                "Place the injured cat into a secure carrier and "
                "minimize movement during transport to the "
                "veterinary hospital."
            ),
            species=["cat"],
            topics=["transport", "injury"],
        ),
    ]


def test_tokenize_normalizes_text():
    result = tokenize(
        "Dog's BLEEDING wound: call a vet!"
    )

    assert result == [
        "dog",
        "s",
        "bleeding",
        "wound",
        "call",
        "a",
        "vet",
    ]


def test_build_index_text_weights_title_and_topics():
    document = make_document(
        document_id="test",
        title="Emergency Bleeding",
        content="Apply pressure.",
        species=["dog"],
        topics=["bleeding", "wound"],
    )

    index_text = build_index_text(document)

    assert index_text.count("Emergency Bleeding") == 3
    assert index_text.count("bleeding wound") == 4
    assert "Apply pressure." in index_text


def test_keyword_search_returns_relevant_result_first(
    sample_documents,
):
    search_engine = KeywordSearch(sample_documents)

    results = search_engine.search(
        "My pet has an uncontrolled bleeding wound"
    )

    assert results[0]["id"] == "bleeding"
    assert results[0]["retrieval_method"] == "keyword"
    assert isinstance(
        results[0]["retrieval_score"],
        float,
    )


def test_keyword_search_filters_by_species(
    sample_documents,
):
    search_engine = KeywordSearch(sample_documents)

    results = search_engine.search(
        query="injured animal transport",
        species="cat",
    )

    assert results
    assert all(
        "cat" in result["species"]
        for result in results
    )


def test_keyword_search_limits_number_of_results(
    sample_documents,
):
    search_engine = KeywordSearch(sample_documents)

    results = search_engine.search(
        query="emergency veterinary care",
        num_results=2,
    )

    assert len(results) == 2


def test_keyword_search_rejects_blank_query(
    sample_documents,
):
    search_engine = KeywordSearch(sample_documents)

    with pytest.raises(
        ValueError,
        match="search query",
    ):
        search_engine.search("   ")


def test_keyword_search_rejects_unknown_species(
    sample_documents,
):
    search_engine = KeywordSearch(sample_documents)

    with pytest.raises(
        ValueError,
        match="species must be",
    ):
        search_engine.search(
            query="emergency",
            species="rabbit",
        )


def test_load_processed_documents(tmp_path):
    documents_path = tmp_path / "documents.json"

    documents = [
        make_document(
            document_id="test",
            title="Test Article",
            content="Veterinary emergency information.",
            species=["dog", "cat"],
            topics=["emergency"],
        )
    ]

    documents_path.write_text(
        json.dumps(documents),
        encoding="utf-8",
    )

    loaded_documents = load_processed_documents(
        documents_path
    )

    assert loaded_documents == documents