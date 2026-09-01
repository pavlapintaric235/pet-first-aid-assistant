import pytest

from src.retrieval.hybrid_search import (
    HybridSearch,
    reciprocal_rank_fusion,
)


def make_result(
    document_id,
    method,
    score,
):
    return {
        "id": document_id,
        "source_id": f"source_{document_id}",
        "publisher": "Test Veterinary Publisher",
        "title": f"Document {document_id}",
        "url": "https://example.com/article",
        "species": ["dog", "cat"],
        "topics": ["emergency"],
        "content": f"Content for {document_id}",
        "retrieval_method": method,
        "retrieval_score": score,
    }


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(
        self,
        query,
        num_results=5,
        species=None,
    ):
        self.calls.append(
            {
                "query": query,
                "num_results": num_results,
                "species": species,
            }
        )

        return self.results[:num_results]


def test_rrf_ranks_document_present_in_both_lists_first():
    keyword_results = [
        make_result("shared", "keyword", 8.0),
        make_result("keyword_only", "keyword", 7.0),
    ]

    vector_results = [
        make_result("vector_only", "vector", 0.95),
        make_result("shared", "vector", 0.90),
    ]

    results = reciprocal_rank_fusion(
        result_lists=[
            keyword_results,
            vector_results,
        ],
        k=60,
        num_results=3,
    )

    assert results[0]["id"] == "shared"
    assert results[0]["retrieval_method"] == "hybrid"
    assert len(results[0]["component_results"]) == 2


def test_rrf_does_not_duplicate_documents():
    keyword_results = [
        make_result("shared", "keyword", 8.0),
    ]

    vector_results = [
        make_result("shared", "vector", 0.9),
    ]

    results = reciprocal_rank_fusion(
        result_lists=[
            keyword_results,
            vector_results,
        ]
    )

    assert len(results) == 1
    assert results[0]["id"] == "shared"


def test_rrf_limits_results():
    keyword_results = [
        make_result("first", "keyword", 3.0),
        make_result("second", "keyword", 2.0),
        make_result("third", "keyword", 1.0),
    ]

    results = reciprocal_rank_fusion(
        result_lists=[keyword_results],
        num_results=2,
    )

    assert len(results) == 2


def test_rrf_rejects_invalid_k():
    with pytest.raises(
        ValueError,
        match="k must be greater",
    ):
        reciprocal_rank_fusion(
            result_lists=[],
            k=0,
        )


def test_hybrid_search_calls_both_engines():
    keyword_engine = FakeSearch(
        [
            make_result("keyword", "keyword", 5.0),
        ]
    )

    vector_engine = FakeSearch(
        [
            make_result("vector", "vector", 0.9),
        ]
    )

    hybrid_engine = HybridSearch(
        keyword_search=keyword_engine,
        vector_search=vector_engine,
    )

    results = hybrid_engine.search(
        query="breathing emergency",
        num_results=2,
        species="dog",
    )

    assert len(keyword_engine.calls) == 1
    assert len(vector_engine.calls) == 1
    assert results


def test_hybrid_search_passes_query_and_species():
    keyword_engine = FakeSearch([])
    vector_engine = FakeSearch([])

    hybrid_engine = HybridSearch(
        keyword_search=keyword_engine,
        vector_search=vector_engine,
    )

    hybrid_engine.search(
        query="cat bleeding",
        num_results=3,
        species="cat",
    )

    keyword_call = keyword_engine.calls[0]
    vector_call = vector_engine.calls[0]

    assert keyword_call["query"] == "cat bleeding"
    assert vector_call["query"] == "cat bleeding"
    assert keyword_call["species"] == "cat"
    assert vector_call["species"] == "cat"


def test_hybrid_search_requests_extra_candidates():
    keyword_engine = FakeSearch([])
    vector_engine = FakeSearch([])

    hybrid_engine = HybridSearch(
        keyword_search=keyword_engine,
        vector_search=vector_engine,
        candidate_multiplier=3,
    )

    hybrid_engine.search(
        query="emergency",
        num_results=5,
    )

    assert (
        keyword_engine.calls[0]["num_results"]
        == 15
    )

    assert (
        vector_engine.calls[0]["num_results"]
        == 15
    )


def test_hybrid_search_rejects_invalid_result_count():
    hybrid_engine = HybridSearch(
        keyword_search=FakeSearch([]),
        vector_search=FakeSearch([]),
    )

    with pytest.raises(
        ValueError,
        match="num_results",
    ):
        hybrid_engine.search(
            query="emergency",
            num_results=0,
        )