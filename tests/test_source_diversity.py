import pytest

from src.retrieval.source_diversity import (
    SourceDiversifiedSearch,
    diversify_results,
)


def make_result(
    document_id,
    source_id,
):
    return {
        "id": document_id,
        "source_id": source_id,
        "retrieval_score": 1.0,
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


def test_diversify_results_limits_each_source():
    results = [
        make_result("a1", "source_a"),
        make_result("a2", "source_a"),
        make_result("a3", "source_a"),
        make_result("b1", "source_b"),
        make_result("c1", "source_c"),
    ]

    diversified = diversify_results(
        results=results,
        num_results=3,
        max_chunks_per_source=1,
    )

    assert [
        result["id"]
        for result in diversified
    ] == ["a1", "b1", "c1"]


def test_diversify_results_allows_two_chunks():
    results = [
        make_result("a1", "source_a"),
        make_result("a2", "source_a"),
        make_result("a3", "source_a"),
        make_result("b1", "source_b"),
    ]

    diversified = diversify_results(
        results=results,
        num_results=3,
        max_chunks_per_source=2,
    )

    assert [
        result["id"]
        for result in diversified
    ] == ["a1", "a2", "b1"]


def test_diversify_results_preserves_ranking():
    results = [
        make_result("first", "source_a"),
        make_result("second", "source_b"),
        make_result("third", "source_c"),
    ]

    diversified = diversify_results(
        results=results,
        num_results=3,
    )

    assert diversified == results


def test_diversify_results_rejects_invalid_limit():
    with pytest.raises(
        ValueError,
        match="max_chunks_per_source",
    ):
        diversify_results(
            results=[],
            num_results=5,
            max_chunks_per_source=0,
        )


def test_diversified_search_requests_larger_pool():
    base_search = FakeSearch(
        [
            make_result("a1", "source_a"),
            make_result("b1", "source_b"),
        ]
    )

    search = SourceDiversifiedSearch(
        search_engine=base_search,
        candidate_multiplier=10,
    )

    search.search(
        query="emergency",
        num_results=5,
        species="dog",
    )

    call = base_search.calls[0]

    assert call["num_results"] == 50
    assert call["query"] == "emergency"
    assert call["species"] == "dog"


def test_diversified_search_adds_metadata():
    base_search = FakeSearch(
        [
            make_result("a1", "source_a"),
            make_result("b1", "source_b"),
        ]
    )

    search = SourceDiversifiedSearch(
        search_engine=base_search,
        max_chunks_per_source=1,
    )

    results = search.search(
        query="emergency",
        num_results=2,
    )

    assert results[0]["source_diversified"] is True
    assert results[0]["max_chunks_per_source"] == 1