import numpy as np
import pytest

from src.retrieval.mmr_search import (
    MMRSearch,
)
from src.retrieval.vector_search import (
    VectorSearch,
)


class FakeEmbedder:
    def encode(
        self,
        text,
        normalize=True,
    ):
        return np.array(
            [
                1.0,
                0.0,
            ],
            dtype=np.float32,
        )


def make_document(
    document_id,
    source_id,
):
    return {
        "id": document_id,
        "source_id": source_id,
        "publisher": "Test Publisher",
        "title": document_id,
        "url": (
            f"https://example.com/"
            f"{document_id}"
        ),
        "species": [
            "dog",
            "cat",
        ],
        "topics": ["test"],
        "content": (
            f"Content for "
            f"{document_id}."
        ),
    }


@pytest.fixture
def vector_search():
    documents = [
        make_document(
            "a",
            "source_a",
        ),
        make_document(
            "b",
            "source_b",
        ),
        make_document(
            "c",
            "source_c",
        ),
    ]

    embeddings = np.array(
        [
            [
                0.90,
                0.4358899,
            ],
            [
                0.89,
                0.4559605,
            ],
            [
                0.80,
                -0.60,
            ],
        ],
        dtype=np.float32,
    )

    return VectorSearch(
        documents=documents,
        embeddings=embeddings,
        embedder=FakeEmbedder(),
    )


def test_lambda_one_reproduces_vector_ranking(
    vector_search,
):
    baseline = vector_search.search(
        query="test query",
        num_results=3,
    )

    mmr = MMRSearch(
        vector_search=vector_search,
        lambda_mult=1.0,
        candidate_multiplier=2,
    )

    results = mmr.search(
        query="test query",
        num_results=3,
    )

    assert [
        result["id"]
        for result in results
    ] == [
        result["id"]
        for result in baseline
    ]


def test_mmr_can_prefer_less_redundant_candidate(
    vector_search,
):
    mmr = MMRSearch(
        vector_search=vector_search,
        lambda_mult=0.75,
        candidate_multiplier=2,
    )

    results = mmr.search(
        query="test query",
        num_results=3,
    )

    assert results[0]["id"] == "a"
    assert results[1]["id"] == "c"
    assert results[2]["id"] == "b"


def test_mmr_preserves_vector_score_metadata(
    vector_search,
):
    mmr = MMRSearch(
        vector_search=vector_search,
        lambda_mult=0.75,
    )

    result = mmr.search(
        query="test query",
        num_results=1,
    )[0]

    assert (
        result["retrieval_method"]
        == "vector_mmr"
    )

    assert result[
        "mmr_lambda"
    ] == pytest.approx(0.75)

    assert "vector_score" in result
    assert "mmr_score" in result
    assert "mmr_redundancy" in result


def test_mmr_respects_species_filter(
    vector_search,
):
    vector_search.documents[
        1
    ]["species"] = ["cat"]

    vector_search.documents[
        2
    ]["species"] = ["cat"]

    mmr = MMRSearch(
        vector_search=vector_search,
    )

    results = mmr.search(
        query="test query",
        num_results=3,
        species="dog",
    )

    assert [
        result["id"]
        for result in results
    ] == ["a"]


def test_mmr_rejects_invalid_lambda(
    vector_search,
):
    with pytest.raises(
        ValueError,
        match="between 0.0 and 1.0",
    ):
        MMRSearch(
            vector_search=vector_search,
            lambda_mult=1.1,
        )


def test_mmr_rejects_invalid_candidate_multiplier(
    vector_search,
):
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        MMRSearch(
            vector_search=vector_search,
            candidate_multiplier=0,
        )