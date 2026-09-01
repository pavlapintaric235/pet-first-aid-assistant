import numpy as np
import pytest

from src.retrieval.vector_search import (
    VectorSearch,
    build_embedding_matrix,
    build_embedding_text,
    load_embedding_matrix,
    save_embedding_matrix,
)


class FakeEmbedder:
    def encode(
        self,
        text,
        normalize=True,
    ):
        vectors = {
            "bleeding query": np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            "heat query": np.array(
                [0.0, 1.0, 0.0],
                dtype=np.float32,
            ),
            "transport query": np.array(
                [0.0, 0.0, 1.0],
                dtype=np.float32,
            ),
        }

        return vectors.get(
            text,
            np.array(
                [0.5, 0.5, 0.0],
                dtype=np.float32,
            ),
        )

    def encode_batch(
        self,
        texts,
        normalize=True,
    ):
        vectors = []

        for index, _ in enumerate(texts):
            vector = np.zeros(3, dtype=np.float32)
            vector[index % 3] = 1.0
            vectors.append(vector)

        return np.array(vectors, dtype=np.float32)


def make_document(
    document_id,
    species,
    title,
    topics,
    content,
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
            species=["dog", "cat"],
            title="Severe Bleeding",
            topics=["bleeding"],
            content="Apply direct pressure to the wound.",
        ),
        make_document(
            document_id="heatstroke",
            species=["dog"],
            title="Heatstroke",
            topics=["heatstroke"],
            content="Begin controlled cooling.",
        ),
        make_document(
            document_id="transport",
            species=["cat"],
            title="Transport",
            topics=["injury", "transport"],
            content="Use a secure carrier.",
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def test_build_embedding_text():
    document = make_document(
        document_id="test",
        species=["dog"],
        title="Emergency Bleeding",
        topics=["bleeding", "wounds"],
        content="Apply direct pressure.",
    )

    result = build_embedding_text(document)

    assert "Title: Emergency Bleeding" in result
    assert "Topics: bleeding, wounds" in result
    assert "Content: Apply direct pressure." in result


def test_build_embedding_matrix(sample_documents):
    matrix = build_embedding_matrix(
        documents=sample_documents,
        embedder=FakeEmbedder(),
        batch_size=2,
    )

    assert matrix.shape == (3, 3)
    assert matrix.dtype == np.float32


def test_save_and_load_embedding_matrix(
    tmp_path,
    sample_embeddings,
):
    output_path = tmp_path / "embeddings.npy"

    save_embedding_matrix(
        embeddings=sample_embeddings,
        output_path=output_path,
    )

    loaded = load_embedding_matrix(output_path)

    assert np.array_equal(
        loaded,
        sample_embeddings,
    )


def test_vector_search_returns_most_similar_document(
    sample_documents,
    sample_embeddings,
):
    search_engine = VectorSearch(
        documents=sample_documents,
        embeddings=sample_embeddings,
        embedder=FakeEmbedder(),
    )

    results = search_engine.search(
        query="bleeding query"
    )

    assert results[0]["id"] == "bleeding"
    assert results[0]["retrieval_method"] == "vector"
    assert results[0]["retrieval_score"] == pytest.approx(1.0)


def test_vector_search_filters_species(
    sample_documents,
    sample_embeddings,
):
    search_engine = VectorSearch(
        documents=sample_documents,
        embeddings=sample_embeddings,
        embedder=FakeEmbedder(),
    )

    results = search_engine.search(
        query="heat query",
        species="cat",
    )

    assert results
    assert all(
        "cat" in result["species"]
        for result in results
    )


def test_vector_search_limits_results(
    sample_documents,
    sample_embeddings,
):
    search_engine = VectorSearch(
        documents=sample_documents,
        embeddings=sample_embeddings,
        embedder=FakeEmbedder(),
    )

    results = search_engine.search(
        query="transport query",
        num_results=2,
    )

    assert len(results) == 2


def test_vector_search_rejects_dimension_mismatch(
    sample_documents,
):
    embeddings = np.ones(
        (3, 4),
        dtype=np.float32,
    )

    search_engine = VectorSearch(
        documents=sample_documents,
        embeddings=embeddings,
        embedder=FakeEmbedder(),
    )

    with pytest.raises(
        ValueError,
        match="dimension",
    ):
        search_engine.search("bleeding query")


def test_vector_search_rejects_document_mismatch(
    sample_documents,
):
    embeddings = np.ones(
        (2, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="number of documents",
    ):
        VectorSearch(
            documents=sample_documents,
            embeddings=embeddings,
            embedder=FakeEmbedder(),
        )