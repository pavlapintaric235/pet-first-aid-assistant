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
                [5.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            "heat query": np.array(
                [0.0, 7.0, 0.0],
                dtype=np.float32,
            ),
            "transport query": np.array(
                [0.0, 0.0, 9.0],
                dtype=np.float32,
            ),
        }

        return vectors.get(
            text,
            np.array(
                [2.0, 2.0, 0.0],
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
            vector = np.zeros(
                3,
                dtype=np.float32,
            )

            vector[index % 3] = float(
                index + 2
            )

            vectors.append(vector)

        return np.array(
            vectors,
            dtype=np.float32,
        )


def make_document(
    document_id,
    species,
    title,
    topics,
    content,
    embedding_text=None,
    heading_path=None,
):
    document = {
        "id": document_id,
        "source_id": (
            f"source_{document_id}"
        ),
        "publisher": (
            "Test Veterinary Publisher"
        ),
        "title": title,
        "url": (
            "https://example.com/article"
        ),
        "species": species,
        "topics": topics,
        "content": content,
    }

    if embedding_text is not None:
        document[
            "embedding_text"
        ] = embedding_text

    if heading_path is not None:
        document[
            "heading_path"
        ] = heading_path

    return document


@pytest.fixture
def sample_documents():
    return [
        make_document(
            document_id="bleeding",
            species=[
                "dog",
                "cat",
            ],
            title="Severe Bleeding",
            topics=["bleeding"],
            content=(
                "Apply direct pressure "
                "to the wound."
            ),
            embedding_text=(
                "Title: Severe Bleeding\n"
                "Species: dog, cat\n"
                "Topics: bleeding\n"
                "Section: Bleeding\n"
                "Content: Apply direct "
                "pressure to the wound."
            ),
            heading_path=[
                "Bleeding",
            ],
        ),
        make_document(
            document_id="heatstroke",
            species=["dog"],
            title="Heatstroke",
            topics=["heatstroke"],
            content=(
                "Begin controlled cooling."
            ),
            embedding_text=(
                "Title: Heatstroke\n"
                "Species: dog\n"
                "Topics: heatstroke\n"
                "Section: Heat Stroke\n"
                "Content: Begin controlled "
                "cooling."
            ),
            heading_path=[
                "Heat Stroke",
            ],
        ),
        make_document(
            document_id="transport",
            species=["cat"],
            title="Transport",
            topics=[
                "injury",
                "transport",
            ],
            content=(
                "Use a secure carrier."
            ),
            embedding_text=(
                "Title: Transport\n"
                "Species: cat\n"
                "Topics: injury, transport\n"
                "Section: Transport\n"
                "Content: Use a secure "
                "carrier."
            ),
            heading_path=[
                "Transport",
            ],
        ),
    ]


@pytest.fixture
def sample_embeddings():
    return np.array(
        [
            [
                10.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                4.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                2.0,
            ],
        ],
        dtype=np.float32,
    )


def test_build_embedding_text_prefers_existing_embedding_text():
    document = make_document(
        document_id="test",
        species=["dog"],
        title="Emergency Bleeding",
        topics=[
            "bleeding",
            "wounds",
        ],
        content=(
            "Apply direct pressure."
        ),
        embedding_text=(
            "Already prepared embedding text."
        ),
        heading_path=[
            "Bleeding",
        ],
    )

    result = build_embedding_text(
        document
    )

    assert result == (
        "Already prepared embedding text."
    )


def test_build_embedding_text_uses_metadata_aware_fallback():
    document = make_document(
        document_id="test",
        species=["dog"],
        title="Emergency Bleeding",
        topics=[
            "bleeding",
            "wounds",
        ],
        content=(
            "Apply direct pressure."
        ),
        heading_path=[
            "First Aid",
            "Bleeding",
        ],
    )

    result = build_embedding_text(
        document
    )

    assert (
        "Title: Emergency Bleeding"
        in result
    )

    assert (
        "Species: dog"
        in result
    )

    assert (
        "Topics: bleeding, wounds"
        in result
    )

    assert (
        "Section: First Aid > Bleeding"
        in result
    )

    assert (
        "Content: Apply direct pressure."
        in result
    )


def test_build_embedding_matrix_normalizes_vectors(
    sample_documents,
):
    matrix = build_embedding_matrix(
        documents=sample_documents,
        embedder=FakeEmbedder(),
        batch_size=2,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
    )

    assert matrix.shape == (
        3,
        3,
    )

    assert (
        matrix.dtype
        == np.float32
    )

    assert np.allclose(
        norms,
        1.0,
    )


def test_save_and_load_embedding_matrix(
    tmp_path,
    sample_embeddings,
):
    output_path = (
        tmp_path / "embeddings.npy"
    )

    save_embedding_matrix(
        embeddings=sample_embeddings,
        output_path=output_path,
    )

    loaded = load_embedding_matrix(
        output_path
    )

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

    assert (
        results[0]["id"]
        == "bleeding"
    )

    assert (
        results[0]["retrieval_method"]
        == "vector"
    )

    assert (
        results[0]["retrieval_score"]
        == pytest.approx(1.0)
    )


def test_vector_search_normalizes_document_embeddings(
    sample_documents,
    sample_embeddings,
):
    search_engine = VectorSearch(
        documents=sample_documents,
        embeddings=sample_embeddings,
        embedder=FakeEmbedder(),
    )

    norms = np.linalg.norm(
        search_engine.embeddings,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
    )


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
        (
            3,
            4,
        ),
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
        search_engine.search(
            "bleeding query"
        )


def test_vector_search_rejects_document_mismatch(
    sample_documents,
):
    embeddings = np.ones(
        (
            2,
            3,
        ),
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


def test_vector_search_rejects_zero_norm_document_embedding(
    sample_documents,
):
    embeddings = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="non-zero L2 norm",
    ):
        VectorSearch(
            documents=sample_documents,
            embeddings=embeddings,
            embedder=FakeEmbedder(),
        )