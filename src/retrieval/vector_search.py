from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.retrieval.embedder import Embedder
from src.retrieval.keyword_search import (
    load_processed_documents,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "embeddings.npy"
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUMBER_OF_RESULTS = 5


class EmbeddingModel(Protocol):
    """Interface required by vector-index functions."""

    def encode(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed one text."""

    def encode_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed multiple texts."""


def build_embedding_text(
    document: dict[str, Any],
) -> str:
    """Combine document fields into text for embedding."""

    title = document["title"]
    topics = ", ".join(document["topics"])
    content = document["content"]

    return (
        f"Title: {title}\n"
        f"Topics: {topics}\n"
        f"Content: {content}"
    )


def build_embedding_matrix(
    documents: list[dict[str, Any]],
    embedder: EmbeddingModel,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """Embed every processed veterinary document."""

    if not documents:
        raise ValueError(
            "At least one document is required."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    texts = [
        build_embedding_text(document)
        for document in documents
    ]

    batches: list[np.ndarray] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        batch_vectors = np.asarray(
            embedder.encode_batch(batch),
            dtype=np.float32,
        )

        if batch_vectors.ndim != 2:
            raise ValueError(
                "The embedder must return a two-dimensional "
                "batch matrix."
            )

        if len(batch_vectors) != len(batch):
            raise ValueError(
                "The number of embeddings does not match "
                "the number of texts."
            )

        batches.append(batch_vectors)

    matrix = np.vstack(batches)

    if len(matrix) != len(documents):
        raise ValueError(
            "The embedding matrix does not match the "
            "document collection."
        )

    return matrix


def save_embedding_matrix(
    embeddings: np.ndarray,
    output_path: Path = EMBEDDINGS_PATH,
) -> Path:
    """Save the embedding matrix as a NumPy file."""

    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "Embeddings must be a two-dimensional matrix."
        )

    if matrix.size == 0:
        raise ValueError(
            "The embedding matrix cannot be empty."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output_path,
        matrix,
        allow_pickle=False,
    )

    return output_path


def load_embedding_matrix(
    embeddings_path: Path = EMBEDDINGS_PATH,
) -> np.ndarray:
    """Load a persisted embedding matrix."""

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding matrix was not found at "
            f"{embeddings_path}. Run the vector-index "
            f"build script first."
        )

    matrix = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    if matrix.ndim != 2:
        raise ValueError(
            "The saved embeddings must form a "
            "two-dimensional matrix."
        )

    return np.asarray(
        matrix,
        dtype=np.float32,
    )


class VectorSearch:
    """Exact cosine vector search over normalized embeddings."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
        embeddings: np.ndarray,
        embedder: EmbeddingModel,
    ) -> None:
        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if not documents:
            raise ValueError(
                "At least one document is required."
            )

        if matrix.ndim != 2:
            raise ValueError(
                "Embeddings must be a two-dimensional matrix."
            )

        if len(documents) != len(matrix):
            raise ValueError(
                "The number of documents must match the "
                "number of embedding rows."
            )

        self.documents = documents
        self.embeddings = matrix
        self.embedder = embedder

    def search(
        self,
        query: str,
        num_results: int = DEFAULT_NUMBER_OF_RESULTS,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return documents with the highest cosine similarity."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                "The search query must contain searchable text."
            )

        if num_results <= 0:
            raise ValueError(
                "num_results must be greater than zero."
            )

        normalized_species = (
            species.lower().strip()
            if species is not None
            else None
        )

        if normalized_species not in {None, "dog", "cat"}:
            raise ValueError(
                "species must be 'dog', 'cat', or None."
            )

        query_vector = np.asarray(
            self.embedder.encode(query),
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "The query embedding must be one-dimensional."
            )

        if query_vector.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                "The query embedding dimension does not "
                "match the document embedding dimension."
            )

        scores = self.embeddings.dot(query_vector)
        ranked_indices = np.argsort(scores)[::-1]

        results: list[dict[str, Any]] = []

        for index in ranked_indices:
            document = self.documents[int(index)]

            if (
                normalized_species is not None
                and normalized_species
                not in document["species"]
            ):
                continue

            result = dict(document)
            result["retrieval_score"] = float(scores[index])
            result["retrieval_method"] = "vector"

            results.append(result)

            if len(results) == num_results:
                break

        return results


def parse_arguments() -> argparse.Namespace:
    """Parse vector-search command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Search veterinary documents using ONNX embeddings."
        )
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Pet first-aid question or symptom description.",
    )

    parser.add_argument(
        "--species",
        choices=["dog", "cat"],
        default=None,
        help="Optionally limit results to one species.",
    )

    parser.add_argument(
        "--num-results",
        type=int,
        default=DEFAULT_NUMBER_OF_RESULTS,
        help=(
            "Number of results to return. "
            f"Default: {DEFAULT_NUMBER_OF_RESULTS}"
        ),
    )

    return parser.parse_args()


def print_results(
    results: list[dict[str, Any]],
) -> None:
    """Print vector-search results."""

    if not results:
        print("No matching documents were found.")
        return

    for position, result in enumerate(results, start=1):
        preview = result["content"][:400].strip()

        print()
        print(f"Result {position}")
        print(f"Score: {result['retrieval_score']:.4f}")
        print(f"Source: {result['publisher']}")
        print(f"Title: {result['title']}")
        print(f"Species: {', '.join(result['species'])}")
        print(f"Topics: {', '.join(result['topics'])}")
        print(f"URL: {result['url']}")
        print(f"Content: {preview}...")


def main() -> None:
    """Load the saved vector index and run a search."""

    arguments = parse_arguments()

    documents = load_processed_documents()
    embeddings = load_embedding_matrix()
    embedder = Embedder()

    search_engine = VectorSearch(
        documents=documents,
        embeddings=embeddings,
        embedder=embedder,
    )

    results = search_engine.search(
        query=arguments.query,
        num_results=arguments.num_results,
        species=arguments.species,
    )

    print_results(results)


if __name__ == "__main__":
    main()