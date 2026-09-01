from __future__ import annotations

import argparse
from typing import Any, Protocol

from src.retrieval.embedder import Embedder
from src.retrieval.keyword_search import (
    KeywordSearch,
    load_processed_documents,
)
from src.retrieval.vector_search import (
    VectorSearch,
    load_embedding_matrix,
)


DEFAULT_RRF_K = 60
DEFAULT_NUMBER_OF_RESULTS = 5
DEFAULT_CANDIDATE_MULTIPLIER = 3


class SearchEngine(Protocol):
    """Interface shared by all retrieval implementations."""

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant veterinary documents."""


def reciprocal_rank_fusion(
    result_lists: list[list[dict[str, Any]]],
    k: int = DEFAULT_RRF_K,
    num_results: int = DEFAULT_NUMBER_OF_RESULTS,
) -> list[dict[str, Any]]:
    """Merge ranked result lists using reciprocal rank fusion."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if num_results <= 0:
        raise ValueError(
            "num_results must be greater than zero."
        )

    fused_scores: dict[str, float] = {}
    documents_by_id: dict[str, dict[str, Any]] = {}
    component_results: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for results in result_lists:
        for rank, document in enumerate(
            results,
            start=1,
        ):
            document_id = document["id"]

            fused_scores[document_id] = (
                fused_scores.get(document_id, 0.0)
                + 1.0 / (k + rank)
            )

            if document_id not in documents_by_id:
                documents_by_id[document_id] = dict(document)

            component_results.setdefault(
                document_id,
                [],
            ).append(
                {
                    "method": document.get(
                        "retrieval_method",
                        "unknown",
                    ),
                    "rank": rank,
                    "score": float(
                        document.get(
                            "retrieval_score",
                            0.0,
                        )
                    ),
                }
            )

    ranked_document_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )

    fused_results: list[dict[str, Any]] = []

    for document_id in ranked_document_ids[:num_results]:
        result = dict(documents_by_id[document_id])

        result["retrieval_score"] = fused_scores[
            document_id
        ]
        result["retrieval_method"] = "hybrid"
        result["component_results"] = (
            component_results[document_id]
        )

        fused_results.append(result)

    return fused_results


class HybridSearch:
    """Combine keyword and vector search with RRF."""

    def __init__(
        self,
        keyword_search: SearchEngine,
        vector_search: SearchEngine,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_multiplier: int = (
            DEFAULT_CANDIDATE_MULTIPLIER
        ),
    ) -> None:
        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than zero."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater "
                "than zero."
            )

        self.keyword_search = keyword_search
        self.vector_search = vector_search
        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier

    def search(
        self,
        query: str,
        num_results: int = DEFAULT_NUMBER_OF_RESULTS,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Run both search methods and fuse their rankings."""

        if num_results <= 0:
            raise ValueError(
                "num_results must be greater than zero."
            )

        num_candidates = max(
            num_results * self.candidate_multiplier,
            num_results,
        )

        keyword_results = self.keyword_search.search(
            query=query,
            num_results=num_candidates,
            species=species,
        )

        vector_results = self.vector_search.search(
            query=query,
            num_results=num_candidates,
            species=species,
        )

        return reciprocal_rank_fusion(
            result_lists=[
                keyword_results,
                vector_results,
            ],
            k=self.rrf_k,
            num_results=num_results,
        )


def parse_arguments() -> argparse.Namespace:
    """Parse hybrid-search command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Search veterinary documents using hybrid "
            "keyword and vector retrieval."
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

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "RRF rank constant. "
            f"Default: {DEFAULT_RRF_K}"
        ),
    )

    return parser.parse_args()


def print_results(
    results: list[dict[str, Any]],
) -> None:
    """Print hybrid-search results."""

    if not results:
        print("No matching documents were found.")
        return

    for position, result in enumerate(results, start=1):
        preview = result["content"][:400].strip()

        print()
        print(f"Result {position}")
        print(
            f"Hybrid score: "
            f"{result['retrieval_score']:.6f}"
        )
        print(f"Source: {result['publisher']}")
        print(f"Title: {result['title']}")
        print(f"Species: {', '.join(result['species'])}")
        print(f"Topics: {', '.join(result['topics'])}")
        print(f"URL: {result['url']}")

        print("Component rankings:")

        for component in result["component_results"]:
            print(
                f"  - {component['method']}: "
                f"rank {component['rank']}, "
                f"score {component['score']:.4f}"
            )

        print(f"Content: {preview}...")


def main() -> None:
    """Build the search engines and run hybrid search."""

    arguments = parse_arguments()

    documents = load_processed_documents()
    embeddings = load_embedding_matrix()
    embedder = Embedder()

    keyword_engine = KeywordSearch(documents)

    vector_engine = VectorSearch(
        documents=documents,
        embeddings=embeddings,
        embedder=embedder,
    )

    hybrid_engine = HybridSearch(
        keyword_search=keyword_engine,
        vector_search=vector_engine,
        rrf_k=arguments.rrf_k,
    )

    results = hybrid_engine.search(
        query=arguments.query,
        num_results=arguments.num_results,
        species=arguments.species,
    )

    print_results(results)


if __name__ == "__main__":
    main()