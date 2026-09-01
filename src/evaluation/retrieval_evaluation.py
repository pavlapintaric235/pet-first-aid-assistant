from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.keyword_search import (
    KeywordSearch,
    load_processed_documents,
)
from src.retrieval.vector_search import (
    VectorSearch,
    load_embedding_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_ground_truth.json"
)

METRICS_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "retrieval_metrics.json"
)

DEFAULT_NUMBER_OF_RESULTS = 5
DEFAULT_RRF_K = 60


class SearchEngine(Protocol):
    """Interface required by retrieval evaluation."""

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked retrieval results."""


def load_ground_truth(
    ground_truth_path: Path = GROUND_TRUTH_PATH,
) -> list[dict[str, Any]]:
    """Load and validate retrieval ground truth."""

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth data was not found at "
            f"{ground_truth_path}"
        )

    with ground_truth_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    if not isinstance(records, list):
        raise ValueError(
            "Ground truth must contain a JSON list."
        )

    if not records:
        raise ValueError(
            "Ground truth cannot be empty."
        )

    required_fields = {
        "id",
        "question",
        "species",
        "category",
        "relevant_source_ids",
    }

    record_ids: list[str] = []

    for position, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(
                f"Ground-truth record {position} "
                f"must be a JSON object."
            )

        missing_fields = required_fields - record.keys()

        if missing_fields:
            missing = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                f"Ground-truth record {position} "
                f"is missing fields: {missing}"
            )

        if not record["question"].strip():
            raise ValueError(
                f"Ground-truth record {position} "
                f"has a blank question."
            )

        if not record["relevant_source_ids"]:
            raise ValueError(
                f"Ground-truth record {position} "
                f"has no relevant source IDs."
            )

        if record["species"] not in {
            None,
            "dog",
            "cat",
        }:
            raise ValueError(
                f"Ground-truth record {position} "
                f"has an unsupported species."
            )

        record_ids.append(record["id"])

    if len(record_ids) != len(set(record_ids)):
        raise ValueError(
            "Ground-truth record IDs must be unique."
        )

    return records


def compute_relevance(
    record: dict[str, Any],
    results: list[dict[str, Any]],
) -> list[int]:
    """Create a binary relevance list for ranked results."""

    relevant_source_ids = set(
        record["relevant_source_ids"]
    )

    return [
        int(
            result["source_id"]
            in relevant_source_ids
        )
        for result in results
    ]


def hit_rate(
    relevance_total: list[list[int]],
) -> float:
    """Calculate the fraction of queries with a hit."""

    if not relevance_total:
        raise ValueError(
            "Relevance data cannot be empty."
        )

    hits = sum(
        int(any(relevance))
        for relevance in relevance_total
    )

    return hits / len(relevance_total)


def mean_reciprocal_rank(
    relevance_total: list[list[int]],
) -> float:
    """Calculate Mean Reciprocal Rank."""

    if not relevance_total:
        raise ValueError(
            "Relevance data cannot be empty."
        )

    reciprocal_ranks: list[float] = []

    for relevance in relevance_total:
        reciprocal_rank = 0.0

        for rank, is_relevant in enumerate(
            relevance,
            start=1,
        ):
            if is_relevant:
                reciprocal_rank = 1.0 / rank
                break

        reciprocal_ranks.append(
            reciprocal_rank
        )

    return sum(reciprocal_ranks) / len(
        reciprocal_ranks
    )


def evaluate_search(
    ground_truth: list[dict[str, Any]],
    search_engine: SearchEngine,
    num_results: int = DEFAULT_NUMBER_OF_RESULTS,
) -> dict[str, Any]:
    """Evaluate one search engine on all questions."""

    if num_results <= 0:
        raise ValueError(
            "num_results must be greater than zero."
        )

    relevance_total: list[list[int]] = []
    question_results: list[dict[str, Any]] = []

    for record in ground_truth:
        results = search_engine.search(
            query=record["question"],
            num_results=num_results,
            species=record["species"],
        )

        relevance = compute_relevance(
            record=record,
            results=results,
        )

        relevance_total.append(relevance)

        first_relevant_rank = None

        for rank, is_relevant in enumerate(
            relevance,
            start=1,
        ):
            if is_relevant:
                first_relevant_rank = rank
                break

        question_results.append(
            {
                "id": record["id"],
                "question": record["question"],
                "species": record["species"],
                "category": record["category"],
                "relevant_source_ids": (
                    record["relevant_source_ids"]
                ),
                "retrieved_source_ids": [
                    result["source_id"]
                    for result in results
                ],
                "retrieved_document_ids": [
                    result["id"]
                    for result in results
                ],
                "relevance": relevance,
                "first_relevant_rank": (
                    first_relevant_rank
                ),
            }
        )

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mean_reciprocal_rank(
            relevance_total
        ),
        "num_questions": len(ground_truth),
        "num_results": num_results,
        "questions": question_results,
    }


def save_metrics(
    metrics: dict[str, Any],
    output_path: Path = METRICS_OUTPUT_PATH,
) -> Path:
    """Save evaluation metrics and per-question results."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse evaluation command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate keyword, vector, and hybrid retrieval."
        )
    )

    parser.add_argument(
        "--num-results",
        type=int,
        default=DEFAULT_NUMBER_OF_RESULTS,
        help=(
            "Number of retrieved chunks evaluated per query. "
            f"Default: {DEFAULT_NUMBER_OF_RESULTS}"
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "RRF constant used by hybrid search. "
            f"Default: {DEFAULT_RRF_K}"
        ),
    )

    return parser.parse_args()


def print_summary(
    metrics: dict[str, Any],
) -> None:
    """Print a compact comparison table."""

    print()
    print("Retrieval evaluation")
    print(
        f"Questions: {metrics['settings']['num_questions']}"
    )
    print(
        f"Results per question: "
        f"{metrics['settings']['num_results']}"
    )
    print()

    print(
        f"{'Method':<12}"
        f"{'Hit Rate':>12}"
        f"{'MRR':>12}"
    )

    print("-" * 36)

    for method in [
        "keyword",
        "vector",
        "hybrid",
    ]:
        result = metrics["methods"][method]

        print(
            f"{method:<12}"
            f"{result['hit_rate']:>12.4f}"
            f"{result['mrr']:>12.4f}"
        )


def main() -> None:
    """Evaluate all retrieval implementations."""

    arguments = parse_arguments()

    ground_truth = load_ground_truth()
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

    keyword_metrics = evaluate_search(
        ground_truth=ground_truth,
        search_engine=keyword_engine,
        num_results=arguments.num_results,
    )

    vector_metrics = evaluate_search(
        ground_truth=ground_truth,
        search_engine=vector_engine,
        num_results=arguments.num_results,
    )

    hybrid_metrics = evaluate_search(
        ground_truth=ground_truth,
        search_engine=hybrid_engine,
        num_results=arguments.num_results,
    )

    metrics = {
        "settings": {
            "num_questions": len(ground_truth),
            "num_results": arguments.num_results,
            "rrf_k": arguments.rrf_k,
            "relevance_level": "source_id",
        },
        "methods": {
            "keyword": keyword_metrics,
            "vector": vector_metrics,
            "hybrid": hybrid_metrics,
        },
    }

    output_path = save_metrics(metrics)

    print_summary(metrics)
    print()
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()