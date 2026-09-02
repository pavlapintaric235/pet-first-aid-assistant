from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.evaluation.retrieval_evaluation import (
    evaluate_search,
    load_ground_truth,
)
from src.retrieval.embedder import Embedder
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.keyword_search import (
    KeywordSearch,
    load_processed_documents,
)
from src.retrieval.query_expansion import (
    QueryExpansionSearch,
    expand_query,
    expansion_terms,
    matched_expansions,
)
from src.retrieval.source_diversity import (
    SourceDiversifiedSearch,
)
from src.retrieval.vector_search import (
    VectorSearch,
    load_embedding_matrix,
)


DEFAULT_NUMBER_OF_RESULTS = 5
DEFAULT_RRF_K = 60
DEFAULT_MAX_CHUNKS_PER_SOURCE = 1
DEFAULT_CANDIDATE_MULTIPLIER = 10


def parse_arguments() -> argparse.Namespace:
    """Parse query-expansion evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline retrieval with deterministic "
            "query terminology expansion."
        )
    )

    parser.add_argument(
        "--num-results",
        type=int,
        default=DEFAULT_NUMBER_OF_RESULTS,
        help=(
            "Number of retrieved results per question. "
            f"Default: {DEFAULT_NUMBER_OF_RESULTS}"
        ),
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=(
            "RRF constant used by hybrid retrieval. "
            f"Default: {DEFAULT_RRF_K}"
        ),
    )

    return parser.parse_args()


def evaluate_engine(
    ground_truth: list[dict[str, Any]],
    search_engine: Any,
    num_results: int,
) -> dict[str, Any]:
    """Evaluate one retrieval configuration."""

    return evaluate_search(
        ground_truth=ground_truth,
        search_engine=search_engine,
        num_results=num_results,
    )


def compare_question_ranks(
    baseline: dict[str, Any],
    expanded: dict[str, Any],
) -> dict[str, int]:
    """Count per-question rank improvements and regressions."""

    baseline_by_id = {
        question["id"]: question
        for question in baseline["questions"]
    }

    improved = 0
    unchanged = 0
    worsened = 0

    for question in expanded["questions"]:
        question_id = question["id"]

        baseline_rank = baseline_by_id[
            question_id
        ]["first_relevant_rank"]

        expanded_rank = question[
            "first_relevant_rank"
        ]

        baseline_value = (
            baseline_rank
            if baseline_rank is not None
            else float("inf")
        )

        expanded_value = (
            expanded_rank
            if expanded_rank is not None
            else float("inf")
        )

        if expanded_value < baseline_value:
            improved += 1
        elif expanded_value > baseline_value:
            worsened += 1
        else:
            unchanged += 1

    return {
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
    }


def print_expansions(
    ground_truth: list[dict[str, Any]],
) -> None:
    """Print every evaluation query changed by the rules."""

    print()
    print("Matched query expansions")
    print("------------------------")

    matched_count = 0

    for record in ground_truth:
        terms = expansion_terms(
            record["question"]
        )

        if not terms:
            continue

        matched_count += 1

        rules = [
            match["rule"]
            for match in matched_expansions(
                record["question"]
            )
        ]

        print()
        print(
            f"{record['id']} | "
            f"{record['category']}"
        )
        print(
            f"Original: {record['question']}"
        )
        print(
            f"Rules: {', '.join(rules)}"
        )
        print(
            f"Added: {', '.join(terms)}"
        )
        print(
            "Expanded: "
            f"{expand_query(record['question'])}"
        )

    print()
    print(
        "Expanded evaluation questions: "
        f"{matched_count}/{len(ground_truth)}"
    )


def main() -> None:
    """Run a controlled baseline-versus-expansion experiment."""

    arguments = parse_arguments()

    documents = load_processed_documents()
    embeddings = load_embedding_matrix()
    ground_truth = load_ground_truth()

    embedder = Embedder()

    keyword_engine = KeywordSearch(
        documents
    )

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

    baseline_engines = {
        "keyword": keyword_engine,
        "vector": vector_engine,
        "hybrid": hybrid_engine,
        "keyword_diverse_1": SourceDiversifiedSearch(
            search_engine=keyword_engine,
            max_chunks_per_source=(
                DEFAULT_MAX_CHUNKS_PER_SOURCE
            ),
            candidate_multiplier=(
                DEFAULT_CANDIDATE_MULTIPLIER
            ),
        ),
        "vector_diverse_1": SourceDiversifiedSearch(
            search_engine=vector_engine,
            max_chunks_per_source=(
                DEFAULT_MAX_CHUNKS_PER_SOURCE
            ),
            candidate_multiplier=(
                DEFAULT_CANDIDATE_MULTIPLIER
            ),
        ),
        "hybrid_diverse_1": SourceDiversifiedSearch(
            search_engine=hybrid_engine,
            max_chunks_per_source=(
                DEFAULT_MAX_CHUNKS_PER_SOURCE
            ),
            candidate_multiplier=(
                DEFAULT_CANDIDATE_MULTIPLIER
            ),
        ),
    }

    expanded_engines = {
        name: QueryExpansionSearch(
            search_engine=engine
        )
        for name, engine in baseline_engines.items()
    }

    print()
    print("Deterministic query-expansion evaluation")
    print("----------------------------------------")
    print(
        f"Documents: {len(documents)}"
    )
    print(
        f"Questions: {len(ground_truth)}"
    )
    print(
        "Embedding model: existing saved MiniLM index"
    )
    print(
        "Expansion type: deterministic terminology only"
    )

    print_expansions(
        ground_truth
    )

    rows: list[dict[str, Any]] = []

    for name, baseline_engine in baseline_engines.items():
        baseline_metrics = evaluate_engine(
            ground_truth=ground_truth,
            search_engine=baseline_engine,
            num_results=arguments.num_results,
        )

        expanded_metrics = evaluate_engine(
            ground_truth=ground_truth,
            search_engine=expanded_engines[name],
            num_results=arguments.num_results,
        )

        rank_changes = compare_question_ranks(
            baseline=baseline_metrics,
            expanded=expanded_metrics,
        )

        rows.append(
            {
                "method": name,
                "baseline_hit_rate": (
                    baseline_metrics["hit_rate"]
                ),
                "expanded_hit_rate": (
                    expanded_metrics["hit_rate"]
                ),
                "baseline_mrr": (
                    baseline_metrics["mrr"]
                ),
                "expanded_mrr": (
                    expanded_metrics["mrr"]
                ),
                "mrr_delta": (
                    expanded_metrics["mrr"]
                    - baseline_metrics["mrr"]
                ),
                **rank_changes,
            }
        )

    print()
    print("Retrieval comparison")
    print("--------------------")
    print(
        f"{'Method':<22}"
        f"{'Base HR':>10}"
        f"{'Exp HR':>10}"
        f"{'Base MRR':>11}"
        f"{'Exp MRR':>10}"
        f"{'Delta':>10}"
    )
    print("-" * 73)

    for row in rows:
        print(
            f"{row['method']:<22}"
            f"{row['baseline_hit_rate']:>10.4f}"
            f"{row['expanded_hit_rate']:>10.4f}"
            f"{row['baseline_mrr']:>11.4f}"
            f"{row['expanded_mrr']:>10.4f}"
            f"{row['mrr_delta']:>+10.4f}"
        )

    print()
    print("Per-question first-relevant-rank changes")
    print("----------------------------------------")
    print(
        f"{'Method':<22}"
        f"{'Improved':>10}"
        f"{'Same':>10}"
        f"{'Worse':>10}"
    )
    print("-" * 52)

    for row in rows:
        print(
            f"{row['method']:<22}"
            f"{row['improved']:>10}"
            f"{row['unchanged']:>10}"
            f"{row['worsened']:>10}"
        )

    print()
    print(
        "No production search implementation was modified."
    )
    print(
        "No embedding files or evaluation ground truth "
        "were modified."
    )


if __name__ == "__main__":
    main()