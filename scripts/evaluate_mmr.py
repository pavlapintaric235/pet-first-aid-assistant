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
from src.retrieval.keyword_search import (
    load_processed_documents,
)
from src.retrieval.mmr_search import (
    MMRSearch,
)
from src.retrieval.source_diversity import (
    SourceDiversifiedSearch,
)
from src.retrieval.vector_search import (
    VectorSearch,
    load_embedding_matrix,
)


DEFAULT_NUMBER_OF_RESULTS = 5

DEFAULT_CANDIDATE_MULTIPLIER = 4

DEFAULT_SOURCE_CANDIDATE_MULTIPLIER = 10

DEFAULT_LAMBDAS = (
    1.0,
    0.9,
    0.75,
    0.5,
)


def parse_arguments() -> argparse.Namespace:
    """Parse MMR evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Maximum Marginal Relevance "
            "over the saved MiniLM vector index."
        )
    )

    parser.add_argument(
        "--num-results",
        type=int,
        default=DEFAULT_NUMBER_OF_RESULTS,
        help=(
            "Number of final results per question. "
            f"Default: "
            f"{DEFAULT_NUMBER_OF_RESULTS}"
        ),
    )

    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=(
            DEFAULT_CANDIDATE_MULTIPLIER
        ),
        help=(
            "How many vector candidates "
            "MMR considers. "
            f"Default: "
            f"{DEFAULT_CANDIDATE_MULTIPLIER}"
        ),
    )

    return parser.parse_args()


def compare_question_ranks(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, int]:
    """Compare first-relevant ranks question by question."""

    baseline_by_id = {
        question["id"]: question
        for question in baseline[
            "questions"
        ]
    }

    improved = 0
    unchanged = 0
    worsened = 0

    for question in candidate[
        "questions"
    ]:
        baseline_rank = (
            baseline_by_id[
                question["id"]
            ][
                "first_relevant_rank"
            ]
        )

        candidate_rank = question[
            "first_relevant_rank"
        ]

        baseline_value = (
            baseline_rank
            if baseline_rank is not None
            else float("inf")
        )

        candidate_value = (
            candidate_rank
            if candidate_rank is not None
            else float("inf")
        )

        if (
            candidate_value
            < baseline_value
        ):
            improved += 1

        elif (
            candidate_value
            > baseline_value
        ):
            worsened += 1

        else:
            unchanged += 1

    return {
        "improved": improved,
        "unchanged": unchanged,
        "worsened": worsened,
    }


def main() -> None:
    """Run controlled MMR experiments."""

    arguments = parse_arguments()

    documents = (
        load_processed_documents()
    )

    embeddings = (
        load_embedding_matrix()
    )

    ground_truth = (
        load_ground_truth()
    )

    embedder = Embedder()

    vector_engine = VectorSearch(
        documents=documents,
        embeddings=embeddings,
        embedder=embedder,
    )

    vector_diverse_engine = (
        SourceDiversifiedSearch(
            search_engine=vector_engine,
            max_chunks_per_source=1,
            candidate_multiplier=(
                DEFAULT_SOURCE_CANDIDATE_MULTIPLIER
            ),
        )
    )

    baseline_vector = evaluate_search(
        ground_truth=ground_truth,
        search_engine=vector_engine,
        num_results=(
            arguments.num_results
        ),
    )

    baseline_diverse = evaluate_search(
        ground_truth=ground_truth,
        search_engine=(
            vector_diverse_engine
        ),
        num_results=(
            arguments.num_results
        ),
    )

    print()
    print("MMR evaluation")
    print("--------------")
    print(
        f"Documents: {len(documents)}"
    )
    print(
        f"Questions: {len(ground_truth)}"
    )
    print(
        "Embedding model: "
        "saved MiniLM index"
    )
    print(
        "MMR candidate multiplier: "
        f"{arguments.candidate_multiplier}"
    )
    print()

    print(
        f"{'Method':<24}"
        f"{'Hit Rate':>12}"
        f"{'MRR':>12}"
        f"{'Improved':>11}"
        f"{'Same':>8}"
        f"{'Worse':>8}"
    )

    print("-" * 75)

    print(
        f"{'vector_baseline':<24}"
        f"{baseline_vector['hit_rate']:>12.4f}"
        f"{baseline_vector['mrr']:>12.4f}"
        f"{'-':>11}"
        f"{'-':>8}"
        f"{'-':>8}"
    )

    print(
        f"{'vector_diverse_1':<24}"
        f"{baseline_diverse['hit_rate']:>12.4f}"
        f"{baseline_diverse['mrr']:>12.4f}"
        f"{'-':>11}"
        f"{'-':>8}"
        f"{'-':>8}"
    )

    for lambda_mult in DEFAULT_LAMBDAS:
        mmr_engine = MMRSearch(
            vector_search=vector_engine,
            lambda_mult=lambda_mult,
            candidate_multiplier=(
                arguments.candidate_multiplier
            ),
        )

        mmr_metrics = evaluate_search(
            ground_truth=ground_truth,
            search_engine=mmr_engine,
            num_results=(
                arguments.num_results
            ),
        )

        mmr_changes = (
            compare_question_ranks(
                baseline=baseline_vector,
                candidate=mmr_metrics,
            )
        )

        name = (
            f"mmr_lambda_"
            f"{lambda_mult:g}"
        )

        print(
            f"{name:<24}"
            f"{mmr_metrics['hit_rate']:>12.4f}"
            f"{mmr_metrics['mrr']:>12.4f}"
            f"{mmr_changes['improved']:>11}"
            f"{mmr_changes['unchanged']:>8}"
            f"{mmr_changes['worsened']:>8}"
        )

        mmr_diverse_engine = (
            SourceDiversifiedSearch(
                search_engine=mmr_engine,
                max_chunks_per_source=1,
                candidate_multiplier=(
                    DEFAULT_SOURCE_CANDIDATE_MULTIPLIER
                ),
            )
        )

        mmr_diverse_metrics = (
            evaluate_search(
                ground_truth=ground_truth,
                search_engine=(
                    mmr_diverse_engine
                ),
                num_results=(
                    arguments.num_results
                ),
            )
        )

        diverse_changes = (
            compare_question_ranks(
                baseline=baseline_diverse,
                candidate=(
                    mmr_diverse_metrics
                ),
            )
        )

        diverse_name = (
            f"mmr_diverse_"
            f"{lambda_mult:g}"
        )

        print(
            f"{diverse_name:<24}"
            f"{mmr_diverse_metrics['hit_rate']:>12.4f}"
            f"{mmr_diverse_metrics['mrr']:>12.4f}"
            f"{diverse_changes['improved']:>11}"
            f"{diverse_changes['unchanged']:>8}"
            f"{diverse_changes['worsened']:>8}"
        )

    print()

    print(
        "lambda=1.0 should reproduce the "
        "ordinary vector ranking before "
        "source-diversity filtering."
    )

    print(
        "No saved embeddings, ground truth, "
        "or production search configuration "
        "were modified."
    )


if __name__ == "__main__":
    main()