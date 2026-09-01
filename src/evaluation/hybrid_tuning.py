from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from src.retrieval.source_diversity import (
    SourceDiversifiedSearch,
)
from src.retrieval.vector_search import (
    VectorSearch,
    load_embedding_matrix,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "hybrid_tuning_metrics.json"
)

NUMBER_OF_RESULTS = 5
MAX_CHUNKS_PER_SOURCE = 1
RRF_K_VALUES = [1, 10, 30, 60, 100]


def save_results(
    results: dict[str, Any],
) -> Path:
    """Save hybrid-tuning evaluation results."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return OUTPUT_PATH


def main() -> None:
    """Evaluate diversified hybrid search with several RRF values."""

    ground_truth = load_ground_truth()
    documents = load_processed_documents()
    embeddings = load_embedding_matrix()
    embedder = Embedder()

    base_keyword_engine = KeywordSearch(
        documents
    )

    base_vector_engine = VectorSearch(
        documents=documents,
        embeddings=embeddings,
        embedder=embedder,
    )

    diversified_keyword_engine = (
        SourceDiversifiedSearch(
            search_engine=base_keyword_engine,
            max_chunks_per_source=(
                MAX_CHUNKS_PER_SOURCE
            ),
            candidate_multiplier=10,
        )
    )

    diversified_vector_engine = (
        SourceDiversifiedSearch(
            search_engine=base_vector_engine,
            max_chunks_per_source=(
                MAX_CHUNKS_PER_SOURCE
            ),
            candidate_multiplier=10,
        )
    )

    methods: dict[str, Any] = {}

    for rrf_k in RRF_K_VALUES:
        hybrid_engine = HybridSearch(
            keyword_search=(
                diversified_keyword_engine
            ),
            vector_search=(
                diversified_vector_engine
            ),
            rrf_k=rrf_k,
            candidate_multiplier=3,
        )

        final_engine = SourceDiversifiedSearch(
            search_engine=hybrid_engine,
            max_chunks_per_source=(
                MAX_CHUNKS_PER_SOURCE
            ),
            candidate_multiplier=10,
        )

        method_name = (
            f"hybrid_pre_post_diverse_k_{rrf_k}"
        )

        methods[method_name] = evaluate_search(
            ground_truth=ground_truth,
            search_engine=final_engine,
            num_results=NUMBER_OF_RESULTS,
        )

    results = {
        "settings": {
            "num_questions": len(ground_truth),
            "num_results": NUMBER_OF_RESULTS,
            "max_chunks_per_source": (
                MAX_CHUNKS_PER_SOURCE
            ),
            "rrf_k_values": RRF_K_VALUES,
            "diversity_applied_before_fusion": True,
            "diversity_applied_after_fusion": True,
        },
        "methods": methods,
    }

    output_path = save_results(results)

    print()
    print("Diversified hybrid tuning")
    print()
    print(
        f"{'Method':<40}"
        f"{'Hit Rate':>12}"
        f"{'MRR':>12}"
    )
    print("-" * 64)

    for method_name, result in methods.items():
        print(
            f"{method_name:<40}"
            f"{result['hit_rate']:>12.4f}"
            f"{result['mrr']:>12.4f}"
        )

    best_method = max(
        methods,
        key=lambda name: (
            methods[name]["hit_rate"],
            methods[name]["mrr"],
        ),
    )

    best_result = methods[best_method]

    print()
    print(f"Best method: {best_method}")
    print(
        f"Best Hit Rate: "
        f"{best_result['hit_rate']:.4f}"
    )
    print(
        f"Best MRR: "
        f"{best_result['mrr']:.4f}"
    )
    print()
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()