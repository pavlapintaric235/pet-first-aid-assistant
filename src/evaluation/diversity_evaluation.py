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
    / "diversity_metrics.json"
)

NUMBER_OF_RESULTS = 5


def save_results(
    results: dict[str, Any],
) -> Path:
    """Save source-diversity evaluation results."""

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
    """Evaluate source-diversity configurations."""

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
    )

    base_engines = {
        "keyword": keyword_engine,
        "vector": vector_engine,
        "hybrid": hybrid_engine,
    }

    methods: dict[str, Any] = {}

    for max_chunks in [1, 2]:
        for method_name, base_engine in base_engines.items():
            evaluation_name = (
                f"{method_name}_diverse_{max_chunks}"
            )

            diversified_engine = SourceDiversifiedSearch(
                search_engine=base_engine,
                max_chunks_per_source=max_chunks,
                candidate_multiplier=10,
            )

            methods[evaluation_name] = evaluate_search(
                ground_truth=ground_truth,
                search_engine=diversified_engine,
                num_results=NUMBER_OF_RESULTS,
            )

    results = {
        "settings": {
            "num_questions": len(ground_truth),
            "num_results": NUMBER_OF_RESULTS,
            "candidate_multiplier": 10,
            "tested_max_chunks_per_source": [1, 2],
        },
        "methods": methods,
    }

    output_path = save_results(results)

    print()
    print("Source-diversity evaluation")
    print()
    print(
        f"{'Method':<25}"
        f"{'Hit Rate':>12}"
        f"{'MRR':>12}"
    )
    print("-" * 49)

    for method_name, result in methods.items():
        print(
            f"{method_name:<25}"
            f"{result['hit_rate']:>12.4f}"
            f"{result['mrr']:>12.4f}"
        )

    print()
    print(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()