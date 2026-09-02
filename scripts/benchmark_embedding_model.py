from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


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
from src.retrieval.source_diversity import (
    SourceDiversifiedSearch,
)
from src.retrieval.vector_search import (
    VectorSearch,
    build_embedding_matrix,
)


DEFAULT_MODEL_PATH = (
    "models/Xenova/all-MiniLM-L6-v2"
)

DEFAULT_LABEL = "all-MiniLM-L6-v2"

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUMBER_OF_RESULTS = 5
DEFAULT_RRF_K = 60
DEFAULT_POOLING = "mean"


def parse_arguments() -> argparse.Namespace:
    """Parse benchmark command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one ONNX embedding model "
            "against the Pet First Aid retrieval set."
        )
    )

    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Local directory containing "
            "tokenizer.json and model.onnx."
        ),
    )

    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help=(
            "Readable model name shown in "
            "benchmark output."
        ),
    )

    parser.add_argument(
        "--pooling",
        choices=[
            "mean",
            "cls",
        ],
        default=DEFAULT_POOLING,
        help=(
            "Token pooling strategy used by "
            "the embedding model."
        ),
    )

    parser.add_argument(
        "--query-prefix",
        default="",
        help=(
            "Optional prefix applied only to "
            "retrieval queries, never documents."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Number of documents embedded per batch. "
            f"Default: {DEFAULT_BATCH_SIZE}"
        ),
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
            "RRF constant used for hybrid retrieval. "
            f"Default: {DEFAULT_RRF_K}"
        ),
    )

    return parser.parse_args()


def directory_size_mb(
    path: Path,
) -> float:
    """Return total size of files in a directory."""

    total_bytes = sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )

    return (
        total_bytes
        / (1024 * 1024)
    )


def evaluate_engine(
    name: str,
    ground_truth: list[dict[str, Any]],
    search_engine: Any,
    num_results: int,
) -> dict[str, Any]:
    """Evaluate one engine and record execution time."""

    started = time.perf_counter()

    metrics = evaluate_search(
        ground_truth=ground_truth,
        search_engine=search_engine,
        num_results=num_results,
    )

    elapsed = (
        time.perf_counter()
        - started
    )

    return {
        "name": name,
        "hit_rate": metrics[
            "hit_rate"
        ],
        "mrr": metrics["mrr"],
        "seconds": elapsed,
    }


def main() -> None:
    """Benchmark without replacing the saved index."""

    arguments = parse_arguments()

    model_path = Path(
        arguments.model_path
    )

    if not model_path.is_absolute():
        model_path = (
            PROJECT_ROOT
            / model_path
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory was not found: "
            f"{model_path}"
        )

    tokenizer_path = (
        model_path
        / "tokenizer.json"
    )

    onnx_path = (
        model_path
        / "model.onnx"
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer was not found: "
            f"{tokenizer_path}"
        )

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX model was not found: "
            f"{onnx_path}"
        )

    documents = (
        load_processed_documents()
    )

    ground_truth = (
        load_ground_truth()
    )

    print()
    print(
        "Embedding model benchmark"
    )
    print(
        "-------------------------"
    )
    print(
        f"Model: {arguments.label}"
    )
    print(
        f"Path: {model_path}"
    )
    print(
        f"Pooling: {arguments.pooling}"
    )
    print(
        "Query prefix: "
        + (
            repr(arguments.query_prefix)
            if arguments.query_prefix
            else "none"
        )
    )
    print(
        f"Documents: {len(documents)}"
    )
    print(
        f"Questions: {len(ground_truth)}"
    )
    print(
        "Model directory size: "
        f"{directory_size_mb(model_path):.2f} MB"
    )
    print()

    print(
        "Loading ONNX model..."
    )

    load_started = (
        time.perf_counter()
    )

    embedder = Embedder(
        path=model_path,
        pooling=arguments.pooling,
        query_prefix=(
            arguments.query_prefix
        ),
    )

    model_load_seconds = (
        time.perf_counter()
        - load_started
    )

    print(
        "Model load time: "
        f"{model_load_seconds:.3f} seconds"
    )

    print()
    print(
        "Building temporary embedding matrix..."
    )

    build_started = (
        time.perf_counter()
    )

    embeddings = (
        build_embedding_matrix(
            documents=documents,
            embedder=embedder,
            batch_size=(
                arguments.batch_size
            ),
        )
    )

    build_seconds = (
        time.perf_counter()
        - build_started
    )

    norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )
    print(
        "Index build time: "
        f"{build_seconds:.3f} seconds"
    )
    print(
        "Minimum vector norm: "
        f"{norms.min():.8f}"
    )
    print(
        "Maximum vector norm: "
        f"{norms.max():.8f}"
    )

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

    vector_diverse_engine = (
        SourceDiversifiedSearch(
            search_engine=vector_engine,
            max_chunks_per_source=1,
            candidate_multiplier=10,
        )
    )

    hybrid_diverse_engine = (
        SourceDiversifiedSearch(
            search_engine=hybrid_engine,
            max_chunks_per_source=1,
            candidate_multiplier=10,
        )
    )

    print()
    print(
        "Running retrieval evaluation..."
    )

    results = [
        evaluate_engine(
            name="vector",
            ground_truth=ground_truth,
            search_engine=vector_engine,
            num_results=(
                arguments.num_results
            ),
        ),
        evaluate_engine(
            name="hybrid",
            ground_truth=ground_truth,
            search_engine=hybrid_engine,
            num_results=(
                arguments.num_results
            ),
        ),
        evaluate_engine(
            name="vector_diverse_1",
            ground_truth=ground_truth,
            search_engine=(
                vector_diverse_engine
            ),
            num_results=(
                arguments.num_results
            ),
        ),
        evaluate_engine(
            name="hybrid_diverse_1",
            ground_truth=ground_truth,
            search_engine=(
                hybrid_diverse_engine
            ),
            num_results=(
                arguments.num_results
            ),
        ),
    ]

    print()

    print(
        f"{'Method':<22}"
        f"{'Hit Rate':>12}"
        f"{'MRR':>12}"
        f"{'Seconds':>12}"
    )

    print(
        "-" * 58
    )

    for result in results:
        print(
            f"{result['name']:<22}"
            f"{result['hit_rate']:>12.4f}"
            f"{result['mrr']:>12.4f}"
            f"{result['seconds']:>12.3f}"
        )

    print()
    print(
        "Benchmark complete."
    )
    print(
        "The saved data/processed/"
        "embeddings.npy was not modified."
    )


if __name__ == "__main__":
    main()