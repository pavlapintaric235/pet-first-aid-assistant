from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "documents.json"
)

DEFAULT_NUMBER_OF_RESULTS = 5


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase alphanumeric search tokens."""

    if not isinstance(text, str):
        raise TypeError("Search text must be a string.")

    return re.findall(r"[a-z0-9]+", text.lower())


def load_processed_documents(
    documents_path: Path = PROCESSED_DOCUMENTS_PATH,
) -> list[dict[str, Any]]:
    """Load the processed retrieval documents."""

    if not documents_path.exists():
        raise FileNotFoundError(
            f"Processed documents were not found at "
            f"{documents_path}. Run process_sources.py first."
        )

    with documents_path.open("r", encoding="utf-8") as file:
        documents = json.load(file)

    if not isinstance(documents, list):
        raise ValueError(
            "Processed documents must contain a JSON list."
        )

    if not documents:
        raise ValueError(
            "Processed documents cannot be empty."
        )

    required_fields = {
        "id",
        "source_id",
        "publisher",
        "title",
        "url",
        "species",
        "topics",
        "content",
    }

    for position, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(
                f"Document at position {position} "
                f"must be a JSON object."
            )

        missing_fields = required_fields - document.keys()

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))

            raise ValueError(
                f"Document at position {position} "
                f"is missing fields: {missing}"
            )

    return documents


def build_index_text(
    document: dict[str, Any],
) -> str:
    """Build weighted index text from searchable document fields."""

    title = document["title"]
    topics = " ".join(document["topics"])
    content = document["content"]

    weighted_parts = [
        title,
        title,
        title,
        topics,
        topics,
        topics,
        topics,
        content,
    ]

    return " ".join(weighted_parts)


class KeywordSearch:
    """BM25 keyword search over processed veterinary documents."""

    def __init__(
        self,
        documents: list[dict[str, Any]],
    ) -> None:
        if not documents:
            raise ValueError(
                "At least one document is required to build search."
            )

        self.documents = documents

        tokenized_documents = [
            tokenize(build_index_text(document))
            for document in documents
        ]

        self.index = BM25Okapi(tokenized_documents)

    def search(
        self,
        query: str,
        num_results: int = DEFAULT_NUMBER_OF_RESULTS,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the highest-scoring documents for a query."""

        query_tokens = tokenize(query)

        if not query_tokens:
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

        scores = self.index.get_scores(query_tokens)

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
            result["retrieval_method"] = "keyword"

            results.append(result)

            if len(results) == num_results:
                break

        return results


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Search processed veterinary documents with BM25."
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
    """Print search results in a readable terminal format."""

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
    """Load the documents and run one keyword search."""

    arguments = parse_arguments()

    documents = load_processed_documents()
    search_engine = KeywordSearch(documents)

    results = search_engine.search(
        query=arguments.query,
        num_results=arguments.num_results,
        species=arguments.species,
    )

    print_results(results)


if __name__ == "__main__":
    main()