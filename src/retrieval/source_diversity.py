from __future__ import annotations

from typing import Any, Protocol


DEFAULT_CANDIDATE_MULTIPLIER = 10
DEFAULT_MAX_CHUNKS_PER_SOURCE = 1


class SearchEngine(Protocol):
    """Interface required by the diversity wrapper."""

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked retrieval results."""


def diversify_results(
    results: list[dict[str, Any]],
    num_results: int,
    max_chunks_per_source: int = (
        DEFAULT_MAX_CHUNKS_PER_SOURCE
    ),
) -> list[dict[str, Any]]:
    """Limit how many chunks one source can contribute."""

    if num_results <= 0:
        raise ValueError(
            "num_results must be greater than zero."
        )

    if max_chunks_per_source <= 0:
        raise ValueError(
            "max_chunks_per_source must be greater than zero."
        )

    source_counts: dict[str, int] = {}
    diversified: list[dict[str, Any]] = []

    for result in results:
        source_id = result["source_id"]

        current_count = source_counts.get(
            source_id,
            0,
        )

        if current_count >= max_chunks_per_source:
            continue

        diversified.append(result)

        source_counts[source_id] = (
            current_count + 1
        )

        if len(diversified) == num_results:
            break

    return diversified


class SourceDiversifiedSearch:
    """Apply source-diversity reranking to any search engine."""

    def __init__(
        self,
        search_engine: SearchEngine,
        max_chunks_per_source: int = (
            DEFAULT_MAX_CHUNKS_PER_SOURCE
        ),
        candidate_multiplier: int = (
            DEFAULT_CANDIDATE_MULTIPLIER
        ),
    ) -> None:
        if max_chunks_per_source <= 0:
            raise ValueError(
                "max_chunks_per_source must be greater "
                "than zero."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater "
                "than zero."
            )

        self.search_engine = search_engine
        self.max_chunks_per_source = (
            max_chunks_per_source
        )
        self.candidate_multiplier = (
            candidate_multiplier
        )

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search a larger pool and diversify final results."""

        if num_results <= 0:
            raise ValueError(
                "num_results must be greater than zero."
            )

        num_candidates = (
            num_results
            * self.candidate_multiplier
        )

        candidates = self.search_engine.search(
            query=query,
            num_results=num_candidates,
            species=species,
        )

        diversified = diversify_results(
            results=candidates,
            num_results=num_results,
            max_chunks_per_source=(
                self.max_chunks_per_source
            ),
        )

        for result in diversified:
            result["source_diversified"] = True
            result["max_chunks_per_source"] = (
                self.max_chunks_per_source
            )

        return diversified