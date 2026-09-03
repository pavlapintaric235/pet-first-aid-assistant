from __future__ import annotations

from typing import Any

import numpy as np

from src.retrieval.vector_search import VectorSearch


DEFAULT_LAMBDA = 0.75
DEFAULT_CANDIDATE_MULTIPLIER = 4


class MMRSearch:
    """Rerank vector candidates with Maximum Marginal Relevance."""

    def __init__(
        self,
        vector_search: VectorSearch,
        lambda_mult: float = DEFAULT_LAMBDA,
        candidate_multiplier: int = DEFAULT_CANDIDATE_MULTIPLIER,
    ) -> None:
        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0.0 and 1.0."
            )

        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier must be greater than zero."
            )

        self.vector_search = vector_search
        self.lambda_mult = lambda_mult
        self.candidate_multiplier = candidate_multiplier

        self._document_index_by_id: dict[str, int] = {}

        for index, document in enumerate(
            self.vector_search.documents
        ):
            document_id = document.get("id")

            if (
                not isinstance(document_id, str)
                or not document_id
            ):
                raise ValueError(
                    "Every document must have a non-empty "
                    "string id for MMR reranking."
                )

            if document_id in self._document_index_by_id:
                raise ValueError(
                    "Document ids must be unique "
                    "for MMR reranking."
                )

            self._document_index_by_id[
                document_id
            ] = index

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return a relevance/diversity-balanced vector ranking."""

        if num_results <= 0:
            raise ValueError(
                "num_results must be greater than zero."
            )

        num_candidates = max(
            num_results,
            num_results * self.candidate_multiplier,
        )

        candidates = self.vector_search.search(
            query=query,
            num_results=num_candidates,
            species=species,
        )

        if not candidates:
            return []

        candidate_indices = np.array(
            [
                self._document_index_by_id[
                    candidate["id"]
                ]
                for candidate in candidates
            ],
            dtype=np.int64,
        )

        candidate_embeddings = (
            self.vector_search.embeddings[
                candidate_indices
            ]
        )

        relevance_scores = np.array(
            [
                float(
                    candidate[
                        "retrieval_score"
                    ]
                )
                for candidate in candidates
            ],
            dtype=np.float32,
        )

        selected_positions: list[int] = []
        remaining_positions = list(
            range(len(candidates))
        )

        while (
            remaining_positions
            and len(selected_positions)
            < num_results
        ):
            best_position: int | None = None
            best_score = float("-inf")
            best_redundancy = 0.0

            for position in remaining_positions:
                relevance = float(
                    relevance_scores[position]
                )

                if not selected_positions:
                    redundancy = 0.0

                else:
                    similarities = (
                        candidate_embeddings[
                            selected_positions
                        ].dot(
                            candidate_embeddings[
                                position
                            ]
                        )
                    )

                    redundancy = float(
                        np.max(
                            similarities
                        )
                    )

                mmr_score = (
                    self.lambda_mult
                    * relevance
                    - (
                        1.0
                        - self.lambda_mult
                    )
                    * redundancy
                )

                if mmr_score > best_score:
                    best_position = position
                    best_score = mmr_score
                    best_redundancy = (
                        redundancy
                    )

            assert best_position is not None

            selected_positions.append(
                best_position
            )

            remaining_positions.remove(
                best_position
            )

            selected = candidates[
                best_position
            ]

            selected["vector_score"] = float(
                relevance_scores[
                    best_position
                ]
            )

            selected["mmr_score"] = float(
                best_score
            )

            selected["mmr_redundancy"] = float(
                best_redundancy
            )

            selected["mmr_lambda"] = (
                self.lambda_mult
            )

            selected["retrieval_score"] = float(
                best_score
            )

            selected[
                "retrieval_method"
            ] = "vector_mmr"

        return [
            candidates[position]
            for position
            in selected_positions
        ]