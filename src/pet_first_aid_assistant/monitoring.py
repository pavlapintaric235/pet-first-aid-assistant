from __future__ import annotations

import json
import os
from typing import Any, Protocol

import psycopg


class MonitoringStore(Protocol):
    """Interface used by the API monitoring layer."""

    @property
    def enabled(self) -> bool:
        """Return whether persistent monitoring is enabled."""

    def initialize(self) -> None:
        """Create required monitoring tables."""

    def record_interaction(
        self,
        interaction_id: str,
        question: str,
        species: str | None,
        answer: str,
        model: str,
        latency_ms: float,
        sources: list[dict[str, Any]],
    ) -> None:
        """Persist one assistant interaction."""

    def record_feedback(
        self,
        interaction_id: str,
        rating: int,
    ) -> bool:
        """Store feedback for an interaction."""

    def metrics(self) -> dict[str, Any]:
        """Return aggregate monitoring metrics."""

    def dashboard(self) -> dict[str, Any]:
        """Return monitoring dashboard datasets."""


class NullMonitoringStore:
    """No-op monitoring implementation for local/test use."""

    @property
    def enabled(self) -> bool:
        return False

    def initialize(self) -> None:
        return None

    def record_interaction(
        self,
        interaction_id: str,
        question: str,
        species: str | None,
        answer: str,
        model: str,
        latency_ms: float,
        sources: list[dict[str, Any]],
    ) -> None:
        return None

    def record_feedback(
        self,
        interaction_id: str,
        rating: int,
    ) -> bool:
        return False

    def metrics(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "total_requests": 0,
            "requests_last_24_hours": 0,
            "average_latency_ms": None,
            "feedback_total": 0,
            "feedback_positive": 0,
            "feedback_negative": 0,
            "positive_feedback_rate": None,
        }

    def dashboard(self) -> dict[str, Any]:
        return {
            "enabled": False,
            "requests_by_day": [],
            "latency_by_day": [],
            "species_breakdown": [],
            "feedback_breakdown": [],
            "top_sources": [],
        }


class PostgresMonitoringStore:
    """PostgreSQL-backed interaction and feedback monitoring."""

    def __init__(
        self,
        database_url: str,
    ):
        if not database_url:
            raise ValueError(
                "database_url must not be blank"
            )

        self.database_url = database_url

    @property
    def enabled(self) -> bool:
        return True

    def _connect(self):
        return psycopg.connect(
            self.database_url
        )

    def initialize(self) -> None:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_interactions (
                        interaction_id UUID PRIMARY KEY,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        question TEXT NOT NULL,
                        species TEXT,
                        answer TEXT NOT NULL,
                        model TEXT NOT NULL,
                        latency_ms DOUBLE PRECISION NOT NULL,
                        source_count INTEGER NOT NULL,
                        sources_json TEXT NOT NULL
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_feedback (
                        interaction_id UUID PRIMARY KEY
                            REFERENCES rag_interactions(interaction_id)
                            ON DELETE CASCADE,
                        rating SMALLINT NOT NULL
                            CHECK (rating IN (-1, 1)),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )

                cursor.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                    idx_rag_interactions_created_at
                    ON rag_interactions(created_at)
                    """
                )

            connection.commit()

    def record_interaction(
        self,
        interaction_id: str,
        question: str,
        species: str | None,
        answer: str,
        model: str,
        latency_ms: float,
        sources: list[dict[str, Any]],
    ) -> None:
        sources_json = json.dumps(
            sources,
            ensure_ascii=False,
        )

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO rag_interactions (
                        interaction_id,
                        question,
                        species,
                        answer,
                        model,
                        latency_ms,
                        source_count,
                        sources_json
                    )
                    VALUES (
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s,
                        %s
                    )
                    """,
                    (
                        interaction_id,
                        question,
                        species,
                        answer,
                        model,
                        latency_ms,
                        len(sources),
                        sources_json,
                    ),
                )

            connection.commit()

    def record_feedback(
        self,
        interaction_id: str,
        rating: int,
    ) -> bool:
        if rating not in {
            -1,
            1,
        }:
            raise ValueError(
                "rating must be -1 or 1"
            )

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 1
                    FROM rag_interactions
                    WHERE interaction_id = %s
                    """,
                    (
                        interaction_id,
                    ),
                )

                if (
                    cursor.fetchone()
                    is None
                ):
                    return False

                cursor.execute(
                    """
                    INSERT INTO rag_feedback (
                        interaction_id,
                        rating
                    )
                    VALUES (%s, %s)
                    ON CONFLICT (interaction_id)
                    DO UPDATE SET
                        rating = EXCLUDED.rating,
                        created_at = NOW()
                    """,
                    (
                        interaction_id,
                        rating,
                    ),
                )

            connection.commit()

        return True

    def metrics(self) -> dict[str, Any]:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (
                            WHERE created_at >= NOW() - INTERVAL '24 hours'
                        ),
                        AVG(latency_ms)
                    FROM rag_interactions
                    """
                )

                interaction_row = (
                    cursor.fetchone()
                )

                cursor.execute(
                    """
                    SELECT
                        COUNT(*),
                        COUNT(*) FILTER (
                            WHERE rating = 1
                        ),
                        COUNT(*) FILTER (
                            WHERE rating = -1
                        )
                    FROM rag_feedback
                    """
                )

                feedback_row = (
                    cursor.fetchone()
                )

        total_requests = int(
            interaction_row[0]
        )

        requests_last_24_hours = int(
            interaction_row[1]
        )

        average_latency = (
            float(
                interaction_row[2]
            )
            if interaction_row[2]
            is not None
            else None
        )

        feedback_total = int(
            feedback_row[0]
        )

        feedback_positive = int(
            feedback_row[1]
        )

        feedback_negative = int(
            feedback_row[2]
        )

        positive_feedback_rate = (
            feedback_positive
            / feedback_total
            if feedback_total
            else None
        )

        return {
            "enabled": True,
            "total_requests": total_requests,
            "requests_last_24_hours": (
                requests_last_24_hours
            ),
            "average_latency_ms": (
                average_latency
            ),
            "feedback_total": (
                feedback_total
            ),
            "feedback_positive": (
                feedback_positive
            ),
            "feedback_negative": (
                feedback_negative
            ),
            "positive_feedback_rate": (
                positive_feedback_rate
            ),
        }

    def dashboard(self) -> dict[str, Any]:
        """
        Return datasets for five monitoring charts.

        Charts:
        1. Requests per day
        2. Average latency per day
        3. Species distribution
        4. Feedback distribution
        5. Most frequently retrieved sources
        """

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        day::date,
                        COUNT(i.interaction_id)
                    FROM generate_series(
                        CURRENT_DATE - INTERVAL '6 days',
                        CURRENT_DATE,
                        INTERVAL '1 day'
                    ) AS day
                    LEFT JOIN rag_interactions AS i
                        ON i.created_at >= day
                        AND i.created_at < day + INTERVAL '1 day'
                    GROUP BY day
                    ORDER BY day
                    """
                )

                request_rows = (
                    cursor.fetchall()
                )

                cursor.execute(
                    """
                    SELECT
                        day::date,
                        AVG(i.latency_ms)
                    FROM generate_series(
                        CURRENT_DATE - INTERVAL '6 days',
                        CURRENT_DATE,
                        INTERVAL '1 day'
                    ) AS day
                    LEFT JOIN rag_interactions AS i
                        ON i.created_at >= day
                        AND i.created_at < day + INTERVAL '1 day'
                    GROUP BY day
                    ORDER BY day
                    """
                )

                latency_rows = (
                    cursor.fetchall()
                )

                cursor.execute(
                    """
                    SELECT
                        COALESCE(
                            NULLIF(species, ''),
                            'not specified'
                        ) AS species_label,
                        COUNT(*)
                    FROM rag_interactions
                    GROUP BY species_label
                    ORDER BY COUNT(*) DESC
                    """
                )

                species_rows = (
                    cursor.fetchall()
                )

                cursor.execute(
                    """
                    SELECT
                        COUNT(*) FILTER (
                            WHERE rating = 1
                        ),
                        COUNT(*) FILTER (
                            WHERE rating = -1
                        )
                    FROM rag_feedback
                    """
                )

                feedback_row = (
                    cursor.fetchone()
                )

                cursor.execute(
                    """
                    SELECT
                        COALESCE(
                            NULLIF(
                                source_item ->> 'publisher',
                                ''
                            ),
                            NULLIF(
                                source_item ->> 'source_id',
                                ''
                            ),
                            'Unknown source'
                        ) AS source_label,
                        COUNT(*) AS retrieval_count
                    FROM rag_interactions
                    CROSS JOIN LATERAL
                        jsonb_array_elements(
                            sources_json::jsonb
                        ) AS source_item
                    GROUP BY source_label
                    ORDER BY retrieval_count DESC
                    LIMIT 5
                    """
                )

                source_rows = (
                    cursor.fetchall()
                )

        requests_by_day = [
            {
                "label": row[0].isoformat(),
                "value": int(
                    row[1]
                ),
            }
            for row in request_rows
        ]

        latency_by_day = [
            {
                "label": row[0].isoformat(),
                "value": (
                    round(
                        float(
                            row[1]
                        ),
                        2,
                    )
                    if row[1]
                    is not None
                    else None
                ),
            }
            for row in latency_rows
        ]

        species_breakdown = [
            {
                "label": str(
                    row[0]
                ),
                "value": int(
                    row[1]
                ),
            }
            for row in species_rows
        ]

        feedback_breakdown = [
            {
                "label": "Positive",
                "value": int(
                    feedback_row[0]
                ),
            },
            {
                "label": "Negative",
                "value": int(
                    feedback_row[1]
                ),
            },
        ]

        top_sources = [
            {
                "label": str(
                    row[0]
                ),
                "value": int(
                    row[1]
                ),
            }
            for row in source_rows
        ]

        return {
            "enabled": True,
            "requests_by_day": (
                requests_by_day
            ),
            "latency_by_day": (
                latency_by_day
            ),
            "species_breakdown": (
                species_breakdown
            ),
            "feedback_breakdown": (
                feedback_breakdown
            ),
            "top_sources": (
                top_sources
            ),
        }


def build_monitoring_store() -> MonitoringStore:
    """Build monitoring from DATABASE_URL when available."""

    database_url = os.getenv(
        "DATABASE_URL"
    )

    if not database_url:
        return (
            NullMonitoringStore()
        )

    return PostgresMonitoringStore(
        database_url=database_url
    )