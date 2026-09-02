from __future__ import annotations

import json
import os
from typing import Any, Protocol

import psycopg


class MonitoringStore(Protocol):
    """Interface used by the API for persistent monitoring."""

    @property
    def enabled(self) -> bool:
        """Return whether persistent monitoring is available."""

    def initialize(self) -> None:
        """Create monitoring tables when required."""

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
        """Store feedback and return whether the interaction exists."""

    def metrics(
        self,
    ) -> dict[str, Any]:
        """Return aggregate monitoring metrics."""


class NullMonitoringStore:
    """
    Monitoring implementation used when DATABASE_URL is unavailable.

    This keeps local development and unit tests independent
    from PostgreSQL.
    """

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

    def metrics(
        self,
    ) -> dict[str, Any]:
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


class PostgresMonitoringStore:
    """PostgreSQL-backed monitoring store."""

    def __init__(
        self,
        database_url: str,
    ):
        if not database_url:
            raise ValueError(
                "database_url must not be empty"
            )

        self.database_url = database_url

    @property
    def enabled(self) -> bool:
        return True

    def _connect(
        self,
    ) -> psycopg.Connection:
        return psycopg.connect(
            self.database_url
        )

    def initialize(self) -> None:
        """Create monitoring tables and indexes."""

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rag_interactions (
                        interaction_id UUID PRIMARY KEY,
                        created_at TIMESTAMPTZ
                            NOT NULL
                            DEFAULT NOW(),
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
                            REFERENCES rag_interactions(
                                interaction_id
                            )
                            ON DELETE CASCADE,
                        rating SMALLINT NOT NULL
                            CHECK (rating IN (-1, 1)),
                        created_at TIMESTAMPTZ
                            NOT NULL
                            DEFAULT NOW()
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
        """Persist one generated answer and its metadata."""

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

    def record_feedback(
        self,
        interaction_id: str,
        rating: int,
    ) -> bool:
        """Insert or replace thumbs-up/down feedback."""

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

                exists = (
                    cursor.fetchone()
                    is not None
                )

                if not exists:
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

        return True

    def metrics(
        self,
    ) -> dict[str, Any]:
        """Return anonymous aggregate monitoring metrics."""

        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS total_requests,
                        COUNT(*) FILTER (
                            WHERE created_at >=
                            NOW() - INTERVAL '24 hours'
                        ) AS requests_last_24_hours,
                        AVG(latency_ms) AS average_latency_ms
                    FROM rag_interactions
                    """
                )

                interaction_row = (
                    cursor.fetchone()
                )

                cursor.execute(
                    """
                    SELECT
                        COUNT(*) AS feedback_total,
                        COUNT(*) FILTER (
                            WHERE rating = 1
                        ) AS feedback_positive,
                        COUNT(*) FILTER (
                            WHERE rating = -1
                        ) AS feedback_negative
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
            "total_requests": (
                total_requests
            ),
            "requests_last_24_hours": (
                requests_last_24_hours
            ),
            "average_latency_ms": (
                round(
                    average_latency,
                    2,
                )
                if average_latency
                is not None
                else None
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
                round(
                    positive_feedback_rate,
                    4,
                )
                if positive_feedback_rate
                is not None
                else None
            ),
        }


def build_monitoring_store(
) -> MonitoringStore:
    """
    Build persistent monitoring when DATABASE_URL exists.

    Local environments without PostgreSQL use a no-op store.
    """

    database_url = os.getenv(
        "DATABASE_URL",
        ""
    ).strip()

    if not database_url:
        return NullMonitoringStore()

    return PostgresMonitoringStore(
        database_url=database_url
    )