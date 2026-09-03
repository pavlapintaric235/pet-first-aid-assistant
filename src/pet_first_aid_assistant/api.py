from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.pet_first_aid_assistant.assistant import (
    PetFirstAidAssistant,
)
from src.pet_first_aid_assistant.emergency_conditions import (
    list_emergency_conditions,
)
from src.pet_first_aid_assistant.monitoring import (
    MonitoringStore,
    NullMonitoringStore,
    build_monitoring_store,
)


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FRONTEND_DIR = PROJECT_ROOT / "frontend"


class AssistantProtocol(Protocol):
    """Minimal assistant interface used by the API."""

    def ask(
        self,
        question: str,
        species: str | None = None,
    ) -> dict[str, Any]:
        """Answer one pet first-aid question."""


AssistantFactory = Callable[
    [],
    AssistantProtocol,
]

MonitoringFactory = Callable[
    [],
    MonitoringStore,
]


class AskRequest(BaseModel):
    """Request body accepted by POST /ask."""

    question: str = Field(
        min_length=3,
        max_length=2000,
        description=(
            "A dog or cat emergency question "
            "or symptom description."
        ),
    )

    species: Literal[
        "dog",
        "cat",
    ] | None = None

    @field_validator(
        "question",
        mode="before",
    )
    @classmethod
    def strip_question(
        cls,
        value: Any,
    ) -> Any:
        """
        Strip surrounding whitespace before length validation.

        This ensures values such as three spaces do not satisfy
        the minimum-length requirement.
        """

        if isinstance(
            value,
            str,
        ):
            return value.strip()

        return value


class SourceResponse(BaseModel):
    """One source shown with an assistant answer."""

    label: str
    source_id: str | None = None
    publisher: str | None = None
    title: str | None = None
    section: str | None = None
    url: str | None = None
    retrieval_method: str | None = None
    retrieval_score: float | None = None


class RetrievalResponse(BaseModel):
    """Metadata describing the production retrieval pipeline."""

    method: str
    num_sources: int
    max_chunks_per_source: int


class AskResponse(BaseModel):
    """Response returned by POST /ask."""

    interaction_id: str
    answer: str
    species: str | None
    model: str

    sources: list[
        SourceResponse
    ]

    retrieval: RetrievalResponse


class HealthResponse(BaseModel):
    """Application health response."""

    status: str
    service: str
    monitoring_enabled: bool


class EmergencyConditionResponse(BaseModel):
    """One frontend-safe predefined emergency option."""

    id: str
    title: str
    short_description: str
    starter_question: str

    urgency: Literal[
        "urgent",
        "emergency",
    ]

    supported_species: tuple[
        Literal[
            "dog",
            "cat",
        ],
        ...,
    ]


class EmergencyCatalogResponse(BaseModel):
    """Collection returned by GET /emergencies."""

    conditions: list[
        EmergencyConditionResponse
    ]


class FeedbackRequest(BaseModel):
    """User feedback for one monitored interaction."""

    interaction_id: UUID

    rating: Literal[
        -1,
        1,
    ]


class FeedbackResponse(BaseModel):
    """Result of storing user feedback."""

    accepted: bool
    monitoring_enabled: bool
    message: str


class MetricsResponse(BaseModel):
    """Aggregate monitoring metrics."""

    enabled: bool
    total_requests: int
    requests_last_24_hours: int
    average_latency_ms: float | None
    feedback_total: int
    feedback_positive: int
    feedback_negative: int
    positive_feedback_rate: float | None


class ChartPoint(BaseModel):
    """One label/value point used by the monitoring dashboard."""

    label: str
    value: int | float | None


class DashboardDataResponse(BaseModel):
    """Datasets used by the five monitoring charts."""

    enabled: bool

    requests_by_day: list[
        ChartPoint
    ]

    latency_by_day: list[
        ChartPoint
    ]

    species_breakdown: list[
        ChartPoint
    ]

    feedback_breakdown: list[
        ChartPoint
    ]

    top_sources: list[
        ChartPoint
    ]


def get_monitoring_store(
    app: FastAPI,
) -> MonitoringStore:
    """Return the active monitoring store."""

    monitoring = getattr(
        app.state,
        "monitoring",
        None,
    )

    if monitoring is None:
        return NullMonitoringStore()

    return monitoring


def create_app(
    assistant_factory: AssistantFactory | None = None,
    monitoring_factory: MonitoringFactory | None = None,
) -> FastAPI:
    """Create the Pet First Aid Assistant FastAPI application."""

    actual_assistant_factory = (
        assistant_factory
        or PetFirstAidAssistant
    )

    actual_monitoring_factory = (
        monitoring_factory
        or build_monitoring_store
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        app.state.assistant = (
            actual_assistant_factory()
        )

        try:
            monitoring = (
                actual_monitoring_factory()
            )

            monitoring.initialize()

        except Exception:
            LOGGER.exception(
                "Monitoring initialization failed. "
                "Continuing without monitoring."
            )

            monitoring = (
                NullMonitoringStore()
            )

        app.state.monitoring = monitoring

        yield

        app.state.assistant = None
        app.state.monitoring = None

    app = FastAPI(
        title="Pet First Aid Assistant",
        description=(
            "Safety-focused, source-grounded first-aid "
            "information for dog and cat emergencies. "
            "This application does not diagnose conditions "
            "and does not replace a veterinarian."
        ),
        version="0.5.0",
        lifespan=lifespan,
    )

    app.mount(
        "/static",
        StaticFiles(
            directory=FRONTEND_DIR
        ),
        name="static",
    )

    @app.get(
        "/",
        include_in_schema=False,
    )
    def frontend():
        """Serve the main web application."""

        return FileResponse(
            FRONTEND_DIR
            / "index.html"
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=[
            "system",
        ],
    )
    def health() -> HealthResponse:
        """Return application and monitoring health information."""

        monitoring = (
            get_monitoring_store(
                app
            )
        )

        return HealthResponse(
            status="ok",
            service=(
                "pet-first-aid-assistant"
            ),
            monitoring_enabled=(
                monitoring.enabled
            ),
        )

    @app.get(
        "/emergencies",
        response_model=EmergencyCatalogResponse,
        tags=[
            "assistant",
        ],
    )
    def emergencies() -> EmergencyCatalogResponse:
        """Return predefined non-diagnostic emergency topics."""

        return EmergencyCatalogResponse(
            conditions=(
                list_emergency_conditions()
            )
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=[
            "assistant",
        ],
    )
    def ask(
        request: AskRequest,
    ) -> AskResponse:
        """Run retrieval and grounded generation."""

        assistant = getattr(
            app.state,
            "assistant",
            None,
        )

        if assistant is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "The assistant is temporarily unavailable. "
                    "For an emergency, contact a veterinarian "
                    "or emergency veterinary clinic directly."
                ),
            )

        started_at = (
            perf_counter()
        )

        try:
            result = assistant.ask(
                question=(
                    request.question
                ),
                species=(
                    request.species
                ),
            )

        except Exception as exc:
            LOGGER.exception(
                "Assistant request failed."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "The assistant is temporarily unavailable. "
                    "For an emergency, contact a veterinarian "
                    "or emergency veterinary clinic directly."
                ),
            ) from exc

        latency_ms = (
            perf_counter()
            - started_at
        ) * 1000

        interaction_id = str(
            uuid4()
        )

        monitoring = (
            get_monitoring_store(
                app
            )
        )

        try:
            monitoring.record_interaction(
                interaction_id=(
                    interaction_id
                ),
                question=(
                    request.question
                ),
                species=(
                    request.species
                ),
                answer=(
                    result["answer"]
                ),
                model=(
                    result["model"]
                ),
                latency_ms=(
                    latency_ms
                ),
                sources=(
                    result["sources"]
                ),
            )

        except Exception:
            LOGGER.exception(
                "Failed to persist monitoring interaction."
            )

        return AskResponse(
            interaction_id=(
                interaction_id
            ),
            answer=(
                result["answer"]
            ),
            species=(
                result["species"]
            ),
            model=(
                result["model"]
            ),
            sources=(
                result["sources"]
            ),
            retrieval=(
                result["retrieval"]
            ),
        )

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        tags=[
            "monitoring",
        ],
    )
    def feedback(
        request: FeedbackRequest,
    ) -> FeedbackResponse:
        """Record positive or negative feedback."""

        monitoring = (
            get_monitoring_store(
                app
            )
        )

        if not monitoring.enabled:
            return FeedbackResponse(
                accepted=False,
                monitoring_enabled=False,
                message=(
                    "Feedback storage is not enabled "
                    "for this environment."
                ),
            )

        try:
            accepted = (
                monitoring.record_feedback(
                    interaction_id=str(
                        request.interaction_id
                    ),
                    rating=(
                        request.rating
                    ),
                )
            )

        except Exception as exc:
            LOGGER.exception(
                "Feedback storage failed."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Feedback could not be stored."
                ),
            ) from exc

        if not accepted:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The interaction was not found."
                ),
            )

        return FeedbackResponse(
            accepted=True,
            monitoring_enabled=True,
            message="Feedback recorded.",
        )

    @app.get(
        "/metrics",
        response_model=MetricsResponse,
        tags=[
            "monitoring",
        ],
    )
    def metrics() -> MetricsResponse:
        """Return aggregate monitoring metrics."""

        monitoring = (
            get_monitoring_store(
                app
            )
        )

        try:
            data = monitoring.metrics()

        except Exception as exc:
            LOGGER.exception(
                "Metrics query failed."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Monitoring metrics are "
                    "temporarily unavailable."
                ),
            ) from exc

        return MetricsResponse(
            **data
        )

    @app.get(
        "/dashboard-data",
        response_model=DashboardDataResponse,
        tags=[
            "monitoring",
        ],
    )
    def dashboard_data() -> DashboardDataResponse:
        """Return datasets for the monitoring dashboard."""

        monitoring = (
            get_monitoring_store(
                app
            )
        )

        try:
            data = (
                monitoring.dashboard()
            )

        except Exception as exc:
            LOGGER.exception(
                "Dashboard query failed."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Monitoring dashboard data "
                    "is temporarily unavailable."
                ),
            ) from exc

        return DashboardDataResponse(
            **data
        )

    return app


app = create_app()