from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

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


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

FRONTEND_DIR = (
    PROJECT_ROOT
    / "frontend"
)


class AssistantService(Protocol):
    """Interface required by the API layer."""

    def ask(
        self,
        question: str,
        species: str | None = None,
    ) -> dict[str, Any]:
        """Return one grounded pet first-aid response."""


AssistantFactory = Callable[
    [],
    AssistantService,
]

MonitoringFactory = Callable[
    [],
    MonitoringStore,
]


class AskRequest(BaseModel):
    """Request body for the pet first-aid assistant."""

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
    ] | None = Field(
        default=None,
        description=(
            "Optional species used to restrict retrieval."
        ),
    )

    @field_validator("question")
    @classmethod
    def strip_question(
        cls,
        value: str,
    ) -> str:
        """Reject whitespace-only questions and normalize edges."""

        stripped = value.strip()

        if not stripped:
            raise ValueError(
                "question must contain text"
            )

        return stripped


class SourceResponse(BaseModel):
    """One retrieved source exposed to the API client."""

    label: str
    source_id: str | None = None
    publisher: str | None = None
    title: str | None = None
    section: str | None = None
    url: str | None = None
    retrieval_method: str | None = None
    retrieval_score: float | None = None


class RetrievalResponse(BaseModel):
    """Metadata describing the retrieval configuration."""

    method: str
    num_sources: int
    max_chunks_per_source: int


class AskResponse(BaseModel):
    """Successful response returned by POST /ask."""

    interaction_id: str

    answer: str

    species: Literal[
        "dog",
        "cat",
    ] | None = None

    model: str

    sources: list[
        SourceResponse
    ]

    retrieval: RetrievalResponse


class HealthResponse(BaseModel):
    """Basic API health response."""

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
    """Thumbs-up/down feedback for one generated answer."""

    interaction_id: UUID

    rating: Literal[
        -1,
        1,
    ]


class FeedbackResponse(BaseModel):
    """Result of recording user feedback."""

    accepted: bool
    monitoring_enabled: bool
    message: str


class MetricsResponse(BaseModel):
    """Anonymous aggregate monitoring metrics."""

    enabled: bool
    total_requests: int
    requests_last_24_hours: int
    average_latency_ms: float | None
    feedback_total: int
    feedback_positive: int
    feedback_negative: int
    positive_feedback_rate: float | None


def get_assistant(
    request: Request,
) -> AssistantService:
    """Return the assistant created during application startup."""

    assistant = getattr(
        request.app.state,
        "assistant",
        None,
    )

    if assistant is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The assistant service is not initialized."
            ),
        )

    return assistant


def get_monitoring_store(
    request: Request,
) -> MonitoringStore:
    """Return the monitoring store initialized during startup."""

    monitoring = getattr(
        request.app.state,
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
    """
    Create the FastAPI application.

    Factories are injectable so unit tests can avoid loading
    ONNX, OpenAI, and PostgreSQL.
    """

    assistant_builder = (
        assistant_factory
        or PetFirstAidAssistant
    )

    monitoring_builder = (
        monitoring_factory
        or build_monitoring_store
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        """Create expensive application resources once."""

        app.state.assistant = (
            assistant_builder()
        )

        try:
            monitoring = (
                monitoring_builder()
            )

            monitoring.initialize()

        except Exception:
            logger.exception(
                "Monitoring initialization failed. "
                "Continuing without persistent monitoring."
            )

            monitoring = (
                NullMonitoringStore()
            )

        app.state.monitoring = (
            monitoring
        )

        try:
            yield

        finally:
            app.state.assistant = None
            app.state.monitoring = None

    app = FastAPI(
        title=(
            "Pet First Aid Assistant API"
        ),
        description=(
            "Safety-focused, source-grounded first-aid "
            "information for dog and cat emergencies. "
            "This application does not diagnose conditions "
            "and does not replace a veterinarian."
        ),
        version="0.4.0",
        lifespan=lifespan,
    )

    app.mount(
        "/static",
        StaticFiles(
            directory=FRONTEND_DIR,
        ),
        name="static",
    )

    @app.get(
        "/",
        include_in_schema=False,
    )
    def frontend() -> FileResponse:
        """Serve the Pet First Aid Assistant frontend."""

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
    def health(
        monitoring: MonitoringStore = Depends(
            get_monitoring_store
        ),
    ) -> HealthResponse:
        """Return application health and monitoring availability."""

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
        response_model=(
            EmergencyCatalogResponse
        ),
        tags=[
            "assistant",
        ],
    )
    def emergencies() -> EmergencyCatalogResponse:
        """Return predefined non-diagnostic emergency topics."""

        return (
            EmergencyCatalogResponse(
                conditions=(
                    list_emergency_conditions()
                )
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
        request_body: AskRequest,
        assistant: AssistantService = Depends(
            get_assistant
        ),
        monitoring: MonitoringStore = Depends(
            get_monitoring_store
        ),
    ) -> AskResponse:
        """
        Retrieve veterinary context and generate a grounded answer.

        Monitoring failures never block the medical response.
        """

        started_at = (
            perf_counter()
        )

        try:
            result = assistant.ask(
                question=(
                    request_body.question
                ),
                species=(
                    request_body.species
                ),
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(
                    exc
                ),
            ) from exc

        except Exception as exc:
            logger.exception(
                "Pet First Aid Assistant request failed."
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

        try:
            monitoring.record_interaction(
                interaction_id=interaction_id,
                question=(
                    request_body.question
                ),
                species=(
                    request_body.species
                ),
                answer=str(
                    result.get(
                        "answer",
                        "",
                    )
                ),
                model=str(
                    result.get(
                        "model",
                        "",
                    )
                ),
                latency_ms=latency_ms,
                sources=list(
                    result.get(
                        "sources",
                        [],
                    )
                ),
            )

        except Exception:
            logger.exception(
                "Monitoring record failed for interaction %s.",
                interaction_id,
            )

        response_payload = dict(
            result
        )

        response_payload[
            "interaction_id"
        ] = interaction_id

        return AskResponse.model_validate(
            response_payload
        )

    @app.post(
        "/feedback",
        response_model=FeedbackResponse,
        tags=[
            "monitoring",
        ],
    )
    def feedback(
        request_body: FeedbackRequest,
        monitoring: MonitoringStore = Depends(
            get_monitoring_store
        ),
    ) -> FeedbackResponse:
        """Store thumbs-up or thumbs-down feedback."""

        if not monitoring.enabled:
            return FeedbackResponse(
                accepted=False,
                monitoring_enabled=False,
                message=(
                    "Persistent feedback storage "
                    "is not enabled."
                ),
            )

        try:
            accepted = (
                monitoring.record_feedback(
                    interaction_id=str(
                        request_body.interaction_id
                    ),
                    rating=(
                        request_body.rating
                    ),
                )
            )

        except Exception as exc:
            logger.exception(
                "Feedback storage failed."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Feedback could not be stored "
                    "at this time."
                ),
            ) from exc

        if not accepted:
            raise HTTPException(
                status_code=404,
                detail=(
                    "The referenced interaction "
                    "was not found."
                ),
            )

        return FeedbackResponse(
            accepted=True,
            monitoring_enabled=True,
            message=(
                "Thank you for your feedback."
            ),
        )

    @app.get(
        "/metrics",
        response_model=MetricsResponse,
        tags=[
            "monitoring",
        ],
    )
    def metrics(
        monitoring: MonitoringStore = Depends(
            get_monitoring_store
        ),
    ) -> MetricsResponse:
        """Return aggregate anonymous usage and feedback metrics."""

        try:
            values = (
                monitoring.metrics()
            )

        except Exception as exc:
            logger.exception(
                "Monitoring metrics query failed."
            )

            raise HTTPException(
                status_code=503,
                detail=(
                    "Monitoring metrics are temporarily unavailable."
                ),
            ) from exc

        return MetricsResponse.model_validate(
            values
        )

    return app


app = create_app()