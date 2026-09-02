from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any, Literal, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from src.pet_first_aid_assistant.assistant import (
    PetFirstAidAssistant,
)


logger = logging.getLogger(__name__)


class AssistantService(Protocol):
    """Interface required by the API layer."""

    def ask(
        self,
        question: str,
        species: str | None = None,
    ) -> dict[str, Any]:
        """Return one grounded pet first-aid response."""


AssistantFactory = Callable[[], AssistantService]


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

    answer: str

    species: Literal[
        "dog",
        "cat",
    ] | None = None

    model: str

    sources: list[SourceResponse]

    retrieval: RetrievalResponse


class HealthResponse(BaseModel):
    """Basic API health response."""

    status: str
    service: str


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


def create_app(
    assistant_factory: AssistantFactory | None = None,
) -> FastAPI:
    """
    Create the FastAPI application.

    A factory is injectable so tests can use a fake assistant
    without loading ONNX or calling the OpenAI API.
    """

    factory = (
        assistant_factory
        or PetFirstAidAssistant
    )

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        """Create expensive application resources once at startup."""

        app.state.assistant = factory()

        try:
            yield
        finally:
            app.state.assistant = None

    app = FastAPI(
        title="Pet First Aid Assistant API",
        description=(
            "Safety-focused, source-grounded first-aid "
            "information for dog and cat emergencies. "
            "This application does not diagnose conditions "
            "and does not replace a veterinarian."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["system"],
    )
    def health() -> HealthResponse:
        """Return a lightweight service health check."""

        return HealthResponse(
            status="ok",
            service="pet-first-aid-assistant",
        )

    @app.post(
        "/ask",
        response_model=AskResponse,
        tags=["assistant"],
    )
    def ask(
        request_body: AskRequest,
        assistant: AssistantService = Depends(
            get_assistant
        ),
    ) -> AskResponse:
        """
        Retrieve authoritative veterinary context and
        generate one grounded first-aid response.
        """

        try:
            result = assistant.ask(
                question=request_body.question,
                species=request_body.species,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
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

        return AskResponse.model_validate(
            result
        )

    return app


app = create_app()