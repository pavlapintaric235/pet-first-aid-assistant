from __future__ import annotations

import os
from typing import Any, Protocol

from openai import OpenAI

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


DEFAULT_MODEL = "gpt-5.6-terra"

DEFAULT_NUMBER_OF_SOURCES = 4

DEFAULT_MAX_CHUNKS_PER_SOURCE = 1

DEFAULT_DIVERSITY_CANDIDATE_MULTIPLIER = 10

DEFAULT_PROMPT_PROFILE = "strict_relevance"


BASELINE_SYSTEM_INSTRUCTIONS = """
You are a safety-focused educational pet first-aid assistant
for dog and cat emergencies.

Follow these rules strictly:

- Do not diagnose a medical condition or present a suspected
  diagnosis as a confirmed fact.
- Do not replace, discourage, or delay professional veterinary care.
- Do not prescribe medications or provide medication doses.
- Do not recommend inducing vomiting unless the retrieved source
  explicitly states that this should only be done under direct
  veterinarian or animal poison-control instruction.
- Never provide a hydrogen-peroxide dose.
- Do not claim that the application or its advice is clinically validated,
  clinically proven, medically validated, or clinically verified.
- Use only the retrieved source excerpts for factual first-aid
  instructions. Do not add first-aid instructions from memory.
- If the retrieved context is insufficient, say so and recommend
  contacting a veterinarian or emergency veterinary service.
- If retrieved sources conflict, do not guess. State that the
  guidance differs and recommend professional veterinary guidance.
- Express uncertainty when the available information is uncertain.
- Prioritize immediate safety, safe handling, safe transport,
  and veterinary escalation when supported by retrieved sources.
- Cite factual first-aid statements using source labels such as
  [S1], [S2], and [S3].
- Never invent a citation.
- Keep the response concise and practical.

Use this general response structure when appropriate:

1. Urgency
2. What to do now
3. What not to do / safety warning
4. When to seek veterinary care

Do not include a separate bibliography because the application
displays the source links separately.
""".strip()


STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS = """
You are a safety-focused educational pet first-aid assistant
for dog and cat emergencies.

Follow these rules strictly:

- Do not diagnose a medical condition or present a suspected
  diagnosis as a confirmed fact.
- Do not replace, discourage, or delay professional veterinary care.
- Do not prescribe medications or provide medication doses.
- Do not recommend inducing vomiting unless the retrieved source
  explicitly states that this should only be done under direct
  veterinarian or animal poison-control instruction.
- Never provide a hydrogen-peroxide dose.
- Do not claim that the application or its advice is clinically validated,
  clinically proven, medically validated, or clinically verified.

Grounding and relevance rules:

- Use only the retrieved source excerpts for factual first-aid
  instructions. Do not add first-aid instructions from memory.
- Use only excerpts that are directly relevant to the situation
  described by the user.
- Ignore retrieved passages that concern a different emergency,
  mechanism of injury, exposure, or clinical situation.
- Do not include tangential first-aid instructions merely because
  they appeared in retrieval results.
- Do not assume an exposure, injury mechanism, or symptom that the
  user did not mention.
- Do not introduce condition-specific treatment for a condition
  that has not been described by the user.
- Do not introduce CPR, rescue breathing, chest-compression
  instructions, or CPR rates unless the user describes the pet as
  unresponsive, not breathing, without a heartbeat, or asks about
  CPR directly.
- Conditional advice about deterioration should be included only
  when it is necessary for immediate safety and directly supported
  by a relevant retrieved excerpt.
- If the retrieved context does not directly answer the user's
  situation, state that the available information is insufficient
  rather than filling the gap with loosely related material.
- If retrieved sources conflict, do not guess. State that the
  guidance differs and recommend professional veterinary guidance.
- Express uncertainty when the available information is uncertain.

Response rules:

- Prioritize immediate safety, safe handling, safe transport,
  and veterinary escalation when directly relevant and supported.
- Cite factual first-aid statements using source labels such as
  [S1], [S2], and [S3].
- Never invent a citation.
- Keep the response concise and focused on the user's actual
  situation.

Use this general response structure when appropriate:

1. Urgency
2. What to do now
3. What not to do / safety warning
4. When to seek veterinary care

Do not include a separate bibliography because the application
displays the source links separately.
""".strip()


PROMPT_PROFILES = {
    "baseline": BASELINE_SYSTEM_INSTRUCTIONS,
    "strict_relevance": STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS,
}


# Compatibility alias that always represents the active production prompt.
SYSTEM_INSTRUCTIONS = PROMPT_PROFILES[
    DEFAULT_PROMPT_PROFILE
]


class SearchEngine(Protocol):
    """Search interface required by the assistant."""

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return relevant document chunks."""


class ResponsesAPI(Protocol):
    """Minimal Responses API interface used by the assistant."""

    def create(
        self,
        **kwargs: Any,
    ) -> Any:
        """Create one model response."""


class OpenAIClient(Protocol):
    """Minimal OpenAI client interface."""

    responses: ResponsesAPI


def resolve_prompt_profile(
    prompt_profile: str,
) -> str:
    """Return system instructions for a named prompt profile."""

    if prompt_profile not in PROMPT_PROFILES:
        available = ", ".join(
            sorted(
                PROMPT_PROFILES
            )
        )

        raise ValueError(
            "Unknown prompt profile "
            f"'{prompt_profile}'. "
            f"Available profiles: {available}"
        )

    return PROMPT_PROFILES[
        prompt_profile
    ]


def _section_path(
    document: dict[str, Any],
) -> str:
    """Return the most useful section label available."""

    heading_path = document.get(
        "heading_path"
    )

    if isinstance(
        heading_path,
        list,
    ):
        values = [
            str(value).strip()
            for value in heading_path
            if str(value).strip()
        ]

        if values:
            return " > ".join(
                values
            )

    if isinstance(
        heading_path,
        str,
    ):
        cleaned = (
            heading_path.strip()
        )

        if cleaned:
            return cleaned

    section_heading = document.get(
        "section_heading"
    )

    if section_heading:
        return str(
            section_heading
        ).strip()

    return ""


def build_context(
    results: list[dict[str, Any]],
) -> str:
    """Build labelled source context for the generation model."""

    if not results:
        return (
            "No retrieved source excerpts "
            "were available."
        )

    blocks: list[str] = []

    for index, document in enumerate(
        results,
        start=1,
    ):
        label = f"S{index}"

        publisher = str(
            document.get(
                "publisher",
                "",
            )
        ).strip()

        title = str(
            document.get(
                "title",
                "",
            )
        ).strip()

        url = str(
            document.get(
                "url",
                "",
            )
        ).strip()

        section = _section_path(
            document
        )

        content = str(
            document.get(
                "content",
                "",
            )
        ).strip()

        lines = [
            f"[{label}]",
            f"Publisher: {publisher}",
            f"Title: {title}",
        ]

        if section:
            lines.append(
                f"Section: {section}"
            )

        if url:
            lines.append(
                f"URL: {url}"
            )

        lines.extend(
            [
                "Excerpt:",
                content,
            ]
        )

        blocks.append(
            "\n".join(
                lines
            )
        )

    return "\n\n".join(
        blocks
    )


def build_user_input(
    question: str,
    species: str | None,
    results: list[dict[str, Any]],
) -> str:
    """Build one grounded generation request."""

    cleaned_question = (
        question.strip()
    )

    if not cleaned_question:
        raise ValueError(
            "question must not be blank"
        )

    if (
        species is not None
        and species not in {
            "dog",
            "cat",
        }
    ):
        raise ValueError(
            "species must be 'dog', 'cat', or None"
        )

    species_text = (
        species
        if species
        else "not specified"
    )

    context = build_context(
        results
    )

    return f"""
Pet species: {species_text}

User question:
{cleaned_question}

Retrieved veterinary source excerpts:
{context}

Answer using only the retrieved excerpts for factual first-aid
instructions.

Do not infer a diagnosis or confirm a suspected diagnosis.

If the retrieved excerpts do not directly support an answer,
say that the available information is insufficient and recommend
professional veterinary guidance.
""".strip()


def build_production_retriever(
) -> SearchEngine:
    """Build the production retrieval pipeline."""

    documents = (
        load_processed_documents()
    )

    embeddings = (
        load_embedding_matrix()
    )

    embedder = Embedder()

    keyword_search = (
        KeywordSearch(
            documents=documents
        )
    )

    vector_search = (
        VectorSearch(
            documents=documents,
            embeddings=embeddings,
            embedder=embedder,
        )
    )

    hybrid_search = (
        HybridSearch(
            keyword_search=keyword_search,
            vector_search=vector_search,
        )
    )

    return SourceDiversifiedSearch(
        search_engine=hybrid_search,
        max_chunks_per_source=(
            DEFAULT_MAX_CHUNKS_PER_SOURCE
        ),
        candidate_multiplier=(
            DEFAULT_DIVERSITY_CANDIDATE_MULTIPLIER
        ),
    )


def source_summaries(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return frontend-safe source metadata."""

    summaries: list[
        dict[str, Any]
    ] = []

    for index, document in enumerate(
        results,
        start=1,
    ):
        summaries.append(
            {
                "label": f"S{index}",
                "source_id": document.get(
                    "source_id"
                ),
                "publisher": document.get(
                    "publisher"
                ),
                "title": document.get(
                    "title"
                ),
                "section": _section_path(
                    document
                ),
                "url": document.get(
                    "url"
                ),
                "retrieval_method": document.get(
                    "retrieval_method"
                ),
                "retrieval_score": document.get(
                    "retrieval_score"
                ),
            }
        )

    return summaries


class PetFirstAidAssistant:
    """Production grounded pet first-aid RAG assistant."""

    def __init__(
        self,
        retriever: SearchEngine | None = None,
        client: OpenAIClient | None = None,
        model: str | None = None,
        num_sources: int = DEFAULT_NUMBER_OF_SOURCES,
        prompt_profile: str = DEFAULT_PROMPT_PROFILE,
    ):
        if num_sources <= 0:
            raise ValueError(
                "num_sources must be greater than zero"
            )

        self.retriever = (
            retriever
            or build_production_retriever()
        )

        self.client = (
            client
            or OpenAI()
        )

        self.model = (
            model
            or os.getenv(
                "OPENAI_MODEL",
                DEFAULT_MODEL,
            )
        )

        self.num_sources = (
            num_sources
        )

        self.prompt_profile = (
            prompt_profile
        )

        self.instructions = (
            resolve_prompt_profile(
                prompt_profile
            )
        )

    def ask(
        self,
        question: str,
        species: str | None = None,
    ) -> dict[str, Any]:
        """Retrieve sources and generate one grounded answer."""

        cleaned_question = (
            question.strip()
        )

        if not cleaned_question:
            raise ValueError(
                "question must not be blank"
            )

        if (
            species is not None
            and species not in {
                "dog",
                "cat",
            }
        ):
            raise ValueError(
                "species must be 'dog', 'cat', or None"
            )

        results = (
            self.retriever.search(
                query=cleaned_question,
                num_results=(
                    self.num_sources
                ),
                species=species,
            )
        )

        user_input = (
            build_user_input(
                question=cleaned_question,
                species=species,
                results=results,
            )
        )

        response = (
            self.client.responses.create(
                model=self.model,
                reasoning={
                    "effort": "low",
                },
                instructions=(
                    self.instructions
                ),
                input=user_input,
            )
        )

        answer = str(
            response.output_text
        ).strip()

        if not answer:
            raise RuntimeError(
                "The generation model returned an empty answer."
            )

        return {
            "answer": answer,
            "species": species,
            "model": self.model,
            "sources": source_summaries(
                results
            ),
            "retrieval": {
                "method": (
                    "hybrid_source_diverse"
                ),
                "num_sources": (
                    self.num_sources
                ),
                "max_chunks_per_source": (
                    DEFAULT_MAX_CHUNKS_PER_SOURCE
                ),
            },
        }