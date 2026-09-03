import pytest

from src.pet_first_aid_assistant.assistant import (
    BASELINE_SYSTEM_INSTRUCTIONS,
    DEFAULT_PROMPT_PROFILE,
    PROMPT_PROFILES,
    STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS,
    PetFirstAidAssistant,
    resolve_prompt_profile,
)


class FakeRetriever:
    def search(
        self,
        query,
        num_results=5,
        species=None,
    ):
        return [
            {
                "source_id": "test",
                "publisher": "Test Publisher",
                "title": "Test Source",
                "heading_path": [
                    "Emergency",
                ],
                "url": (
                    "https://example.com"
                ),
                "content": (
                    "Seek veterinary care."
                ),
                "retrieval_method": (
                    "hybrid"
                ),
                "retrieval_score": 1.0,
            }
        ]


class FakeResponse:
    output_text = (
        "Seek veterinary care [S1]."
    )


class FakeResponses:
    def __init__(self):
        self.request = None

    def create(
        self,
        **kwargs,
    ):
        self.request = kwargs

        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.responses = (
            FakeResponses()
        )


def test_prompt_profiles_exist():
    assert set(
        PROMPT_PROFILES
    ) == {
        "baseline",
        "strict_relevance",
    }


def test_production_default_uses_winning_profile():
    assert (
        DEFAULT_PROMPT_PROFILE
        == "strict_relevance"
    )


def test_resolve_baseline_profile():
    assert (
        resolve_prompt_profile(
            "baseline"
        )
        == BASELINE_SYSTEM_INSTRUCTIONS
    )


def test_resolve_strict_relevance_profile():
    assert (
        resolve_prompt_profile(
            "strict_relevance"
        )
        == STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS
    )


def test_unknown_prompt_profile_is_rejected():
    with pytest.raises(
        ValueError
    ):
        resolve_prompt_profile(
            "unknown"
        )


def test_strict_profile_contains_relevance_controls():
    lowered = (
        STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS.lower()
    )

    assert (
        "directly relevant"
        in lowered
    )

    assert (
        "tangential"
        in lowered
    )

    assert (
        "do not assume"
        in lowered
    )

    assert (
        "cpr"
        in lowered
    )


def test_assistant_uses_selected_prompt_profile():
    client = FakeClient()

    assistant = (
        PetFirstAidAssistant(
            retriever=(
                FakeRetriever()
            ),
            client=client,
            model="fake-model",
            prompt_profile=(
                "strict_relevance"
            ),
        )
    )

    assistant.ask(
        question=(
            "My dog is injured."
        ),
        species="dog",
    )

    request = (
        client.responses.request
    )

    assert (
        request["instructions"]
        == STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS
    )


def test_default_assistant_uses_strict_relevance_profile():
    client = FakeClient()

    assistant = (
        PetFirstAidAssistant(
            retriever=(
                FakeRetriever()
            ),
            client=client,
            model="fake-model",
        )
    )

    assistant.ask(
        question=(
            "My dog is injured."
        ),
        species="dog",
    )

    assert (
        assistant.prompt_profile
        == "strict_relevance"
    )

    assert (
        client.responses.request[
            "instructions"
        ]
        == STRICT_RELEVANCE_SYSTEM_INSTRUCTIONS
    )