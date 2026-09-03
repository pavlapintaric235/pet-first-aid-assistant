import pytest

from src.pet_first_aid_assistant.assistant import (
    PetFirstAidAssistant,
    SYSTEM_INSTRUCTIONS,
    build_context,
    build_user_input,
    source_summaries,
)


class FakeRetriever:
    def __init__(self):
        self.last_query = None
        self.last_num_results = None
        self.last_species = None

    def search(
        self,
        query,
        num_results=5,
        species=None,
    ):
        self.last_query = query
        self.last_num_results = num_results
        self.last_species = species

        return [
            {
                "id": "doc_1",
                "source_id": "merck_test",
                "publisher": "Merck Veterinary Manual",
                "title": "Emergency Care",
                "url": "https://example.com/merck",
                "species": ["dog", "cat"],
                "topics": ["bleeding"],
                "heading_path": [
                    "Emergency Care",
                    "Bleeding",
                ],
                "content": (
                    "Apply direct pressure to external bleeding."
                ),
                "retrieval_method": "hybrid",
                "retrieval_score": 0.03,
            },
            {
                "id": "doc_2",
                "source_id": "vca_test",
                "publisher": "VCA Animal Hospitals",
                "title": "First Aid for Dogs",
                "url": "https://example.com/vca",
                "species": ["dog"],
                "topics": ["transport"],
                "section_heading": "Transport",
                "content": (
                    "Transport the injured dog carefully."
                ),
                "retrieval_method": "hybrid",
                "retrieval_score": 0.02,
            },
        ]


class FakeResponse:
    output_text = (
        "Seek veterinary care promptly. Apply direct pressure to "
        "external bleeding [S1]."
    )


class FakeResponsesAPI:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeClient:
    def __init__(self):
        self.responses = FakeResponsesAPI()


def test_system_instructions_contain_core_safety_rules():
    lowered = SYSTEM_INSTRUCTIONS.lower()

    assert "do not diagnose" in lowered
    assert "medication doses" in lowered
    assert "inducing vomiting" in lowered
    assert "clinically validated" in lowered
    assert "retrieved source excerpts" in lowered


def test_build_context_labels_sources():
    results = FakeRetriever().search("bleeding")

    context = build_context(results)

    assert "[S1]" in context
    assert "[S2]" in context
    assert "Merck Veterinary Manual" in context
    assert (
        "Section: Emergency Care > Bleeding"
        in context
    )
    assert "Apply direct pressure" in context


def test_build_user_input_contains_question_species_and_context():
    results = FakeRetriever().search("bleeding")

    user_input = build_user_input(
        question="My dog is bleeding heavily.",
        species="dog",
        results=results,
    )

    assert "Pet species: dog" in user_input
    assert "My dog is bleeding heavily." in user_input
    assert "[S1]" in user_input
    assert "Do not infer a diagnosis" in user_input


def test_build_user_input_rejects_invalid_species():
    with pytest.raises(
        ValueError,
        match="species",
    ):
        build_user_input(
            question="Help",
            species="rabbit",
            results=[],
        )


def test_source_summaries_expose_metadata_not_content():
    results = FakeRetriever().search("bleeding")

    sources = source_summaries(results)

    assert sources[0]["label"] == "S1"
    assert sources[0]["source_id"] == "merck_test"
    assert (
        sources[0]["section"]
        == "Emergency Care > Bleeding"
    )
    assert "content" not in sources[0]


def test_assistant_runs_retrieval_then_generation():
    retriever = FakeRetriever()
    client = FakeClient()

    assistant = PetFirstAidAssistant(
        retriever=retriever,
        client=client,
        model="test-model",
        num_sources=2,
    )

    result = assistant.ask(
        question="My dog is bleeding heavily.",
        species="dog",
    )

    assert (
        retriever.last_query
        == "My dog is bleeding heavily."
    )
    assert retriever.last_num_results == 2
    assert retriever.last_species == "dog"

    assert result["model"] == "test-model"
    assert (
        result["retrieval"]["method"]
        == "hybrid_source_diverse"
    )
    assert len(result["sources"]) == 2
    assert "[S1]" in result["answer"]

    request = client.responses.kwargs

    assert request["model"] == "test-model"
    assert (
        request["instructions"]
        == SYSTEM_INSTRUCTIONS
    )
    assert "[S1]" in request["input"]
    assert request["reasoning"] == {
        "effort": "low"
    }


def test_assistant_rejects_blank_question_before_retrieval():
    assistant = PetFirstAidAssistant(
        retriever=FakeRetriever(),
        client=FakeClient(),
        model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="question",
    ):
        assistant.ask("   ")


def test_assistant_rejects_invalid_species():
    assistant = PetFirstAidAssistant(
        retriever=FakeRetriever(),
        client=FakeClient(),
        model="test-model",
    )

    with pytest.raises(
        ValueError,
        match="species",
    ):
        assistant.ask(
            question="Help",
            species="rabbit",
        )