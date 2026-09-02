from fastapi.testclient import TestClient

from src.pet_first_aid_assistant.api import (
    create_app,
)


class FakeAssistant:
    def __init__(self):
        self.question = None
        self.species = None

    def ask(
        self,
        question,
        species=None,
    ):
        self.question = question
        self.species = species

        return {
            "answer": (
                "Apply direct pressure to external "
                "bleeding [S1]. Seek veterinary care."
            ),
            "species": species,
            "model": "fake-model",
            "sources": [
                {
                    "label": "S1",
                    "source_id": "merck_test",
                    "publisher": (
                        "Merck Veterinary Manual"
                    ),
                    "title": (
                        "What to Do in a "
                        "Dog or Cat Emergency"
                    ),
                    "section": "Bleeding",
                    "url": (
                        "https://example.com/merck"
                    ),
                    "retrieval_method": "hybrid",
                    "retrieval_score": 0.04,
                }
            ],
            "retrieval": {
                "method": (
                    "hybrid_source_diverse"
                ),
                "num_sources": 1,
                "max_chunks_per_source": 1,
            },
        }


class BrokenAssistant:
    def ask(
        self,
        question,
        species=None,
    ):
        raise RuntimeError(
            "Simulated provider failure."
        )


def test_health_endpoint():
    fake_assistant = FakeAssistant()

    app = create_app(
        assistant_factory=lambda: fake_assistant
    )

    with TestClient(app) as client:
        response = client.get(
            "/health"
        )

    assert response.status_code == 200

    assert response.json() == {
        "status": "ok",
        "service": (
            "pet-first-aid-assistant"
        ),
    }


def test_ask_endpoint_returns_grounded_response():
    fake_assistant = FakeAssistant()

    app = create_app(
        assistant_factory=lambda: fake_assistant
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My dog is bleeding heavily."
                ),
                "species": "dog",
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert "[S1]" in body["answer"]

    assert body["species"] == "dog"

    assert body["model"] == (
        "fake-model"
    )

    assert (
        body["retrieval"]["method"]
        == "hybrid_source_diverse"
    )

    assert len(
        body["sources"]
    ) == 1

    assert (
        body["sources"][0]["publisher"]
        == "Merck Veterinary Manual"
    )

    assert (
        fake_assistant.question
        == "My dog is bleeding heavily."
    )

    assert (
        fake_assistant.species
        == "dog"
    )


def test_ask_endpoint_accepts_missing_species():
    fake_assistant = FakeAssistant()

    app = create_app(
        assistant_factory=lambda: fake_assistant
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My pet is bleeding heavily."
                ),
            },
        )

    assert response.status_code == 200

    assert (
        response.json()["species"]
        is None
    )

    assert (
        fake_assistant.species
        is None
    )


def test_ask_endpoint_rejects_invalid_species():
    fake_assistant = FakeAssistant()

    app = create_app(
        assistant_factory=lambda: fake_assistant
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My rabbit is bleeding."
                ),
                "species": "rabbit",
            },
        )

    assert response.status_code == 422


def test_ask_endpoint_rejects_blank_question():
    fake_assistant = FakeAssistant()

    app = create_app(
        assistant_factory=lambda: fake_assistant
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "   ",
                "species": "dog",
            },
        )

    assert response.status_code == 422


def test_ask_endpoint_rejects_too_short_question():
    fake_assistant = FakeAssistant()

    app = create_app(
        assistant_factory=lambda: fake_assistant
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": "hi",
                "species": "dog",
            },
        )

    assert response.status_code == 422


def test_provider_failure_returns_safe_error():
    app = create_app(
        assistant_factory=BrokenAssistant
    )

    with TestClient(app) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My dog is bleeding heavily."
                ),
                "species": "dog",
            },
        )

    assert response.status_code == 503

    body = response.json()

    assert "temporarily unavailable" in (
        body["detail"]
    )

    assert "veterinarian" in (
        body["detail"]
    )

    assert (
        "Simulated provider failure"
        not in body["detail"]
    )