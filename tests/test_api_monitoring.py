from fastapi.testclient import TestClient

from src.pet_first_aid_assistant.api import (
    create_app,
)


class FakeAssistant:
    def ask(
        self,
        question,
        species=None,
    ):
        return {
            "answer": (
                "Apply direct pressure [S1]. "
                "Seek veterinary care."
            ),
            "species": species,
            "model": "fake-model",
            "sources": [
                {
                    "label": "S1",
                    "source_id": "test-source",
                    "publisher": "Test Publisher",
                    "title": "Test Source",
                    "section": "Bleeding",
                    "url": "https://example.com",
                    "retrieval_method": "hybrid",
                    "retrieval_score": 0.5,
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


class FakeMonitoringStore:
    def __init__(self):
        self.interactions = {}
        self.feedback = {}
        self.initialized = False

    @property
    def enabled(self):
        return True

    def initialize(self):
        self.initialized = True

    def record_interaction(
        self,
        interaction_id,
        question,
        species,
        answer,
        model,
        latency_ms,
        sources,
    ):
        self.interactions[
            interaction_id
        ] = {
            "question": question,
            "species": species,
            "answer": answer,
            "model": model,
            "latency_ms": latency_ms,
            "sources": sources,
        }

    def record_feedback(
        self,
        interaction_id,
        rating,
    ):
        if (
            interaction_id
            not in self.interactions
        ):
            return False

        self.feedback[
            interaction_id
        ] = rating

        return True

    def metrics(self):
        feedback_total = len(
            self.feedback
        )

        positive = sum(
            1
            for value
            in self.feedback.values()
            if value == 1
        )

        negative = sum(
            1
            for value
            in self.feedback.values()
            if value == -1
        )

        return {
            "enabled": True,
            "total_requests": len(
                self.interactions
            ),
            "requests_last_24_hours": len(
                self.interactions
            ),
            "average_latency_ms": (
                10.0
                if self.interactions
                else None
            ),
            "feedback_total": (
                feedback_total
            ),
            "feedback_positive": (
                positive
            ),
            "feedback_negative": (
                negative
            ),
            "positive_feedback_rate": (
                positive
                / feedback_total
                if feedback_total
                else None
            ),
        }


class BrokenMonitoringStore(
    FakeMonitoringStore
):
    def record_interaction(
        self,
        interaction_id,
        question,
        species,
        answer,
        model,
        latency_ms,
        sources,
    ):
        raise RuntimeError(
            "Database unavailable."
        )


def make_client(
    monitoring=None,
):
    if monitoring is None:
        monitoring = (
            FakeMonitoringStore()
        )

    app = create_app(
        assistant_factory=(
            FakeAssistant
        ),
        monitoring_factory=(
            lambda: monitoring
        ),
    )

    return (
        app,
        monitoring,
    )


def test_monitoring_is_initialized():
    app, monitoring = (
        make_client()
    )

    with TestClient(
        app
    ):
        assert (
            monitoring.initialized
            is True
        )


def test_health_reports_monitoring_enabled():
    app, _ = make_client()

    with TestClient(
        app
    ) as client:
        response = client.get(
            "/health"
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "monitoring_enabled"
        ]
        is True
    )


def test_ask_returns_interaction_id():
    app, _ = make_client()

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My dog is bleeding."
                ),
                "species": "dog",
            },
        )

    assert (
        response.status_code
        == 200
    )

    interaction_id = (
        response.json()[
            "interaction_id"
        ]
    )

    assert interaction_id


def test_ask_records_interaction():
    app, monitoring = (
        make_client()
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My dog is bleeding."
                ),
                "species": "dog",
            },
        )

    interaction_id = (
        response.json()[
            "interaction_id"
        ]
    )

    recorded = (
        monitoring.interactions[
            interaction_id
        ]
    )

    assert (
        recorded["question"]
        == "My dog is bleeding."
    )

    assert (
        recorded["species"]
        == "dog"
    )

    assert (
        recorded["model"]
        == "fake-model"
    )

    assert (
        recorded["latency_ms"]
        >= 0
    )


def test_feedback_is_recorded():
    app, monitoring = (
        make_client()
    )

    with TestClient(
        app
    ) as client:
        answer_response = (
            client.post(
                "/ask",
                json={
                    "question": (
                        "My dog is bleeding."
                    ),
                    "species": "dog",
                },
            )
        )

        interaction_id = (
            answer_response.json()[
                "interaction_id"
            ]
        )

        feedback_response = (
            client.post(
                "/feedback",
                json={
                    "interaction_id": (
                        interaction_id
                    ),
                    "rating": 1,
                },
            )
        )

    assert (
        feedback_response.status_code
        == 200
    )

    assert (
        feedback_response.json()[
            "accepted"
        ]
        is True
    )

    assert (
        monitoring.feedback[
            interaction_id
        ]
        == 1
    )


def test_feedback_rejects_invalid_rating():
    app, _ = make_client()

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/feedback",
            json={
                "interaction_id": (
                    "00000000-0000-0000-0000-000000000000"
                ),
                "rating": 0,
            },
        )

    assert (
        response.status_code
        == 422
    )


def test_feedback_returns_404_for_unknown_interaction():
    app, _ = make_client()

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/feedback",
            json={
                "interaction_id": (
                    "00000000-0000-0000-0000-000000000000"
                ),
                "rating": 1,
            },
        )

    assert (
        response.status_code
        == 404
    )


def test_metrics_include_requests_and_feedback():
    app, _ = make_client()

    with TestClient(
        app
    ) as client:
        answer_response = (
            client.post(
                "/ask",
                json={
                    "question": (
                        "My cat is bleeding."
                    ),
                    "species": "cat",
                },
            )
        )

        interaction_id = (
            answer_response.json()[
                "interaction_id"
            ]
        )

        client.post(
            "/feedback",
            json={
                "interaction_id": (
                    interaction_id
                ),
                "rating": 1,
            },
        )

        metrics_response = (
            client.get(
                "/metrics"
            )
        )

    assert (
        metrics_response.status_code
        == 200
    )

    metrics = (
        metrics_response.json()
    )

    assert (
        metrics[
            "total_requests"
        ]
        == 1
    )

    assert (
        metrics[
            "feedback_total"
        ]
        == 1
    )

    assert (
        metrics[
            "feedback_positive"
        ]
        == 1
    )

    assert (
        metrics[
            "positive_feedback_rate"
        ]
        == 1.0
    )


def test_monitoring_failure_does_not_block_answer():
    monitoring = (
        BrokenMonitoringStore()
    )

    app, _ = make_client(
        monitoring=monitoring
    )

    with TestClient(
        app
    ) as client:
        response = client.post(
            "/ask",
            json={
                "question": (
                    "My dog is bleeding."
                ),
                "species": "dog",
            },
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        response.json()[
            "answer"
        ]
    )