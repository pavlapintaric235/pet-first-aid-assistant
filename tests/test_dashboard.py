from fastapi.testclient import (
    TestClient,
)

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
                "Seek veterinary care [S1]."
            ),
            "species": species,
            "model": "fake-model",
            "sources": [],
            "retrieval": {
                "method": (
                    "hybrid_source_diverse"
                ),
                "num_sources": 0,
                "max_chunks_per_source": 1,
            },
        }


class FakeMonitoringStore:
    enabled = True

    def initialize(self):
        return None

    def record_interaction(
        self,
        **kwargs,
    ):
        return None

    def record_feedback(
        self,
        interaction_id,
        rating,
    ):
        return True

    def metrics(self):
        return {
            "enabled": True,
            "total_requests": 10,
            "requests_last_24_hours": 3,
            "average_latency_ms": 500.0,
            "feedback_total": 4,
            "feedback_positive": 3,
            "feedback_negative": 1,
            "positive_feedback_rate": 0.75,
        }

    def dashboard(self):
        return {
            "enabled": True,
            "requests_by_day": [
                {
                    "label": "2026-09-01",
                    "value": 2,
                },
                {
                    "label": "2026-09-02",
                    "value": 3,
                },
            ],
            "latency_by_day": [
                {
                    "label": "2026-09-01",
                    "value": 450.0,
                },
                {
                    "label": "2026-09-02",
                    "value": 550.0,
                },
            ],
            "species_breakdown": [
                {
                    "label": "dog",
                    "value": 6,
                },
                {
                    "label": "cat",
                    "value": 4,
                },
            ],
            "feedback_breakdown": [
                {
                    "label": "Positive",
                    "value": 3,
                },
                {
                    "label": "Negative",
                    "value": 1,
                },
            ],
            "top_sources": [
                {
                    "label": (
                        "Merck Veterinary Manual"
                    ),
                    "value": 8,
                }
            ],
        }


def make_client():
    app = create_app(
        assistant_factory=(
            FakeAssistant
        ),
        monitoring_factory=(
            FakeMonitoringStore
        ),
    )

    return TestClient(
        app
    )


def test_dashboard_data_endpoint():
    with make_client() as client:
        response = client.get(
            "/dashboard-data"
        )

    assert (
        response.status_code
        == 200
    )

    body = response.json()

    assert (
        body["enabled"]
        is True
    )

    assert len(
        body[
            "requests_by_day"
        ]
    ) == 2

    assert len(
        body[
            "latency_by_day"
        ]
    ) == 2

    assert len(
        body[
            "species_breakdown"
        ]
    ) == 2

    assert len(
        body[
            "feedback_breakdown"
        ]
    ) == 2

    assert len(
        body[
            "top_sources"
        ]
    ) == 1


def test_dashboard_contains_five_datasets():
    with make_client() as client:
        body = client.get(
            "/dashboard-data"
        ).json()

    chart_datasets = {
        "requests_by_day",
        "latency_by_day",
        "species_breakdown",
        "feedback_breakdown",
        "top_sources",
    }

    assert chart_datasets.issubset(
        body.keys()
    )


def test_dashboard_source_counts_are_returned():
    with make_client() as client:
        body = client.get(
            "/dashboard-data"
        ).json()

    top_source = (
        body[
            "top_sources"
        ][0]
    )

    assert (
        top_source["label"]
        == "Merck Veterinary Manual"
    )

    assert (
        top_source["value"]
        == 8
    )


def test_dashboard_html_is_served():
    with make_client() as client:
        response = client.get(
            "/static/dashboard.html"
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        "Monitoring Dashboard"
        in response.text
    )

    assert (
        "requests-chart"
        in response.text
    )

    assert (
        "latency-chart"
        in response.text
    )

    assert (
        "species-chart"
        in response.text
    )

    assert (
        "feedback-chart"
        in response.text
    )

    assert (
        "sources-chart"
        in response.text
    )


def test_dashboard_javascript_is_served():
    with make_client() as client:
        response = client.get(
            "/static/dashboard.js"
        )

    assert (
        response.status_code
        == 200
    )

    assert (
        'fetch("/metrics")'
        in response.text
    )

    assert (
        'fetch("/dashboard-data")'
        in response.text
    )

    assert (
        "drawLineChart"
        in response.text
    )

    assert (
        "drawFeedbackChart"
        in response.text
    )

    assert (
        "drawHorizontalBarChart"
        in response.text
    )