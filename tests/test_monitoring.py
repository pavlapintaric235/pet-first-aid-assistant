from src.pet_first_aid_assistant.monitoring import (
    NullMonitoringStore,
    PostgresMonitoringStore,
    build_monitoring_store,
)


def test_null_monitoring_is_disabled():
    store = NullMonitoringStore()

    assert store.enabled is False


def test_null_monitoring_metrics_are_empty():
    store = NullMonitoringStore()

    metrics = store.metrics()

    assert (
        metrics["enabled"]
        is False
    )

    assert (
        metrics["total_requests"]
        == 0
    )

    assert (
        metrics["feedback_total"]
        == 0
    )


def test_null_feedback_returns_false():
    store = NullMonitoringStore()

    assert (
        store.record_feedback(
            interaction_id=(
                "00000000-0000-0000-0000-000000000000"
            ),
            rating=1,
        )
        is False
    )


def test_postgres_monitoring_requires_url():
    try:
        PostgresMonitoringStore(
            database_url=""
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected ValueError"
        )


def test_build_monitoring_store_without_database(
    monkeypatch,
):
    monkeypatch.delenv(
        "DATABASE_URL",
        raising=False,
    )

    store = (
        build_monitoring_store()
    )

    assert isinstance(
        store,
        NullMonitoringStore,
    )


def test_build_monitoring_store_with_database(
    monkeypatch,
):
    monkeypatch.setenv(
        "DATABASE_URL",
        (
            "postgresql://user:"
            "password@localhost/test"
        ),
    )

    store = (
        build_monitoring_store()
    )

    assert isinstance(
        store,
        PostgresMonitoringStore,
    )


def test_postgres_rating_validation():
    store = (
        PostgresMonitoringStore(
            database_url=(
                "postgresql://user:"
                "password@localhost/test"
            )
        )
    )

    try:
        store.record_feedback(
            interaction_id=(
                "00000000-0000-0000-0000-000000000000"
            ),
            rating=0,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected ValueError"
        )