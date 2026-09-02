import numpy as np
import pytest

from src.retrieval.embedder import (
    normalize_rows,
    pool_hidden_states,
    prepare_query_text,
)


def test_prepare_query_text_without_prefix():
    result = prepare_query_text(
        text="My dog is bleeding.",
    )

    assert result == (
        "My dog is bleeding."
    )


def test_prepare_query_text_with_prefix():
    result = prepare_query_text(
        text="My dog is bleeding.",
        query_prefix=(
            "Represent this sentence for "
            "searching relevant passages: "
        ),
    )

    assert result == (
        "Represent this sentence for "
        "searching relevant passages: "
        "My dog is bleeding."
    )


def test_mean_pooling_ignores_padding():
    hidden_states = np.array(
        [
            [
                [1.0, 1.0],
                [3.0, 3.0],
                [100.0, 100.0],
            ],
        ],
        dtype=np.float32,
    )

    attention_mask = np.array(
        [
            [
                1,
                1,
                0,
            ],
        ],
        dtype=np.int64,
    )

    pooled = pool_hidden_states(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        pooling="mean",
    )

    expected = np.array(
        [
            [
                2.0,
                2.0,
            ],
        ],
        dtype=np.float32,
    )

    assert np.allclose(
        pooled,
        expected,
    )


def test_cls_pooling_uses_first_token():
    hidden_states = np.array(
        [
            [
                [5.0, 7.0],
                [20.0, 30.0],
                [40.0, 50.0],
            ],
        ],
        dtype=np.float32,
    )

    attention_mask = np.array(
        [
            [
                1,
                1,
                1,
            ],
        ],
        dtype=np.int64,
    )

    pooled = pool_hidden_states(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        pooling="cls",
    )

    expected = np.array(
        [
            [
                5.0,
                7.0,
            ],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(
        pooled,
        expected,
    )


def test_pooling_rejects_unknown_mode():
    hidden_states = np.ones(
        (
            1,
            2,
            3,
        ),
        dtype=np.float32,
    )

    attention_mask = np.ones(
        (
            1,
            2,
        ),
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="mean.*cls",
    ):
        pool_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooling="invalid",
        )


def test_normalize_rows_produces_unit_vectors():
    matrix = np.array(
        [
            [
                3.0,
                4.0,
            ],
            [
                5.0,
                12.0,
            ],
        ],
        dtype=np.float32,
    )

    normalized = normalize_rows(
        matrix
    )

    norms = np.linalg.norm(
        normalized,
        axis=1,
    )

    assert np.allclose(
        norms,
        1.0,
    )


def test_normalize_rows_rejects_zero_vector():
    matrix = np.array(
        [
            [
                0.0,
                0.0,
            ],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="non-zero L2 norm",
    ):
        normalize_rows(
            matrix
        )