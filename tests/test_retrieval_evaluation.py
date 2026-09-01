import json

import pytest

from src.evaluation.retrieval_evaluation import (
    compute_relevance,
    evaluate_search,
    hit_rate,
    load_ground_truth,
    mean_reciprocal_rank,
)


class FakeSearch:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(
        self,
        query,
        num_results=5,
        species=None,
    ):
        return self.results_by_query[query][
            :num_results
        ]


def make_result(
    document_id,
    source_id,
):
    return {
        "id": document_id,
        "source_id": source_id,
    }


def make_ground_truth_record(
    record_id="gt_001",
    question="Emergency question",
    species="dog",
    relevant_source_ids=None,
):
    if relevant_source_ids is None:
        relevant_source_ids = ["relevant_source"]

    return {
        "id": record_id,
        "question": question,
        "species": species,
        "category": "emergency",
        "relevant_source_ids": relevant_source_ids,
    }


def test_compute_relevance():
    record = make_ground_truth_record()

    results = [
        make_result(
            "document_1",
            "irrelevant_source",
        ),
        make_result(
            "document_2",
            "relevant_source",
        ),
    ]

    relevance = compute_relevance(
        record,
        results,
    )

    assert relevance == [0, 1]


def test_hit_rate():
    relevance_total = [
        [0, 1, 0],
        [0, 0, 0],
        [1, 0, 0],
    ]

    result = hit_rate(relevance_total)

    assert result == pytest.approx(2 / 3)


def test_mean_reciprocal_rank():
    relevance_total = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]

    result = mean_reciprocal_rank(
        relevance_total
    )

    expected = (
        1.0 + 0.5 + (1.0 / 3.0)
    ) / 3.0

    assert result == pytest.approx(expected)


def test_mean_reciprocal_rank_includes_misses():
    relevance_total = [
        [1, 0],
        [0, 0],
    ]

    result = mean_reciprocal_rank(
        relevance_total
    )

    assert result == pytest.approx(0.5)


def test_evaluate_search():
    first_question = "First emergency"
    second_question = "Second emergency"

    ground_truth = [
        make_ground_truth_record(
            record_id="gt_001",
            question=first_question,
            relevant_source_ids=["source_a"],
        ),
        make_ground_truth_record(
            record_id="gt_002",
            question=second_question,
            relevant_source_ids=["source_b"],
        ),
    ]

    search_engine = FakeSearch(
        {
            first_question: [
                make_result("doc_1", "source_a"),
                make_result("doc_2", "source_x"),
            ],
            second_question: [
                make_result("doc_3", "source_x"),
                make_result("doc_4", "source_b"),
            ],
        }
    )

    result = evaluate_search(
        ground_truth=ground_truth,
        search_engine=search_engine,
        num_results=2,
    )

    assert result["hit_rate"] == 1.0
    assert result["mrr"] == pytest.approx(0.75)
    assert result["num_questions"] == 2
    assert (
        result["questions"][1]["first_relevant_rank"]
        == 2
    )


def test_load_ground_truth(tmp_path):
    ground_truth_path = (
        tmp_path / "ground_truth.json"
    )

    records = [
        make_ground_truth_record()
    ]

    ground_truth_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    loaded = load_ground_truth(
        ground_truth_path
    )

    assert loaded == records


def test_load_ground_truth_rejects_duplicate_ids(
    tmp_path,
):
    ground_truth_path = (
        tmp_path / "ground_truth.json"
    )

    records = [
        make_ground_truth_record(
            question="First question",
        ),
        make_ground_truth_record(
            question="Second question",
        ),
    ]

    ground_truth_path.write_text(
        json.dumps(records),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        load_ground_truth(
            ground_truth_path
        )