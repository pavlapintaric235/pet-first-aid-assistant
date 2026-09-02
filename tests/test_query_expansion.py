import pytest

from src.retrieval.query_expansion import (
    QueryExpansionSearch,
    expand_query,
    expansion_terms,
    matched_expansions,
)


class RecordingSearch:
    def __init__(self):
        self.query = None
        self.num_results = None
        self.species = None

    def search(
        self,
        query,
        num_results=5,
        species=None,
    ):
        self.query = query
        self.num_results = num_results
        self.species = species

        return [
            {
                "id": "doc_1",
                "source_id": "source_1",
                "content": "Example content.",
            }
        ]


def test_expand_query_preserves_original_wording():
    query = (
        "My dog cannot breathe normally."
    )

    expanded = expand_query(
        query
    )

    assert expanded.startswith(query)
    assert "breathing difficulty" in expanded
    assert "respiratory distress" in expanded


def test_poisoning_expansion_adds_retrieval_terms():
    terms = expansion_terms(
        "My cat swallowed something poisonous."
    )

    assert "poisoning" in terms
    assert "toxin ingestion" in terms


def test_bleeding_expansion_is_deduplicated():
    terms = expansion_terms(
        "My dog has severe bleeding."
    )

    assert "severe bleeding" not in terms
    assert terms.count("hemorrhage") == 1


def test_safe_transport_expansion():
    terms = expansion_terms(
        "How do I safely move an injured dog into my car?"
    )

    assert "first aid transport" in terms
    assert "safe handling" in terms


def test_query_without_matching_rule_is_unchanged():
    query = "My dog seems uncomfortable today."

    assert expand_query(query) == query
    assert expansion_terms(query) == []


def test_matched_expansions_reports_rule_names():
    matches = matched_expansions(
        "My cat has pale gums and cold paws."
    )

    rule_names = {
        match["rule"]
        for match in matches
    }

    assert "shock_signs" in rule_names


def test_expansion_does_not_add_medication_or_dosing_terms():
    expanded = expand_query(
        "My dog swallowed something toxic."
    ).lower()

    forbidden_terms = {
        "dose",
        "dosage",
        "hydrogen peroxide",
        "medication",
        "medicine",
        "induce vomiting",
    }

    assert all(
        term not in expanded
        for term in forbidden_terms
    )


def test_query_expansion_wrapper_passes_expanded_query():
    underlying = RecordingSearch()

    search_engine = QueryExpansionSearch(
        search_engine=underlying
    )

    results = search_engine.search(
        query="My dog is having repeated seizures.",
        num_results=3,
        species="dog",
    )

    assert "neurologic emergency" in underlying.query
    assert underlying.num_results == 3
    assert underlying.species == "dog"

    assert results[0]["query_expanded"] is True
    assert results[0]["original_query"] == (
        "My dog is having repeated seizures."
    )
    assert "seizures" in results[0][
        "query_expansion_rules"
    ]


def test_query_expansion_wrapper_marks_unchanged_query():
    underlying = RecordingSearch()

    search_engine = QueryExpansionSearch(
        search_engine=underlying
    )

    results = search_engine.search(
        query="My dog seems uncomfortable today.",
    )

    assert underlying.query == (
        "My dog seems uncomfortable today."
    )
    assert results[0]["query_expanded"] is False
    assert results[0]["query_expansion_terms"] == []


def test_blank_query_is_rejected():
    with pytest.raises(
        ValueError,
        match="searchable text",
    ):
        expand_query("   ")


def test_non_string_query_is_rejected():
    with pytest.raises(
        TypeError,
        match="string",
    ):
        expand_query(None)