from src.pet_first_aid_assistant.emergency_conditions import (
    EMERGENCY_CONDITIONS,
    get_emergency_condition,
    list_emergency_conditions,
)


def test_emergency_condition_ids_are_unique():
    condition_ids = [
        condition.id
        for condition in EMERGENCY_CONDITIONS
    ]

    assert len(
        condition_ids
    ) == len(
        set(
            condition_ids
        )
    )


def test_catalog_contains_expected_core_conditions():
    condition_ids = {
        condition.id
        for condition in EMERGENCY_CONDITIONS
    }

    assert {
        "heavy_bleeding",
        "breathing_difficulty",
        "choking",
        "unconscious",
        "possible_poisoning",
        "burn",
        "heat_emergency",
        "cold_emergency",
        "embedded_object",
        "injury_transport",
    }.issubset(
        condition_ids
    )


def test_all_conditions_support_dogs_and_cats():
    for condition in EMERGENCY_CONDITIONS:
        assert set(
            condition.supported_species
        ) == {
            "dog",
            "cat",
        }


def test_all_conditions_have_safe_starter_questions():
    for condition in EMERGENCY_CONDITIONS:
        assert condition.starter_question.strip()
        assert "?" in condition.starter_question

        lowered = (
            condition.starter_question.lower()
        )

        assert (
            "hydrogen peroxide"
            not in lowered
        )

        assert (
            "how much medication"
            not in lowered
        )


def test_catalog_does_not_include_cpr_preset_yet():
    condition_ids = {
        condition.id
        for condition in EMERGENCY_CONDITIONS
    }

    assert "cpr" not in condition_ids


def test_list_emergency_conditions_returns_serializable_records():
    conditions = (
        list_emergency_conditions()
    )

    assert conditions

    first = conditions[0]

    assert "id" in first
    assert "title" in first
    assert "starter_question" in first
    assert "supported_species" in first


def test_get_emergency_condition_returns_match():
    condition = (
        get_emergency_condition(
            "heavy_bleeding"
        )
    )

    assert condition is not None
    assert (
        condition.title
        == "Heavy bleeding"
    )


def test_get_emergency_condition_returns_none_for_unknown_id():
    assert (
        get_emergency_condition(
            "not_a_real_condition"
        )
        is None
    )


def test_condition_titles_do_not_make_diagnostic_claims():
    forbidden_titles = {
        "heatstroke",
        "hypothermia",
        "poisoned",
    }

    titles = {
        condition.title.lower()
        for condition in EMERGENCY_CONDITIONS
    }

    assert titles.isdisjoint(
        forbidden_titles
    )