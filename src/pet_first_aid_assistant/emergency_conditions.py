from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EmergencyCondition:
    """One predefined emergency topic exposed to the frontend."""

    id: str
    title: str
    short_description: str
    starter_question: str
    urgency: str
    supported_species: tuple[str, ...]


EMERGENCY_CONDITIONS: tuple[
    EmergencyCondition,
    ...,
] = (
    EmergencyCondition(
        id="heavy_bleeding",
        title="Heavy bleeding",
        short_description=(
            "Active or severe bleeding from a wound."
        ),
        starter_question=(
            "My pet is bleeding heavily from a wound. "
            "What should I do right now?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="breathing_difficulty",
        title="Breathing difficulty",
        short_description=(
            "Serious difficulty breathing or abnormal breathing."
        ),
        starter_question=(
            "My pet is having serious trouble breathing. "
            "What should I do right now?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="choking",
        title="Possible choking",
        short_description=(
            "Your pet appears to be choking or may have "
            "something blocking the airway."
        ),
        starter_question=(
            "My pet looks like they may be choking and "
            "cannot breathe normally. What should I do?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="unconscious",
        title="Unconscious or unresponsive",
        short_description=(
            "Your pet is unconscious, collapsed, or not responding."
        ),
        starter_question=(
            "My pet is unconscious and not responding. "
            "What should I do right now?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="possible_poisoning",
        title="Possible poisoning",
        short_description=(
            "Your pet may have swallowed or contacted "
            "something harmful or toxic."
        ),
        starter_question=(
            "My pet may have been exposed to something poisonous. "
            "What should I do right now?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="burn",
        title="Burn",
        short_description=(
            "A burn caused by heat or another damaging source."
        ),
        starter_question=(
            "My pet has been burned. "
            "What first aid should I give?"
        ),
        urgency="urgent",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="heat_emergency",
        title="Heat emergency",
        short_description=(
            "Your pet became extremely hot, weak, "
            "or distressed after heat exposure."
        ),
        starter_question=(
            "My pet became extremely hot, weak, and distressed "
            "after being in the heat. What should I do?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="cold_emergency",
        title="Cold emergency",
        short_description=(
            "Your pet is extremely cold, weak, "
            "or poorly responsive after cold exposure."
        ),
        starter_question=(
            "My pet is extremely cold and weak after being "
            "outside in the cold. What should I do?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="embedded_object",
        title="Object embedded in wound",
        short_description=(
            "An object is stuck deeply in a wound "
            "or penetrating the body."
        ),
        starter_question=(
            "There is an object stuck deeply in my pet's wound. "
            "What should I do?"
        ),
        urgency="emergency",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
    EmergencyCondition(
        id="injury_transport",
        title="Safe transport after injury",
        short_description=(
            "You need to move or transport an injured pet safely."
        ),
        starter_question=(
            "My pet is injured. How can I move and transport "
            "them safely to a veterinarian?"
        ),
        urgency="urgent",
        supported_species=(
            "dog",
            "cat",
        ),
    ),
)


def list_emergency_conditions() -> list[dict[str, object]]:
    """Return the complete frontend-safe emergency catalog."""

    return [
        asdict(
            condition
        )
        for condition in EMERGENCY_CONDITIONS
    ]


def get_emergency_condition(
    condition_id: str,
) -> EmergencyCondition | None:
    """Return one emergency condition by its stable identifier."""

    normalized_id = (
        condition_id.strip()
        if isinstance(
            condition_id,
            str,
        )
        else ""
    )

    if not normalized_id:
        return None

    for condition in EMERGENCY_CONDITIONS:
        if condition.id == normalized_id:
            return condition

    return None