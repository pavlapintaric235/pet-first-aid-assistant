from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.pet_first_aid_assistant.assistant import (
    PetFirstAidAssistant,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Ask the Pet First Aid Assistant "
            "a grounded question."
        )
    )

    parser.add_argument(
        "--question",
        required=True,
        help=(
            "Dog or cat first-aid question "
            "or symptom description."
        ),
    )

    parser.add_argument(
        "--species",
        choices=[
            "dog",
            "cat",
        ],
        default=None,
        help=(
            "Optionally restrict retrieval "
            "to one species."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run one end-to-end RAG request from the terminal."""

    arguments = parse_arguments()

    assistant = PetFirstAidAssistant()

    result = assistant.ask(
        question=arguments.question,
        species=arguments.species,
    )

    print()
    print("Pet First Aid Assistant")
    print("-----------------------")
    print(result["answer"])

    print()
    print("Sources")
    print("-------")

    for source in result["sources"]:
        print(
            f"[{source['label']}] "
            f"{source['publisher']} — "
            f"{source['title']}"
        )

        if source["section"]:
            print(
                f"    Section: "
                f"{source['section']}"
            )

        print(
            f"    {source['url']}"
        )


if __name__ == "__main__":
    main()