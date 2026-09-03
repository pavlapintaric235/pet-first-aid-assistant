from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.evaluation.generation_safety import (
    evaluate_generation_result,
    load_generation_cases,
)
from src.pet_first_aid_assistant.assistant import (
    PetFirstAidAssistant,
)


DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "generation_safety_results.json"
)


def parse_arguments() -> argparse.Namespace:
    """Parse generation safety evaluation arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run Pet First Aid Assistant answers through "
            "deterministic hard safety checks."
        )
    )

    parser.add_argument(
        "--case-id",
        default=None,
        help=(
            "Optionally run only one evaluation case, "
            "for example gen_002."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "Path used to save evaluation results."
        ),
    )

    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Re-evaluate answers already stored in the output "
            "file instead of making new generation API requests."
        ),
    )

    return parser.parse_args()


def load_existing_results(
    path: Path,
) -> dict[str, dict[str, Any]]:
    """Load previously generated answers keyed by case id."""

    if not path.exists():
        raise FileNotFoundError(
            "Existing generation safety results were not found: "
            f"{path}"
        )

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    results = payload.get(
        "results",
        [],
    )

    if not isinstance(results, list):
        raise ValueError(
            "Existing generation results must contain a results list."
        )

    return {
        result["case_id"]: result
        for result in results
        if isinstance(result, dict)
        and result.get("case_id")
    }


def existing_result_to_assistant_result(
    existing: dict[str, Any],
) -> dict[str, Any]:
    """Convert a saved evaluation record back to assistant-result shape."""

    return {
        "answer": existing.get(
            "answer",
            "",
        ),
        "sources": existing.get(
            "sources",
            [],
        ),
    }


def print_evaluation(
    evaluation: dict[str, Any],
) -> None:
    """Print one evaluated answer."""

    status = (
        "PASS"
        if evaluation[
            "hard_pass"
        ]
        else "FAIL"
    )

    print(
        f"Hard safety: {status}"
    )

    for check_name, passed in (
        evaluation["checks"].items()
    ):
        marker = (
            "PASS"
            if passed
            else "FAIL"
        )

        print(
            f"  {marker:<4} "
            f"{check_name}"
        )

    print()
    print(
        "Answer:"
    )

    print(
        evaluation["answer"]
    )

    print()
    print(
        "Expected manual-review behavior:"
    )

    print(
        evaluation[
            "expected_behavior"
        ]
    )

    print()
    print(
        "-" * 72
    )


def main() -> None:
    """Generate or reuse answers and evaluate hard safety rules."""

    arguments = parse_arguments()

    cases = load_generation_cases()

    if arguments.case_id:
        cases = [
            case
            for case in cases
            if case["id"]
            == arguments.case_id
        ]

        if not cases:
            raise ValueError(
                f"Unknown case id: "
                f"{arguments.case_id}"
            )

    existing_results: dict[
        str,
        dict[str, Any],
    ] = {}

    assistant: PetFirstAidAssistant | None = None

    if arguments.reuse_existing:
        existing_results = (
            load_existing_results(
                arguments.output
            )
        )

        missing_case_ids = [
            case["id"]
            for case in cases
            if case["id"]
            not in existing_results
        ]

        if missing_case_ids:
            raise ValueError(
                "Existing results are missing cases: "
                + ", ".join(
                    missing_case_ids
                )
            )

    else:
        assistant = (
            PetFirstAidAssistant()
        )

    evaluations: list[
        dict[str, Any]
    ] = []

    print()
    print(
        "Generation safety evaluation"
    )
    print(
        "----------------------------"
    )
    print(
        f"Cases: {len(cases)}"
    )

    if arguments.reuse_existing:
        print(
            "Mode: re-evaluate existing saved answers"
        )
        print(
            "No generation API requests will be made."
        )

    else:
        print(
            "Mode: generate new answers"
        )
        print(
            "These requests use the real generation model."
        )

    print()

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index}/{len(cases)}] "
            f"{case['id']} — "
            f"{case['category']}"
        )

        if arguments.reuse_existing:
            result = (
                existing_result_to_assistant_result(
                    existing_results[
                        case["id"]
                    ]
                )
            )

        else:
            assert assistant is not None

            result = assistant.ask(
                question=case["question"],
                species=case.get(
                    "species"
                ),
            )

        evaluation = (
            evaluate_generation_result(
                case=case,
                result=result,
            )
        )

        evaluations.append(
            evaluation
        )

        print_evaluation(
            evaluation
        )

    passed = sum(
        1
        for evaluation in evaluations
        if evaluation["hard_pass"]
    )

    failed = (
        len(evaluations)
        - passed
    )

    payload = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "evaluation_mode": (
            "reused_existing_answers"
            if arguments.reuse_existing
            else "new_generation"
        ),
        "summary": {
            "cases": len(
                evaluations
            ),
            "hard_passed": passed,
            "hard_failed": failed,
            "hard_pass_rate": (
                passed
                / len(evaluations)
                if evaluations
                else 0.0
            ),
            "manual_review_required": True,
        },
        "results": evaluations,
    }

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Summary"
    )
    print(
        "-------"
    )

    print(
        f"Cases:       "
        f"{len(evaluations)}"
    )

    print(
        f"Hard passed: "
        f"{passed}"
    )

    print(
        f"Hard failed: "
        f"{failed}"
    )

    print(
        "Hard pass rate: "
        f"{payload['summary']['hard_pass_rate']:.4f}"
    )

    print()
    print(
        "Important: a hard PASS does not mean the "
        "medical answer has been clinically validated."
    )

    print(
        "Every generated answer still requires "
        "manual grounding and safety review."
    )

    print()
    print(
        f"Saved results to: "
        f"{arguments.output}"
    )


if __name__ == "__main__":
    main()