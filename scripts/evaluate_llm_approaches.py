from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

if str(
    PROJECT_ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(
            PROJECT_ROOT
        ),
    )


from src.evaluation.generation_safety import (
    evaluate_generation_result,
    load_generation_cases,
)
from src.pet_first_aid_assistant.assistant import (
    PetFirstAidAssistant,
    build_production_retriever,
)


OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "llm_approach_results.json"
)


PROMPT_PROFILES = (
    "baseline",
    "strict_relevance",
)


RELEVANCE_FORBIDDEN_TERMS: dict[
    str,
    tuple[str, ...],
] = {
    "gen_003": (
        "100–120",
        "100-120",
        "30:2",
        "chest compressions",
        "rescue breathing",
    ),
    "gen_006": (
        "smoke exposure",
        "car windows",
    ),
    "gen_012": (
        "cold exposure",
        "water bottles",
        "frostbite",
        "apply snow",
        "hypothermia",
    ),
}


def normalize_text(
    value: str,
) -> str:
    """Normalize text for deterministic relevance checks."""

    return (
        value.lower()
        .replace(
            "–",
            "-",
        )
        .replace(
            "—",
            "-",
        )
    )


def evaluate_relevance(
    case_id: str,
    answer: str,
) -> dict[str, Any]:
    """
    Check for known irrelevant-answer patterns.

    These rules target concrete relevance failures observed during
    manual review of the baseline generation evaluation.
    """

    normalized_answer = (
        normalize_text(
            answer
        )
    )

    forbidden_terms = (
        RELEVANCE_FORBIDDEN_TERMS.get(
            case_id,
            (),
        )
    )

    found_terms: list[str] = []

    for term in forbidden_terms:
        normalized_term = (
            normalize_text(
                term
            )
        )

        if (
            normalized_term
            in normalized_answer
        ):
            found_terms.append(
                term
            )

    return {
        "pass": (
            len(
                found_terms
            )
            == 0
        ),
        "forbidden_terms": list(
            forbidden_terms
        ),
        "found_terms": (
            found_terms
        ),
    }


def word_count(
    answer: str,
) -> int:
    """Return a simple answer word count."""

    return len(
        answer.split()
    )


def run_profile(
    profile: str,
    cases: list[
        dict[str, Any]
    ],
    retriever,
    client,
) -> dict[str, Any]:
    """Run one prompt profile over the complete evaluation set."""

    assistant = (
        PetFirstAidAssistant(
            retriever=retriever,
            client=client,
            prompt_profile=profile,
        )
    )

    results: list[
        dict[str, Any]
    ] = []

    print()
    print(
        f"Prompt profile: {profile}"
    )
    print(
        "-" * (
            len(
                profile
            )
            + 16
        )
    )

    for index, case in enumerate(
        cases,
        start=1,
    ):
        print(
            f"[{index}/{len(cases)}] "
            f"{case['id']} — "
            f"{case['category']}"
        )

        generated = (
            assistant.ask(
                question=(
                    case["question"]
                ),
                species=(
                    case.get(
                        "species"
                    )
                ),
            )
        )

        safety = (
            evaluate_generation_result(
                case=case,
                result=generated,
            )
        )

        relevance = (
            evaluate_relevance(
                case_id=(
                    case["id"]
                ),
                answer=(
                    generated[
                        "answer"
                    ]
                ),
            )
        )

        combined_pass = (
            safety["hard_pass"]
            and relevance["pass"]
        )

        result = {
            "case_id": (
                case["id"]
            ),
            "category": (
                case["category"]
            ),
            "question": (
                case["question"]
            ),
            "species": (
                case.get(
                    "species"
                )
            ),
            "answer": (
                generated["answer"]
            ),
            "sources": (
                generated["sources"]
            ),
            "hard_safety_pass": (
                safety[
                    "hard_pass"
                ]
            ),
            "hard_safety_checks": (
                safety["checks"]
            ),
            "relevance_pass": (
                relevance["pass"]
            ),
            "relevance_details": (
                relevance
            ),
            "combined_pass": (
                combined_pass
            ),
            "answer_words": (
                word_count(
                    generated[
                        "answer"
                    ]
                )
            ),
        }

        results.append(
            result
        )

        print(
            "  Hard safety: "
            + (
                "PASS"
                if result[
                    "hard_safety_pass"
                ]
                else "FAIL"
            )
        )

        print(
            "  Relevance:   "
            + (
                "PASS"
                if result[
                    "relevance_pass"
                ]
                else "FAIL"
            )
        )

        print(
            "  Combined:    "
            + (
                "PASS"
                if result[
                    "combined_pass"
                ]
                else "FAIL"
            )
        )

        if (
            result[
                "relevance_details"
            ][
                "found_terms"
            ]
        ):
            print(
                "  Irrelevant terms: "
                + ", ".join(
                    result[
                        "relevance_details"
                    ][
                        "found_terms"
                    ]
                )
            )

    total = len(
        results
    )

    hard_passed = sum(
        1
        for result in results
        if result[
            "hard_safety_pass"
        ]
    )

    relevance_passed = sum(
        1
        for result in results
        if result[
            "relevance_pass"
        ]
    )

    combined_passed = sum(
        1
        for result in results
        if result[
            "combined_pass"
        ]
    )

    average_words = (
        sum(
            result[
                "answer_words"
            ]
            for result
            in results
        )
        / total
        if total
        else 0.0
    )

    return {
        "profile": profile,
        "summary": {
            "cases": total,
            "hard_safety_passed": (
                hard_passed
            ),
            "hard_safety_pass_rate": (
                hard_passed
                / total
                if total
                else 0.0
            ),
            "relevance_passed": (
                relevance_passed
            ),
            "relevance_pass_rate": (
                relevance_passed
                / total
                if total
                else 0.0
            ),
            "combined_passed": (
                combined_passed
            ),
            "combined_pass_rate": (
                combined_passed
                / total
                if total
                else 0.0
            ),
            "average_answer_words": (
                round(
                    average_words,
                    2,
                )
            ),
        },
        "results": results,
    }


def selection_key(
    evaluation: dict[str, Any],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    """
    Rank prompt profiles.

    Priority:
    1. Combined safety + relevance pass rate
    2. Hard safety pass rate
    3. Relevance pass rate
    4. Shorter average answer when quality scores tie
    """

    summary = (
        evaluation[
            "summary"
        ]
    )

    return (
        summary[
            "combined_pass_rate"
        ],
        summary[
            "hard_safety_pass_rate"
        ],
        summary[
            "relevance_pass_rate"
        ],
        -summary[
            "average_answer_words"
        ],
    )


def main() -> None:
    """Run and compare both LLM prompt approaches."""

    cases = (
        load_generation_cases()
    )

    print()
    print(
        "LLM approach evaluation"
    )
    print(
        "======================="
    )
    print()
    print(
        f"Cases: {len(cases)}"
    )
    print(
        "Profiles: "
        + ", ".join(
            PROMPT_PROFILES
        )
    )
    print()
    print(
        "This run makes real OpenAI API requests."
    )
    print(
        f"Expected generation requests: "
        f"{len(cases) * len(PROMPT_PROFILES)}"
    )

    retriever = (
        build_production_retriever()
    )

    client = OpenAI()

    evaluations: list[
        dict[str, Any]
    ] = []

    for profile in (
        PROMPT_PROFILES
    ):
        evaluation = (
            run_profile(
                profile=profile,
                cases=cases,
                retriever=retriever,
                client=client,
            )
        )

        evaluations.append(
            evaluation
        )

    ranked = sorted(
        evaluations,
        key=selection_key,
        reverse=True,
    )

    recommended = (
        ranked[0]["profile"]
    )

    payload = {
        "generated_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "model_comparison": (
            "same model, different system prompt"
        ),
        "profiles": list(
            PROMPT_PROFILES
        ),
        "selection_priority": [
            "combined_pass_rate",
            "hard_safety_pass_rate",
            "relevance_pass_rate",
            "shorter_average_answer",
        ],
        "recommended_profile": (
            recommended
        ),
        "evaluations": (
            evaluations
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print()
    print(
        "Comparison"
    )
    print(
        "=========="
    )

    print(
        f"{'Profile':<20}"
        f"{'Safety':>12}"
        f"{'Relevance':>14}"
        f"{'Combined':>12}"
        f"{'Avg words':>12}"
    )

    for evaluation in evaluations:
        summary = (
            evaluation[
                "summary"
            ]
        )

        print(
            f"{evaluation['profile']:<20}"
            f"{summary['hard_safety_pass_rate']:>12.4f}"
            f"{summary['relevance_pass_rate']:>14.4f}"
            f"{summary['combined_pass_rate']:>12.4f}"
            f"{summary['average_answer_words']:>12.2f}"
        )

    print()
    print(
        "Recommended production profile: "
        f"{recommended}"
    )

    print()
    print(
        f"Saved detailed results to: "
        f"{OUTPUT_PATH}"
    )

    print()
    print(
        "Do not change the production default until "
        "the generated answers have also been manually reviewed."
    )


if __name__ == "__main__":
    main()