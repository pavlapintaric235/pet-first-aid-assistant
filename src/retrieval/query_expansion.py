from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ExpansionRule:
    """One conservative terminology-expansion rule."""

    name: str
    patterns: tuple[re.Pattern[str], ...]
    additions: tuple[str, ...]


class SearchEngine(Protocol):
    """Interface required by the query-expansion wrapper."""

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked retrieval results."""


def _compile_patterns(
    *patterns: str,
) -> tuple[re.Pattern[str], ...]:
    """Compile case-insensitive regular expressions."""

    return tuple(
        re.compile(
            pattern,
            flags=re.IGNORECASE,
        )
        for pattern in patterns
    )


EXPANSION_RULES: tuple[ExpansionRule, ...] = (
    ExpansionRule(
        name="breathing_difficulty",
        patterns=_compile_patterns(
            r"\bcan(?:not|'t)\s+(?:breathe|breath)\b",
            r"\bcan(?:not|'t)\s+get\s+(?:enough\s+)?air\b",
            r"\bnot\s+getting\s+(?:enough\s+)?air\b",
            r"\b(?:strange|strangely|abnormal|labored|rapid)\s+breath(?:ing)?\b",
            r"\bbreath(?:ing)?\s+(?:strange|strangely|abnormal|hard|fast)\b",
            r"\bdifficulty\s+breath(?:ing)?\b",
            r"\bshort(?:ness)?\s+of\s+breath\b",
        ),
        additions=(
            "breathing difficulty",
            "respiratory distress",
        ),
    ),
    ExpansionRule(
        name="choking_airway",
        patterns=_compile_patterns(
            r"\bchok(?:e|ing|ed)\b",
            r"\bsomething\s+stuck\s+in\s+(?:the\s+)?throat\b",
            r"\bobject\s+stuck\s+in\s+(?:the\s+)?throat\b",
            r"\bairway\s+block(?:ed|age)?\b",
        ),
        additions=(
            "choking",
            "airway obstruction",
        ),
    ),
    ExpansionRule(
        name="unconsciousness",
        patterns=_compile_patterns(
            r"\bunconscious\b",
            r"\bunresponsive\b",
            r"\bpassed\s+out\b",
            r"\bnot\s+waking\s+up\b",
        ),
        additions=(
            "unconsciousness",
            "unresponsive",
        ),
    ),
    ExpansionRule(
        name="cpr_arrest_signs",
        patterns=_compile_patterns(
            r"\bnot\s+breath(?:ing)?\b",
            r"\bno\s+(?:detectable\s+)?heartbeat\b",
            r"\bcannot\s+(?:detect|feel)\s+(?:a\s+)?heartbeat\b",
            r"\bcpr\b",
        ),
        additions=(
            "CPR",
            "rescue breathing",
        ),
    ),
    ExpansionRule(
        name="severe_bleeding",
        patterns=_compile_patterns(
            r"\bbleed(?:ing|s)?\b",
            r"\blosing\s+(?:a\s+lot\s+of|too\s+much)\s+blood\b",
            r"\bblood\s+(?:will\s+)?not\s+stop\b",
            r"\bdeep\s+wound\b",
        ),
        additions=(
            "severe bleeding",
            "hemorrhage",
        ),
    ),
    ExpansionRule(
        name="burns",
        patterns=_compile_patterns(
            r"\bburn(?:ed|ing|s)?\b",
            r"\bhot\s+surface\b",
            r"\bscald(?:ed|ing)?\b",
        ),
        additions=(
            "burns",
            "thermal injury",
        ),
    ),
    ExpansionRule(
        name="suspected_fracture",
        patterns=_compile_patterns(
            r"\bbroken\s+(?:leg|bone|limb|paw)\b",
            r"\bmay\s+be\s+broken\b",
            r"\bthink\s+(?:it|the\s+leg|the\s+bone)\s+is\s+broken\b",
            r"\bcannot\s+(?:put|bear)\s+weight\b",
        ),
        additions=(
            "suspected fracture",
            "bone injury",
        ),
    ),
    ExpansionRule(
        name="transport",
        patterns=_compile_patterns(
            r"\b(?:safely\s+)?move\s+(?:an?\s+)?injured\b",
            r"\btransport\b",
            r"\binto\s+(?:my|the)\s+car\b",
            r"\binto\s+(?:a|the)\s+carrier\b",
        ),
        additions=(
            "first aid transport",
            "safe handling",
        ),
    ),
    ExpansionRule(
        name="shock_signs",
        patterns=_compile_patterns(
            r"\bpale\s+gums\b",
            r"\bvery\s+pale\s+gums\b",
            r"\bcold\s+(?:feet|paws|limbs)\b",
            r"\bweak\s*,?\s+cold\b",
        ),
        additions=(
            "shock",
            "emergency first aid",
        ),
    ),
    ExpansionRule(
        name="poisoning",
        patterns=_compile_patterns(
            r"\bpoison(?:ed|ing|ous)?\b",
            r"\btoxic\b",
            r"\btoxins?\b",
            r"\bswallow(?:ed|ing)?\s+something\b",
            r"\bate\s+(?:something\s+)?(?:harmful|dangerous|toxic)\b",
        ),
        additions=(
            "poisoning",
            "toxin ingestion",
        ),
    ),
    ExpansionRule(
        name="eye_injury",
        patterns=_compile_patterns(
            r"\beye\s+injur(?:y|ed)\b",
            r"\binjured\s+(?:its|his|her|the)\s+eye\b",
            r"\bkeeping\s+(?:its|his|her|the)\s+eye\s+closed\b",
        ),
        additions=(
            "eye injury",
            "ocular injury",
        ),
    ),
    ExpansionRule(
        name="major_trauma",
        patterns=_compile_patterns(
            r"\bhit\s+by\s+(?:a\s+)?car\b",
            r"\bvehicle\s+(?:accident|collision)\b",
            r"\bmajor\s+trauma\b",
            r"\bserious\s+accident\b",
        ),
        additions=(
            "major trauma",
            "emergency transport",
        ),
    ),
    ExpansionRule(
        name="handler_safety",
        patterns=_compile_patterns(
            r"\btrying\s+to\s+bite\b",
            r"\bmay\s+bite\b",
            r"\bfrightened\s+and\s+(?:trying\s+to\s+)?bite\b",
            r"\baggressive\s+(?:when|while)\s+(?:handled|moving)\b",
        ),
        additions=(
            "handler safety",
            "safe restraint",
        ),
    ),
    ExpansionRule(
        name="seizures",
        patterns=_compile_patterns(
            r"\bseizure(?:s)?\b",
            r"\bconvulsion(?:s)?\b",
            r"\brepeated\s+fits?\b",
        ),
        additions=(
            "seizures",
            "neurologic emergency",
        ),
    ),
    ExpansionRule(
        name="rapid_deterioration",
        patterns=_compile_patterns(
            r"\bextreme\s+pain\b",
            r"\bgetting\s+worse\s+(?:very\s+)?quickly\b",
            r"\bworsen(?:ing|ed)?\s+(?:very\s+)?rapidly\b",
            r"\bdeteriorat(?:ing|ed)\s+(?:very\s+)?rapidly\b",
        ),
        additions=(
            "emergency warning signs",
            "urgent veterinary care",
        ),
    ),
)


def matched_expansions(
    query: str,
) -> list[dict[str, Any]]:
    """Return the rules and terms matched by a query."""

    if not isinstance(query, str):
        raise TypeError(
            "The query must be a string."
        )

    if not query.strip():
        raise ValueError(
            "The query must contain searchable text."
        )

    matches: list[dict[str, Any]] = []

    for rule in EXPANSION_RULES:
        if any(
            pattern.search(query)
            for pattern in rule.patterns
        ):
            matches.append(
                {
                    "rule": rule.name,
                    "additions": list(
                        rule.additions
                    ),
                }
            )

    return matches


def expansion_terms(
    query: str,
) -> list[str]:
    """Return deduplicated terminology additions for a query."""

    matches = matched_expansions(
        query
    )

    query_lower = query.lower()
    seen: set[str] = set()
    terms: list[str] = []

    for match in matches:
        for term in match["additions"]:
            normalized_term = term.lower()

            if normalized_term in query_lower:
                continue

            if normalized_term in seen:
                continue

            seen.add(normalized_term)
            terms.append(term)

    return terms


def expand_query(
    query: str,
) -> str:
    """
    Append conservative retrieval terminology to a query.

    The original query is preserved verbatim. Expansion only adds
    neutral retrieval terms; it does not replace the user's wording.
    """

    terms = expansion_terms(
        query
    )

    if not terms:
        return query

    return (
        f"{query} "
        + " ".join(terms)
    )


class QueryExpansionSearch:
    """Apply deterministic terminology expansion before retrieval."""

    def __init__(
        self,
        search_engine: SearchEngine,
    ) -> None:
        self.search_engine = search_engine

    def search(
        self,
        query: str,
        num_results: int = 5,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Expand the query once, then delegate to the search engine."""

        expanded_query = expand_query(
            query
        )

        terms = expansion_terms(
            query
        )

        matches = matched_expansions(
            query
        )

        results = self.search_engine.search(
            query=expanded_query,
            num_results=num_results,
            species=species,
        )

        for result in results:
            result["query_expanded"] = bool(
                terms
            )
            result["original_query"] = query
            result["expanded_query"] = (
                expanded_query
            )
            result["query_expansion_terms"] = list(
                terms
            )
            result["query_expansion_rules"] = [
                match["rule"]
                for match in matches
            ]

        return results