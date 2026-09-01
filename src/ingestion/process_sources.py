from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "documents.json"
)

DEFAULT_CHUNK_SIZE_WORDS = 220
DEFAULT_OVERLAP_WORDS = 40
MINIMUM_INTRODUCTION_WORDS = 10

REQUIRED_RAW_FIELDS = {
    "source_id",
    "publisher",
    "title",
    "original_url",
    "species",
    "topics",
    "content",
}


def load_raw_records(
    raw_directory: Path = RAW_DATA_DIRECTORY,
) -> list[dict[str, Any]]:
    """Load all successfully ingested raw records."""

    if not raw_directory.exists():
        raise FileNotFoundError(
            "Raw data directory was not found at "
            f"{raw_directory}"
        )

    raw_paths = sorted(
        raw_directory.glob("*.json")
    )

    if not raw_paths:
        raise FileNotFoundError(
            "No raw JSON records were found in "
            f"{raw_directory}"
        )

    records: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        with raw_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            record = json.load(file)

        validate_raw_record(
            record,
            raw_path,
        )
        records.append(record)

    return records


def validate_raw_record(
    record: dict[str, Any],
    source_path: Path | None = None,
) -> None:
    """Validate fields needed for retrieval processing."""

    if not isinstance(record, dict):
        raise ValueError(
            "Each raw source record must be a JSON object."
        )

    missing_fields = (
        REQUIRED_RAW_FIELDS - record.keys()
    )

    if missing_fields:
        location = (
            f" in {source_path}"
            if source_path is not None
            else ""
        )

        missing = ", ".join(
            sorted(missing_fields)
        )

        raise ValueError(
            f"Raw source record{location} is missing "
            f"fields: {missing}"
        )

    if not isinstance(record["content"], str):
        raise ValueError(
            "Raw source content must be a string."
        )

    if not record["content"].strip():
        raise ValueError(
            "Raw source content cannot be blank."
        )

    if not isinstance(record["species"], list):
        raise ValueError(
            "The species field must be a list."
        )

    if not isinstance(record["topics"], list):
        raise ValueError(
            "The topics field must be a list."
        )

    sections = record.get("sections")

    if sections is not None:
        if not isinstance(sections, list):
            raise ValueError(
                "The sections field must be a list."
            )

        for section in sections:
            if not isinstance(section, dict):
                raise ValueError(
                    "Each section must be a JSON object."
                )

            if not isinstance(
                section.get("content"),
                str,
            ):
                raise ValueError(
                    "Each section must contain text content."
                )


def normalize_content(content: str) -> str:
    """Normalize text without changing medical meaning."""

    normalized_paragraphs: list[str] = []

    for paragraph in re.split(
        r"\n\s*\n",
        content,
    ):
        normalized = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if normalized:
            normalized_paragraphs.append(
                normalized
            )

    return "\n\n".join(
        normalized_paragraphs
    )


def validate_chunk_settings(
    chunk_size_words: int,
    overlap_words: int,
) -> None:
    """Validate shared chunking settings."""

    if chunk_size_words <= 0:
        raise ValueError(
            "chunk_size_words must be greater than zero."
        )

    if overlap_words < 0:
        raise ValueError(
            "overlap_words cannot be negative."
        )

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be smaller than "
            "chunk_size_words."
        )


def split_into_word_chunks(
    content: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """
    Split text into overlapping word windows.

    This remains available as a fallback for unusually long
    sentences that cannot be divided safely by punctuation.
    """

    validate_chunk_settings(
        chunk_size_words,
        overlap_words,
    )

    words = content.split()

    if not words:
        return []

    step_size = (
        chunk_size_words - overlap_words
    )
    chunks: list[str] = []

    start = 0

    while start < len(words):
        end = min(
            start + chunk_size_words,
            len(words),
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += step_size

    return chunks


def split_paragraph_into_units(
    paragraph: str,
    maximum_words: int,
) -> list[str]:
    """
    Divide an oversized paragraph at sentence boundaries.

    A word-based split is used only when a single sentence is
    itself larger than the configured chunk size.
    """

    if len(paragraph.split()) <= maximum_words:
        return [paragraph]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        paragraph,
    )

    units: list[str] = []

    for sentence in sentences:
        normalized_sentence = re.sub(
            r"\s+",
            " ",
            sentence,
        ).strip()

        if not normalized_sentence:
            continue

        if (
            len(normalized_sentence.split())
            <= maximum_words
        ):
            units.append(normalized_sentence)
            continue

        fallback_chunks = split_into_word_chunks(
            content=normalized_sentence,
            chunk_size_words=maximum_words,
            overlap_words=0,
        )

        units.extend(fallback_chunks)

    return units


def split_into_semantic_units(
    content: str,
    maximum_words: int,
) -> list[str]:
    """
    Convert content into paragraphs or sentence-level units.

    Paragraphs are preferred because they retain more context.
    """

    normalized_content = normalize_content(
        content
    )

    if not normalized_content:
        return []

    paragraphs = normalized_content.split(
        "\n\n"
    )

    units: list[str] = []

    for paragraph in paragraphs:
        paragraph_units = (
            split_paragraph_into_units(
                paragraph=paragraph,
                maximum_words=maximum_words,
            )
        )

        units.extend(paragraph_units)

    return units


def select_overlap_units(
    units: list[str],
    overlap_words: int,
) -> list[str]:
    """
    Select complete trailing units for the next chunk.

    Complete paragraphs or sentences are preferred over cutting
    the overlap in the middle of an instruction.
    """

    if overlap_words == 0:
        return []

    selected_reversed: list[str] = []
    selected_word_count = 0

    for unit in reversed(units):
        unit_word_count = len(unit.split())

        if (
            selected_word_count + unit_word_count
            > overlap_words
        ):
            break

        selected_reversed.append(unit)
        selected_word_count += unit_word_count

    return list(
        reversed(selected_reversed)
    )


def join_semantic_units(
    units: list[str],
) -> str:
    """Join semantic units into readable chunk content."""

    return "\n\n".join(units).strip()


def split_into_semantic_chunks(
    content: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """
    Create chunks using paragraphs and sentence boundaries.

    Chunks never exceed the configured maximum word count.
    """

    validate_chunk_settings(
        chunk_size_words,
        overlap_words,
    )

    units = split_into_semantic_units(
        content=content,
        maximum_words=chunk_size_words,
    )

    if not units:
        return []

    chunks: list[str] = []
    current_units: list[str] = []
    current_word_count = 0

    for unit in units:
        unit_word_count = len(unit.split())

        would_exceed_limit = (
            current_units
            and (
                current_word_count
                + unit_word_count
                > chunk_size_words
            )
        )

        if would_exceed_limit:
            chunk = join_semantic_units(
                current_units
            )

            if chunk:
                chunks.append(chunk)

            overlap_units = select_overlap_units(
                units=current_units,
                overlap_words=overlap_words,
            )

            overlap_word_count = sum(
                len(overlap_unit.split())
                for overlap_unit in overlap_units
            )

            if (
                overlap_word_count
                + unit_word_count
                > chunk_size_words
            ):
                current_units = []
                current_word_count = 0
            else:
                current_units = overlap_units
                current_word_count = (
                    overlap_word_count
                )

        current_units.append(unit)
        current_word_count += unit_word_count

    final_chunk = join_semantic_units(
        current_units
    )

    if final_chunk:
        chunks.append(final_chunk)

    return chunks


def create_chunk_id(
    source_id: str,
    chunk_index: int,
    content: str,
    section_index: int | None = None,
) -> str:
    """Create a stable identifier for one source chunk."""

    section_identity = (
        section_index
        if section_index is not None
        else "legacy"
    )

    identity = (
        f"{source_id}|{section_identity}|"
        f"{chunk_index}|{content}"
    )

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return (
        f"{source_id}_{chunk_index:04d}_{digest}"
    )


def create_legacy_section(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Create one section for raw records made before headings
    were preserved.
    """

    return {
        "heading": record["title"],
        "heading_level": 1,
        "heading_path": [record["title"]],
        "content": record["content"],
        "word_count": len(
            record["content"].split()
        ),
        "section_index": 0,
    }


def get_processable_sections(
    record: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Return useful structured sections from one raw record.

    Tiny artificial Introduction sections created from page
    accessibility text are excluded. Small medical sections under
    real headings are retained.
    """

    raw_sections = record.get("sections")

    if not raw_sections:
        return [create_legacy_section(record)]

    sections: list[dict[str, Any]] = []

    for fallback_index, section in enumerate(
        raw_sections
    ):
        content = normalize_content(
            section.get("content", "")
        )

        if not content:
            continue

        heading = (
            section.get("heading")
            or record["title"]
        ).strip()

        word_count = len(content.split())

        is_tiny_introduction = (
            heading.casefold() == "introduction"
            and word_count
            < MINIMUM_INTRODUCTION_WORDS
        )

        if is_tiny_introduction:
            continue

        heading_path = section.get(
            "heading_path"
        )

        if not isinstance(
            heading_path,
            list,
        ) or not heading_path:
            heading_path = [heading]

        normalized_heading_path = [
            str(path_part).strip()
            for path_part in heading_path
            if str(path_part).strip()
        ]

        sections.append(
            {
                "heading": heading,
                "heading_level": section.get(
                    "heading_level",
                    1,
                ),
                "heading_path": (
                    normalized_heading_path
                    or [heading]
                ),
                "content": content,
                "word_count": word_count,
                "section_index": section.get(
                    "section_index",
                    fallback_index,
                ),
            }
        )

    if not sections:
        return [create_legacy_section(record)]

    return sections


def build_embedding_text(
    record: dict[str, Any],
    heading_path: list[str],
    chunk_content: str,
) -> str:
    """
    Build text specifically designed for embedding.

    Metadata helps short procedural chunks retain the condition,
    species and article context they belong to.
    """

    species_text = ", ".join(
        record["species"]
    )

    topics_text = ", ".join(
        record["topics"]
    )

    section_text = " > ".join(
        heading_path
    )

    return (
        f"Title: {record['title']}\n"
        f"Species: {species_text}\n"
        f"Topics: {topics_text}\n"
        f"Section: {section_text}\n"
        f"Content: {chunk_content}"
    )


def process_raw_record(
    record: dict[str, Any],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Convert one raw article into retrieval documents."""

    validate_raw_record(record)

    validate_chunk_settings(
        chunk_size_words,
        overlap_words,
    )

    sections = get_processable_sections(record)
    pending_documents: list[dict[str, Any]] = []

    for section in sections:
        section_chunks = split_into_semantic_chunks(
            content=section["content"],
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        section_total_chunks = len(
            section_chunks
        )

        for section_chunk_index, chunk_content in enumerate(
            section_chunks
        ):
            heading_path = section[
                "heading_path"
            ]

            pending_documents.append(
                {
                    "source_id": record[
                        "source_id"
                    ],
                    "publisher": record[
                        "publisher"
                    ],
                    "title": record["title"],
                    "url": record.get(
                        "final_url",
                        record["original_url"],
                    ),
                    "species": record["species"],
                    "topics": record["topics"],
                    "language": record.get(
                        "language",
                        "en",
                    ),
                    "authority_level": record.get(
                        "authority_level"
                    ),
                    "source_status": record.get(
                        "source_status",
                        "approved",
                    ),
                    "section_heading": section[
                        "heading"
                    ],
                    "section_heading_level": section[
                        "heading_level"
                    ],
                    "heading_path": heading_path,
                    "section_index": section[
                        "section_index"
                    ],
                    "section_chunk_index": (
                        section_chunk_index
                    ),
                    "section_total_chunks": (
                        section_total_chunks
                    ),
                    "word_count": len(
                        chunk_content.split()
                    ),
                    "content": chunk_content,
                    "embedding_text": (
                        build_embedding_text(
                            record=record,
                            heading_path=heading_path,
                            chunk_content=(
                                chunk_content
                            ),
                        )
                    ),
                }
            )

    total_chunks = len(pending_documents)
    documents: list[dict[str, Any]] = []

    for chunk_index, document in enumerate(
        pending_documents
    ):
        chunk_id = create_chunk_id(
            source_id=record["source_id"],
            chunk_index=chunk_index,
            content=document["content"],
            section_index=document[
                "section_index"
            ],
        )

        document["id"] = chunk_id
        document["chunk_index"] = chunk_index
        document["total_chunks"] = total_chunks

        documents.append(document)

    return documents


def process_all_sources(
    raw_records: list[dict[str, Any]],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Process every raw source into retrieval documents."""

    documents: list[dict[str, Any]] = []

    for record in raw_records:
        source_documents = process_raw_record(
            record=record,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        )

        documents.extend(source_documents)

    document_ids = [
        document["id"]
        for document in documents
    ]

    if len(document_ids) != len(
        set(document_ids)
    ):
        raise ValueError(
            "Duplicate document IDs were generated."
        )

    return documents


def save_processed_documents(
    documents: list[dict[str, Any]],
    output_path: Path = PROCESSED_DATA_PATH,
) -> Path:
    """Save processed retrieval documents as JSON."""

    if not documents:
        raise ValueError(
            "No processed documents were provided "
            "for saving."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Convert raw veterinary articles into "
            "section-aware retrieval documents."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_WORDS,
        help=(
            "Maximum words per chunk. "
            f"Default: {DEFAULT_CHUNK_SIZE_WORDS}"
        ),
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
        help=(
            "Target overlap in complete semantic units. "
            f"Default: {DEFAULT_OVERLAP_WORDS}"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Process all downloaded sources."""

    arguments = parse_arguments()

    raw_records = load_raw_records()

    documents = process_all_sources(
        raw_records=raw_records,
        chunk_size_words=arguments.chunk_size,
        overlap_words=arguments.overlap,
    )

    output_path = save_processed_documents(
        documents
    )

    relative_output_path = (
        output_path.relative_to(PROJECT_ROOT)
    )

    print(
        f"Raw sources processed: "
        f"{len(raw_records)}"
    )
    print(
        f"Retrieval documents created: "
        f"{len(documents)}"
    )
    print(
        f"Saved to: {relative_output_path}"
    )


if __name__ == "__main__":
    main()