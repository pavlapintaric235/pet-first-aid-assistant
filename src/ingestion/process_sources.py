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
    PROJECT_ROOT / "data" / "processed" / "documents.json"
)

DEFAULT_CHUNK_SIZE_WORDS = 220
DEFAULT_OVERLAP_WORDS = 40

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
    """Load all successfully ingested raw source records."""

    if not raw_directory.exists():
        raise FileNotFoundError(
            f"Raw data directory was not found at {raw_directory}"
        )

    raw_paths = sorted(raw_directory.glob("*.json"))

    if not raw_paths:
        raise FileNotFoundError(
            f"No raw JSON records were found in {raw_directory}"
        )

    records: list[dict[str, Any]] = []

    for raw_path in raw_paths:
        with raw_path.open("r", encoding="utf-8") as file:
            record = json.load(file)

        validate_raw_record(record, raw_path)
        records.append(record)

    return records


def validate_raw_record(
    record: dict[str, Any],
    source_path: Path | None = None,
) -> None:
    """Validate the fields needed for processing and retrieval."""

    if not isinstance(record, dict):
        raise ValueError("Each raw source record must be a JSON object.")

    missing_fields = REQUIRED_RAW_FIELDS - record.keys()

    if missing_fields:
        location = (
            f" in {source_path}"
            if source_path is not None
            else ""
        )
        missing = ", ".join(sorted(missing_fields))

        raise ValueError(
            f"Raw source record{location} is missing fields: "
            f"{missing}"
        )

    if not isinstance(record["content"], str):
        raise ValueError("Raw source content must be a string.")

    if not record["content"].strip():
        raise ValueError("Raw source content cannot be blank.")

    if not isinstance(record["species"], list):
        raise ValueError("The species field must be a list.")

    if not isinstance(record["topics"], list):
        raise ValueError("The topics field must be a list.")


def normalize_content(content: str) -> str:
    """Normalize source text without changing its medical meaning."""

    normalized_paragraphs: list[str] = []

    for paragraph in re.split(r"\n\s*\n", content):
        normalized = re.sub(r"\s+", " ", paragraph).strip()

        if normalized:
            normalized_paragraphs.append(normalized)

    return "\n\n".join(normalized_paragraphs)


def split_into_word_chunks(
    content: str,
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into overlapping fixed-size word windows."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than zero.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError(
            "overlap_words must be smaller than chunk_size_words."
        )

    words = content.split()

    if not words:
        return []

    step_size = chunk_size_words - overlap_words
    chunks: list[str] = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += step_size

    return chunks


def create_chunk_id(
    source_id: str,
    chunk_index: int,
    content: str,
) -> str:
    """Create a stable identifier for one source chunk."""

    identity = (
        f"{source_id}|{chunk_index}|{content}"
    )

    digest = hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()[:16]

    return f"{source_id}_{chunk_index:04d}_{digest}"


def process_raw_record(
    record: dict[str, Any],
    chunk_size_words: int = DEFAULT_CHUNK_SIZE_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Convert one raw article into retrieval documents."""

    validate_raw_record(record)

    normalized_content = normalize_content(record["content"])

    text_chunks = split_into_word_chunks(
        content=normalized_content,
        chunk_size_words=chunk_size_words,
        overlap_words=overlap_words,
    )

    documents: list[dict[str, Any]] = []

    total_chunks = len(text_chunks)

    for chunk_index, chunk_content in enumerate(text_chunks):
        chunk_id = create_chunk_id(
            source_id=record["source_id"],
            chunk_index=chunk_index,
            content=chunk_content,
        )

        document = {
            "id": chunk_id,
            "source_id": record["source_id"],
            "publisher": record["publisher"],
            "title": record["title"],
            "url": record.get(
                "final_url",
                record["original_url"],
            ),
            "species": record["species"],
            "topics": record["topics"],
            "language": record.get("language", "en"),
            "authority_level": record.get("authority_level"),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "word_count": len(chunk_content.split()),
            "content": chunk_content,
        }

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

    if len(document_ids) != len(set(document_ids)):
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
            "No processed documents were provided for saving."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", encoding="utf-8") as file:
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
            "Convert raw veterinary articles into overlapping "
            "retrieval documents."
        )
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_WORDS,
        help=(
            "Maximum number of words per chunk. "
            f"Default: {DEFAULT_CHUNK_SIZE_WORDS}"
        ),
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
        help=(
            "Number of words shared by consecutive chunks. "
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

    output_path = save_processed_documents(documents)

    relative_output_path = output_path.relative_to(
        PROJECT_ROOT
    )

    print(f"Raw sources processed: {len(raw_records)}")
    print(f"Retrieval documents created: {len(documents)}")
    print(f"Saved to: {relative_output_path}")


if __name__ == "__main__":
    main()