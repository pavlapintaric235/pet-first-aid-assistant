import json

import pytest

from src.ingestion.process_sources import (
    create_chunk_id,
    load_raw_records,
    normalize_content,
    process_all_sources,
    process_raw_record,
    save_processed_documents,
    split_into_word_chunks,
)


def make_raw_record(
    source_id="test_source",
    content=None,
):
    if content is None:
        content = " ".join(
            f"word{index}"
            for index in range(500)
        )

    return {
        "source_id": source_id,
        "publisher": "Test Veterinary Publisher",
        "title": "Test First Aid Article",
        "original_url": "https://example.com/article",
        "final_url": "https://example.com/article",
        "species": ["dog", "cat"],
        "topics": ["first_aid", "emergency"],
        "language": "en",
        "authority_level": "test_reference",
        "content": content,
    }


def test_normalize_content():
    content = (
        "First   paragraph with extra spaces.\n\n"
        "\nSecond paragraph.\tWith a tab."
    )

    result = normalize_content(content)

    assert result == (
        "First paragraph with extra spaces.\n\n"
        "Second paragraph. With a tab."
    )


def test_split_into_word_chunks_with_overlap():
    content = " ".join(
        f"word{index}"
        for index in range(30)
    )

    chunks = split_into_word_chunks(
        content=content,
        chunk_size_words=10,
        overlap_words=2,
    )

    assert len(chunks) == 4

    first_chunk = chunks[0].split()
    second_chunk = chunks[1].split()

    assert first_chunk[-2:] == second_chunk[:2]
    assert len(first_chunk) == 10


def test_split_into_word_chunks_rejects_invalid_overlap():
    with pytest.raises(
        ValueError,
        match="overlap_words must be smaller",
    ):
        split_into_word_chunks(
            content="example content",
            chunk_size_words=10,
            overlap_words=10,
        )


def test_create_chunk_id_is_stable():
    first_id = create_chunk_id(
        source_id="source",
        chunk_index=0,
        content="Emergency veterinary content",
    )

    second_id = create_chunk_id(
        source_id="source",
        chunk_index=0,
        content="Emergency veterinary content",
    )

    assert first_id == second_id
    assert first_id.startswith("source_0000_")


def test_process_raw_record_preserves_metadata():
    record = make_raw_record()

    documents = process_raw_record(
        record=record,
        chunk_size_words=100,
        overlap_words=20,
    )

    assert len(documents) > 1

    first_document = documents[0]

    assert first_document["source_id"] == "test_source"
    assert first_document["publisher"] == (
        "Test Veterinary Publisher"
    )
    assert first_document["species"] == ["dog", "cat"]
    assert first_document["topics"] == [
        "first_aid",
        "emergency",
    ]
    assert first_document["chunk_index"] == 0
    assert first_document["total_chunks"] == len(documents)
    assert first_document["word_count"] == 100


def test_process_all_sources_generates_unique_ids():
    first_record = make_raw_record(
        source_id="first_source",
    )

    second_record = make_raw_record(
        source_id="second_source",
    )

    documents = process_all_sources(
        raw_records=[
            first_record,
            second_record,
        ],
        chunk_size_words=100,
        overlap_words=20,
    )

    document_ids = [
        document["id"]
        for document in documents
    ]

    assert len(document_ids) == len(set(document_ids))


def test_load_raw_records(tmp_path):
    first_record = make_raw_record(
        source_id="first_source",
    )

    second_record = make_raw_record(
        source_id="second_source",
    )

    (tmp_path / "first.json").write_text(
        json.dumps(first_record),
        encoding="utf-8",
    )

    (tmp_path / "second.json").write_text(
        json.dumps(second_record),
        encoding="utf-8",
    )

    records = load_raw_records(tmp_path)

    assert len(records) == 2
    assert records[0]["source_id"] == "first_source"
    assert records[1]["source_id"] == "second_source"


def test_load_raw_records_rejects_empty_directory(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="No raw JSON records",
    ):
        load_raw_records(tmp_path)


def test_save_processed_documents(tmp_path):
    output_path = tmp_path / "documents.json"

    documents = process_raw_record(
        record=make_raw_record(),
        chunk_size_words=100,
        overlap_words=20,
    )

    saved_path = save_processed_documents(
        documents=documents,
        output_path=output_path,
    )

    saved_documents = json.loads(
        saved_path.read_text(encoding="utf-8")
    )

    assert saved_path == output_path
    assert len(saved_documents) == len(documents)
    assert saved_documents[0]["source_id"] == "test_source"