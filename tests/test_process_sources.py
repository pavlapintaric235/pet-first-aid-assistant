import json

import pytest

from src.ingestion.process_sources import (
    build_embedding_text,
    create_chunk_id,
    get_processable_sections,
    load_raw_records,
    normalize_content,
    process_all_sources,
    process_raw_record,
    save_processed_documents,
    split_into_semantic_chunks,
    split_into_word_chunks,
)


def make_raw_record(
    source_id="test_source",
    content=None,
    sections=None,
):
    if content is None:
        content = " ".join(
            f"word{index}"
            for index in range(500)
        )

    record = {
        "source_id": source_id,
        "publisher": (
            "Test Veterinary Publisher"
        ),
        "title": "Test First Aid Article",
        "original_url": (
            "https://example.com/article"
        ),
        "final_url": (
            "https://example.com/article"
        ),
        "species": ["dog", "cat"],
        "topics": [
            "first_aid",
            "emergency",
        ],
        "language": "en",
        "authority_level": "test_reference",
        "source_status": "approved",
        "content": content,
    }

    if sections is not None:
        record["sections"] = sections
        record["section_count"] = len(sections)

    return record


def make_section(
    heading,
    content,
    section_index,
    heading_level=2,
    heading_path=None,
):
    if heading_path is None:
        heading_path = [
            "Test First Aid Article",
            heading,
        ]

    return {
        "heading": heading,
        "heading_level": heading_level,
        "heading_path": heading_path,
        "content": content,
        "word_count": len(content.split()),
        "section_index": section_index,
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


def test_semantic_chunks_prefer_paragraph_boundaries():
    first_paragraph = " ".join(
        f"first{index}"
        for index in range(8)
    )

    second_paragraph = " ".join(
        f"second{index}"
        for index in range(8)
    )

    content = (
        f"{first_paragraph}\n\n"
        f"{second_paragraph}"
    )

    chunks = split_into_semantic_chunks(
        content=content,
        chunk_size_words=10,
        overlap_words=2,
    )

    assert len(chunks) == 2
    assert chunks[0] == first_paragraph
    assert chunks[1] == second_paragraph


def test_semantic_chunks_split_oversized_paragraph_by_sentence():
    sentence_one = (
        "The dog requires immediate veterinary care."
    )

    sentence_two = (
        "Transport the dog carefully to avoid injury."
    )

    sentence_three = (
        "Call the emergency clinic before arriving."
    )

    content = " ".join(
        [
            sentence_one,
            sentence_two,
            sentence_three,
        ]
    )

    chunks = split_into_semantic_chunks(
        content=content,
        chunk_size_words=12,
        overlap_words=3,
    )

    assert len(chunks) >= 2
    assert sentence_one in chunks[0]
    assert all(
        len(chunk.split()) <= 12
        for chunk in chunks
    )


def test_create_chunk_id_is_stable():
    first_id = create_chunk_id(
        source_id="source",
        chunk_index=0,
        content=(
            "Emergency veterinary content"
        ),
        section_index=2,
    )

    second_id = create_chunk_id(
        source_id="source",
        chunk_index=0,
        content=(
            "Emergency veterinary content"
        ),
        section_index=2,
    )

    assert first_id == second_id
    assert first_id.startswith(
        "source_0000_"
    )


def test_tiny_introduction_is_removed():
    sections = [
        make_section(
            heading="Introduction",
            content="Skip content",
            section_index=0,
            heading_level=0,
            heading_path=["Introduction"],
        ),
        make_section(
            heading="Bleeding",
            content=(
                "Apply direct pressure and contact "
                "a veterinarian immediately."
            ),
            section_index=1,
        ),
    ]

    record = make_raw_record(
        sections=sections
    )

    processable_sections = (
        get_processable_sections(record)
    )

    assert len(processable_sections) == 1
    assert (
        processable_sections[0]["heading"]
        == "Bleeding"
    )


def test_small_medical_section_is_retained():
    sections = [
        make_section(
            heading="Poisoning",
            content=(
                "Call a veterinarian immediately."
            ),
            section_index=0,
        )
    ]

    record = make_raw_record(
        sections=sections
    )

    processable_sections = (
        get_processable_sections(record)
    )

    assert len(processable_sections) == 1
    assert (
        processable_sections[0]["heading"]
        == "Poisoning"
    )


def test_processing_does_not_cross_section_boundaries():
    bleeding_content = " ".join(
        f"bleeding{index}"
        for index in range(40)
    )

    poisoning_content = " ".join(
        f"poison{index}"
        for index in range(40)
    )

    sections = [
        make_section(
            heading="Bleeding",
            content=bleeding_content,
            section_index=0,
        ),
        make_section(
            heading="Poisoning",
            content=poisoning_content,
            section_index=1,
        ),
    ]

    documents = process_raw_record(
        record=make_raw_record(
            sections=sections
        ),
        chunk_size_words=25,
        overlap_words=5,
    )

    for document in documents:
        contains_bleeding = (
            "bleeding" in document["content"]
        )
        contains_poisoning = (
            "poison" in document["content"]
        )

        assert not (
            contains_bleeding
            and contains_poisoning
        )


def test_process_raw_record_preserves_metadata():
    first_content = " ".join(
        f"first{index}"
        for index in range(120)
    )

    second_content = " ".join(
        f"second{index}"
        for index in range(60)
    )

    sections = [
        make_section(
            heading="Bleeding",
            content=first_content,
            section_index=0,
        ),
        make_section(
            heading="Safe Transport",
            content=second_content,
            section_index=1,
        ),
    ]

    record = make_raw_record(
        sections=sections
    )

    documents = process_raw_record(
        record=record,
        chunk_size_words=100,
        overlap_words=20,
    )

    assert len(documents) > 1

    first_document = documents[0]

    assert (
        first_document["source_id"]
        == "test_source"
    )
    assert first_document["publisher"] == (
        "Test Veterinary Publisher"
    )
    assert first_document["species"] == [
        "dog",
        "cat",
    ]
    assert first_document["topics"] == [
        "first_aid",
        "emergency",
    ]
    assert (
        first_document["section_heading"]
        == "Bleeding"
    )
    assert first_document["heading_path"] == [
        "Test First Aid Article",
        "Bleeding",
    ]
    assert first_document["chunk_index"] == 0
    assert first_document["total_chunks"] == len(
        documents
    )
    assert first_document["word_count"] <= 100
    assert "Title: Test First Aid Article" in (
        first_document["embedding_text"]
    )
    assert (
        "Section: Test First Aid Article > Bleeding"
        in first_document["embedding_text"]
    )


def test_build_embedding_text_includes_metadata():
    record = make_raw_record()

    embedding_text = build_embedding_text(
        record=record,
        heading_path=[
            "Emergency First Aid",
            "Heat Stroke",
        ],
        chunk_content=(
            "Move the pet to a cooler location."
        ),
    )

    assert (
        "Title: Test First Aid Article"
        in embedding_text
    )
    assert "Species: dog, cat" in embedding_text
    assert (
        "Topics: first_aid, emergency"
        in embedding_text
    )
    assert (
        "Section: Emergency First Aid > Heat Stroke"
        in embedding_text
    )
    assert (
        "Move the pet to a cooler location."
        in embedding_text
    )


def test_legacy_record_without_sections_is_supported():
    record = make_raw_record()

    documents = process_raw_record(
        record=record,
        chunk_size_words=100,
        overlap_words=20,
    )

    assert len(documents) > 1

    assert all(
        document["section_heading"]
        == "Test First Aid Article"
        for document in documents
    )


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

    assert len(document_ids) == len(
        set(document_ids)
    )


def test_load_raw_records(tmp_path):
    first_record = make_raw_record(
        source_id="first_source",
    )

    second_record = make_raw_record(
        source_id="second_source",
    )

    (
        tmp_path / "first.json"
    ).write_text(
        json.dumps(first_record),
        encoding="utf-8",
    )

    (
        tmp_path / "second.json"
    ).write_text(
        json.dumps(second_record),
        encoding="utf-8",
    )

    records = load_raw_records(tmp_path)

    assert len(records) == 2
    assert (
        records[0]["source_id"]
        == "first_source"
    )
    assert (
        records[1]["source_id"]
        == "second_source"
    )


def test_load_raw_records_rejects_empty_directory(
    tmp_path,
):
    with pytest.raises(
        FileNotFoundError,
        match="No raw JSON records",
    ):
        load_raw_records(tmp_path)


def test_save_processed_documents(tmp_path):
    output_path = (
        tmp_path / "documents.json"
    )

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
        saved_path.read_text(
            encoding="utf-8"
        )
    )

    assert saved_path == output_path
    assert len(saved_documents) == len(
        documents
    )
    assert (
        saved_documents[0]["source_id"]
        == "test_source"
    )