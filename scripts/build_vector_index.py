from src.retrieval.embedder import Embedder
from src.retrieval.keyword_search import (
    load_processed_documents,
)
from src.retrieval.vector_search import (
    build_embedding_matrix,
    save_embedding_matrix,
)


def main() -> None:
    """Build and persist embeddings for all documents."""

    documents = load_processed_documents()
    embedder = Embedder()

    print(f"Documents to embed: {len(documents)}")
    print("Creating embeddings...")

    embeddings = build_embedding_matrix(
        documents=documents,
        embedder=embedder,
        batch_size=16,
    )

    output_path = save_embedding_matrix(embeddings)

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()