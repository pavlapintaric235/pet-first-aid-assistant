from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    documents_with_embedding_text = sum(
        1
        for document in documents
        if isinstance(document.get("embedding_text"), str)
        and document["embedding_text"].strip()
    )

    print(f"Documents to embed: {len(documents)}")
    print(
        "Documents with embedding_text: "
        f"{documents_with_embedding_text}/{len(documents)}"
    )
    print("Creating embeddings...")

    embeddings = build_embedding_matrix(
        documents=documents,
        embedder=embedder,
        batch_size=16,
    )

    output_path = save_embedding_matrix(embeddings)

    vector_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    print(f"Embedding shape: {embeddings.shape}")
    print(
        "Minimum vector norm: "
        f"{vector_norms.min():.6f}"
    )
    print(
        "Maximum vector norm: "
        f"{vector_norms.max():.6f}"
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()