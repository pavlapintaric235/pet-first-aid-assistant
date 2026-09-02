from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


PoolingMode = Literal["mean", "cls"]

DEFAULT_MODEL_PATH = "models/Xenova/all-MiniLM-L6-v2"
DEFAULT_POOLING: PoolingMode = "mean"


def prepare_query_text(
    text: str,
    query_prefix: str = "",
) -> str:
    """Apply an optional model-specific prefix to a query."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    if not isinstance(query_prefix, str):
        raise TypeError("query_prefix must be a string.")

    return f"{query_prefix}{text}"


def pool_hidden_states(
    hidden_states: np.ndarray,
    attention_mask: np.ndarray,
    pooling: PoolingMode = DEFAULT_POOLING,
) -> np.ndarray:
    """Pool token embeddings into one vector per input text."""

    hidden = np.asarray(
        hidden_states,
        dtype=np.float32,
    )

    mask = np.asarray(
        attention_mask,
        dtype=np.float32,
    )

    if hidden.ndim != 3:
        raise ValueError(
            "hidden_states must have shape "
            "(batch, sequence, hidden_size)."
        )

    if mask.ndim != 2:
        raise ValueError(
            "attention_mask must have shape "
            "(batch, sequence)."
        )

    if hidden.shape[:2] != mask.shape:
        raise ValueError(
            "hidden_states and attention_mask "
            "must have matching batch and sequence dimensions."
        )

    if pooling == "cls":
        return np.asarray(
            hidden[:, 0, :],
            dtype=np.float32,
        )

    if pooling == "mean":
        expanded_mask = mask[..., None]

        token_counts = expanded_mask.sum(
            axis=1,
        )

        if np.any(token_counts == 0):
            raise ValueError(
                "Cannot mean-pool a sequence with "
                "no unmasked tokens."
            )

        summed = (
            hidden * expanded_mask
        ).sum(
            axis=1,
        )

        return np.asarray(
            summed / token_counts,
            dtype=np.float32,
        )

    raise ValueError(
        "pooling must be either 'mean' or 'cls'."
    )


def normalize_rows(
    matrix: np.ndarray,
) -> np.ndarray:
    """L2-normalize every row of a matrix."""

    array = np.asarray(
        matrix,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "Embedding matrix must be two-dimensional."
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    if np.any(norms == 0):
        raise ValueError(
            "Embedding vectors must have non-zero L2 norm."
        )

    return np.asarray(
        array / norms,
        dtype=np.float32,
    )


class Embedder:
    """Generate text embeddings with an ONNX transformer model."""

    def __init__(
        self,
        path: str | Path = DEFAULT_MODEL_PATH,
        pooling: PoolingMode = DEFAULT_POOLING,
        query_prefix: str = "",
    ) -> None:
        if pooling not in {
            "mean",
            "cls",
        }:
            raise ValueError(
                "pooling must be either 'mean' or 'cls'."
            )

        if not isinstance(query_prefix, str):
            raise TypeError(
                "query_prefix must be a string."
            )

        path = Path(path)

        tokenizer_path = (
            path / "tokenizer.json"
        )

        model_path = (
            path / "model.onnx"
        )

        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer was not found: "
                f"{tokenizer_path}"
            )

        if not model_path.exists():
            raise FileNotFoundError(
                f"ONNX model was not found: "
                f"{model_path}"
            )

        self.pooling = pooling
        self.query_prefix = query_prefix

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        self.session = ort.InferenceSession(
            str(model_path),
            providers=[
                "CPUExecutionProvider",
            ],
        )

        self.input_names = {
            input_info.name
            for input_info
            in self.session.get_inputs()
        }

    def encode(
        self,
        text: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Embed one retrieval query.

        An optional query prefix is applied here only.
        Document batches are embedded without that prefix.
        """

        query_text = prepare_query_text(
            text=text,
            query_prefix=self.query_prefix,
        )

        return self.encode_batch(
            [query_text],
            normalize=normalize,
        )[0]

    def encode_batch(
        self,
        texts: list[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a batch of texts without a query prefix."""

        if not texts:
            raise ValueError(
                "At least one text is required."
            )

        self.tokenizer.enable_padding()

        encoded = self.tokenizer.encode_batch(
            texts
        )

        input_ids = np.array(
            [
                item.ids
                for item in encoded
            ],
            dtype=np.int64,
        )

        attention_mask = np.array(
            [
                item.attention_mask
                for item in encoded
            ],
            dtype=np.int64,
        )

        token_type_ids = np.array(
            [
                item.type_ids
                for item in encoded
            ],
            dtype=np.int64,
        )

        feed: dict[str, np.ndarray] = {}

        if "input_ids" in self.input_names:
            feed[
                "input_ids"
            ] = input_ids

        if (
            "attention_mask"
            in self.input_names
        ):
            feed[
                "attention_mask"
            ] = attention_mask

        if (
            "token_type_ids"
            in self.input_names
        ):
            feed[
                "token_type_ids"
            ] = token_type_ids

        hidden_states = self.session.run(
            None,
            feed,
        )[0]

        pooled = pool_hidden_states(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            pooling=self.pooling,
        )

        if normalize:
            pooled = normalize_rows(
                pooled
            )

        return np.asarray(
            pooled,
            dtype=np.float32,
        )