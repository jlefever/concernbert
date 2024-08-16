import abc
import os
from io import BytesIO
from typing import Any

from hashlib import sha256

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

import lmdb


class Embedder(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts: list[str], *, pbar: bool) -> dict[str, np.ndarray]:
        pass

    @abc.abstractmethod
    def device(self) -> str:
        pass


class RawEmbedder(Embedder):
    def __init__(self, model: SentenceTransformer | str, batch_size: int):
        if isinstance(model, str):
            self._model = SentenceTransformer(model)
        else:
            self._model = model
        self._batch_size = batch_size

    def embed(self, texts: list[str], *, pbar: bool) -> dict[str, np.ndarray]:
        ret: dict[str, Any] = {}
        unique_texts = list(set(texts))
        with torch.no_grad():
            embeddings: list[np.ndarray] = self._model.encode(
                unique_texts,
                batch_size=self._batch_size,
                show_progress_bar=pbar,
                convert_to_numpy=True,
            )  # type: ignore
            for text, embedding in zip(unique_texts, embeddings):
                ret[text] = embedding
        return ret

    def device(self) -> str:
        return str(self._model.device)


class CachingEmbedder(Embedder):
    def __init__(self, embedder: RawEmbedder, cache_path: str):
        self._embedder = embedder
        self._cache_path = cache_path

    def embed(self, texts: list[str], *, pbar: bool) -> dict[str, np.ndarray]:
        ret: dict[str, Any] = {}
        os.makedirs(self._cache_path, exist_ok=True)
        with lmdb.open(self._cache_path, subdir=True, map_size=int(1e+12)) as env:
            with env.begin() as txn:
                unique_texts = set(texts)
                for text in unique_texts:
                    text_sha256 = sha256(text.encode()).digest()
                    embedding = txn.get(text_sha256, None)
                    if embedding is not None:
                        ret[text] = np.array(np.load(BytesIO(embedding)))
            remaining = list(unique_texts - ret.keys())
            if len(remaining) == 0:
                return ret
            with env.begin(write=True) as txn:
                for text, embedding in self._embedder.embed(
                    remaining, pbar=pbar
                ).items():
                    ret[text] = embedding
                    buffer = BytesIO()
                    np.save(buffer, embedding)
                    txn.put(sha256(text.encode()).digest(), buffer.getvalue())
        return ret

    def device(self) -> str:
        return self._embedder.device()


def load_embedder(model: str, batch_size: int) -> RawEmbedder:
    return RawEmbedder(model, batch_size)


def load_caching_embedder(
    model: str, cache_dir: str, batch_size: int
) -> CachingEmbedder:
    model_name = os.path.basename(os.path.normpath(model))
    cache_path = os.path.join(cache_dir, model_name)
    return CachingEmbedder(load_embedder(model, batch_size), cache_path)
