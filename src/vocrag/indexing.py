"""Criação dos embeddings e armazenamento/indexação na base vetorial (ChromaDB persistente)."""
from functools import lru_cache

import chromadb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from . import config
from .data_prep import metadata_for_index


@lru_cache(maxsize=1)
def get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_passages(texts: list[str], show_progress: bool = False) -> np.ndarray:
    return get_embedder().encode(
        [config.PASSAGE_PREFIX + t for t in texts],
        batch_size=config.EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )


def embed_queries(queries: list[str]) -> np.ndarray:
    return get_embedder().encode(
        [config.QUERY_PREFIX + q for q in queries], normalize_embeddings=True
    )


def get_client():
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    return get_client().get_collection(config.COLLECTION_NAME)


def build_index(docs: pd.DataFrame, reset: bool = True, chunk: int = 2000):
    """Gera os embeddings de todas as avaliações e grava texto + vetor + metadados no Chroma.

    Cada avaliação é um documento inteiro (sem chunking): os comentários são curtos (mediana de poucas
    dezenas de palavras), então dividir quebraria o contexto da opinião do cliente.
    """
    client = get_client()
    if reset and config.COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION_NAME)
    collection = client.get_or_create_collection(
        config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": config.EMBEDDING_MODEL},
    )

    for start in tqdm(range(0, len(docs), chunk), desc="Indexando"):
        part = docs.iloc[start : start + chunk]
        collection.add(
            ids=part["review_id"].tolist(),
            documents=part["text"].tolist(),
            embeddings=embed_passages(part["text"].tolist()),
            metadatas=[metadata_for_index(r) for r in part.itertuples(index=False)],
        )
    return collection
