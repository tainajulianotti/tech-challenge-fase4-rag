"""Recuperação semântica: busca multi-consulta, fusão, deduplicação, limiar de relevância e MMR."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .indexing import embed_queries, get_collection


@dataclass
class Evidence:
    ref: str                 # identificador curto usado nas citações, ex.: "R3"
    review_id: str
    text: str
    similarity: float
    metadata: dict
    identical_count: int = 1  # quantas avaliações recuperadas têm exatamente o mesmo texto
    matched_queries: list = field(default_factory=list)


@dataclass
class RetrievalResult:
    queries: list[str]
    evidences: list[Evidence]
    candidates: pd.DataFrame  # todos os candidatos (antes do corte), útil para análise
    support: dict             # estimativa de volume/perfil das avaliações relacionadas ao tema


class Retriever:
    def __init__(self, collection=None):
        self.collection = collection or get_collection()

    def _search(self, queries: list[str], n_results: int, where: dict | None = None, with_embeddings: bool = True) -> pd.DataFrame:
        q_emb = embed_queries(queries)
        res = self.collection.query(
            query_embeddings=q_emb.tolist(),
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"] + (["embeddings"] if with_embeddings else []),
        )
        rows = []
        for qi, query in enumerate(queries):
            for rid, doc, meta, dist, emb in zip(
                res["ids"][qi], res["documents"][qi], res["metadatas"][qi], res["distances"][qi],
                res["embeddings"][qi] if with_embeddings else [None] * len(res["ids"][qi]),
            ):
                rows.append(
                    {"review_id": rid, "text": doc, "metadata": meta, "similarity": 1 - dist,
                     "embedding": None if emb is None else np.asarray(emb), "query": query}
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _fuse(hits: pd.DataFrame) -> pd.DataFrame:
        """Fusão das buscas: cada avaliação fica com a maior similaridade obtida entre as consultas."""
        if hits.empty:
            return hits
        matched = hits.groupby("review_id")["query"].apply(list)
        fused = hits.sort_values("similarity", ascending=False).drop_duplicates("review_id").set_index("review_id")
        fused["matched_queries"] = matched
        return fused.reset_index().sort_values("similarity", ascending=False)

    @staticmethod
    def _dedup_texts(cands: pd.DataFrame) -> pd.DataFrame:
        """Comentários idênticos ('Recomendo', 'Ótimo produto') ocupariam o contexto sem agregar informação.
        Mantemos um representante e registramos quantas vezes o texto apareceu."""
        cands = cands.copy()
        cands["text_key"] = cands["metadata"].map(lambda m: m.get("text_key", ""))
        counts = cands.groupby("text_key")["review_id"].transform("count")
        cands["identical_count"] = counts
        return cands.drop_duplicates("text_key", keep="first")

    @staticmethod
    def _mmr(cands: pd.DataFrame, k: int, lam: float) -> pd.DataFrame:
        """Maximal Marginal Relevance: equilibra relevância e diversidade para que o contexto cubra
        diferentes aspectos do tema (ex.: atraso, extravio, entrega parcial) em vez de 15 relatos iguais."""
        if len(cands) <= k:
            return cands
        emb = np.vstack(cands["embedding"].values)
        rel = cands["similarity"].values
        selected = [0]
        remaining = list(range(1, len(cands)))
        while len(selected) < k and remaining:
            sim_to_sel = emb[remaining] @ emb[selected].T
            scores = lam * rel[remaining] - (1 - lam) * sim_to_sel.max(axis=1)
            best = remaining[int(np.argmax(scores))]
            selected.append(best)
            remaining.remove(best)
        return cands.iloc[selected]

    def _support(self, queries: list[str], where: dict | None) -> dict:
        """Estimativa do volume de avaliações sobre o tema: quantas, entre as mais próximas, passam do limiar.
        Ajuda o LLM a falar de recorrência com cautela, sem inventar números."""
        pool = self._fuse(self._search(queries, config.SUPPORT_POOL, where, with_embeddings=False))
        relevant = pool[pool["similarity"] >= config.MIN_SIMILARITY] if not pool.empty else pool
        if relevant.empty:
            return {"avaliacoes_relacionadas": 0}
        scores = relevant["metadata"].map(lambda m: m["review_score"])
        cats = relevant["metadata"].map(lambda m: m["product_category"]).value_counts().head(5)
        late = relevant["metadata"].map(lambda m: m.get("delivered_late"))
        return {
            "avaliacoes_relacionadas": int(len(relevant)),
            "limite_da_amostra": len(relevant) >= config.SUPPORT_POOL,
            "nota_media": round(float(scores.mean()), 2),
            "pct_notas_1_2": round(float((scores <= 2).mean() * 100), 1),
            "pct_entregues_com_atraso": round(float((late == True).sum() / max(late.notna().sum(), 1) * 100), 1),  # noqa: E712
            "principais_categorias": cats.to_dict(),
        }

    def retrieve(self, queries: list[str], where: dict | None = None, top_k: int = config.FINAL_TOP_K) -> RetrievalResult:
        cands = self._fuse(self._search(queries, config.CANDIDATES_PER_QUERY, where))
        relevant = cands[cands["similarity"] >= config.MIN_SIMILARITY] if not cands.empty else cands
        evidences = []
        if not relevant.empty:
            selected = self._mmr(self._dedup_texts(relevant), top_k, config.MMR_LAMBDA)
            for i, row in enumerate(selected.itertuples(index=False), start=1):
                evidences.append(
                    Evidence(
                        ref=f"R{i}", review_id=row.review_id, text=row.text, similarity=round(float(row.similarity), 4),
                        metadata=row.metadata, identical_count=int(row.identical_count),
                        matched_queries=row.matched_queries,
                    )
                )
        support = self._support(queries, where) if evidences else {"avaliacoes_relacionadas": 0}
        return RetrievalResult(queries=queries, evidences=evidences, candidates=cands.drop(columns="embedding", errors="ignore"), support=support)
