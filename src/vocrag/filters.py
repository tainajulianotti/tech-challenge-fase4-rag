"""Desafio adicional: filtros por metadados extraídos da pergunta e convertidos em filtros `where` do ChromaDB."""
import re
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, Field

from . import config

_MONTH_RE = re.compile(r"\d{4}-(0[1-9]|1[0-2])")


class MetadataFilters(BaseModel):
    min_score: int | None = Field(None, description="Nota mínima (1-5). Ex.: 'avaliações positivas' -> 4")
    max_score: int | None = Field(None, description="Nota máxima (1-5). Ex.: 'nota baixa', 'insatisfeitos' -> 2")
    product_categories: list[str] | None = Field(None, description="Categorias de produto, usando os nomes exatos da lista")
    month_from: str | None = Field(None, description="Início do período no formato AAAA-MM")
    month_to: str | None = Field(None, description="Fim do período no formato AAAA-MM")
    delivered_late: bool | None = Field(None, description="true = pedidos entregues após o prazo estimado; false = no prazo")
    customer_states: list[str] | None = Field(None, description="Siglas das UFs dos clientes, ex.: ['SP', 'RJ']")


@lru_cache(maxsize=1)
def _docs_meta() -> pd.DataFrame:
    return pd.read_parquet(config.DOCS_PATH, columns=["product_category", "customer_state", "year_month"])


@lru_cache(maxsize=1)
def categories() -> list[str]:
    return sorted(_docs_meta()["product_category"].unique())


@lru_cache(maxsize=1)
def months() -> list[str]:
    return sorted(_docs_meta()["year_month"].dropna().unique())


@lru_cache(maxsize=1)
def states() -> list[str]:
    return sorted(_docs_meta()["customer_state"].unique())


def __getattr__(name):  # CATEGORIES, MONTHS, STATES carregados sob demanda
    if name in {"CATEGORIES", "MONTHS", "STATES"}:
        return {"CATEGORIES": categories, "MONTHS": months, "STATES": states}[name]()
    raise AttributeError(name)


def sanitize(f: MetadataFilters) -> MetadataFilters:
    """Valida os filtros sugeridos pelo LLM contra os valores reais da base; o que for inválido é descartado."""
    clean = f.model_copy()
    if clean.min_score is not None:
        clean.min_score = min(max(clean.min_score, 1), 5)
    if clean.max_score is not None:
        clean.max_score = min(max(clean.max_score, 1), 5)
    if clean.min_score == 1:
        clean.min_score = None
    if clean.max_score == 5:
        clean.max_score = None
    if clean.product_categories:
        clean.product_categories = [c for c in clean.product_categories if c in categories()] or None
    if clean.customer_states:
        clean.customer_states = [s.upper() for s in clean.customer_states if s.upper() in states()] or None
    # Período: só aceita AAAA-MM; um período fora da base resulta em lista vazia de meses (e, portanto, sem evidências).
    clean.month_from = clean.month_from if clean.month_from and _MONTH_RE.fullmatch(clean.month_from) else None
    clean.month_to = clean.month_to if clean.month_to and _MONTH_RE.fullmatch(clean.month_to) else None
    return clean


def build_where(f: MetadataFilters | None) -> dict | None:
    """Converte os filtros em sintaxe `where` do ChromaDB. Período vira uma lista de meses ($in)."""
    if f is None:
        return None
    f = sanitize(f)
    conds = []
    if f.min_score is not None:
        conds.append({"review_score": {"$gte": f.min_score}})
    if f.max_score is not None:
        conds.append({"review_score": {"$lte": f.max_score}})
    if f.product_categories:
        conds.append({"product_category": {"$in": f.product_categories}})
    if f.month_from or f.month_to:
        lo, hi = f.month_from or months()[0], f.month_to or months()[-1]
        period = [m for m in months() if lo <= m <= hi] or ["fora_do_periodo_da_base"]
        conds.append({"year_month": {"$in": period}})
    if f.delivered_late is not None:
        conds.append({"delivered_late": f.delivered_late})
    if f.customer_states:
        conds.append({"customer_state": {"$in": f.customer_states}})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}
