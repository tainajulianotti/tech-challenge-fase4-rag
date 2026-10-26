"""Construção da base de conhecimento: seleção, limpeza e organização das avaliações em documentos."""
import re
import unicodedata

import numpy as np
import pandas as pd

from . import config

RAW_FILES = {
    "reviews": "olist_order_reviews_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "items": "olist_order_items_dataset.csv",
    "products": "olist_products_dataset.csv",
    "customers": "olist_customers_dataset.csv",
}

_WS_RE = re.compile(r"\s+")
_LETTER_RE = re.compile(r"[A-Za-zÀ-ÿ]")
_REPEAT_RE = re.compile(r"([!?.])\1{2,}")


def load_raw(raw_dir=config.RAW_DIR) -> dict[str, pd.DataFrame]:
    """Lê os CSVs originais da Olist necessários para a solução."""
    missing = [f for f in RAW_FILES.values() if not (raw_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Arquivos não encontrados em {raw_dir}: {missing}. Rode `python scripts/download_data.py`."
        )
    return {name: pd.read_csv(raw_dir / fname) for name, fname in RAW_FILES.items()}


def clean_text(text) -> str:
    """Limpeza leve: preserva a linguagem natural (o modelo de embeddings entende contexto, acentos e
    negações), removendo apenas ruídos de formatação."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _REPEAT_RE.sub(r"\1\1", text)  # "!!!!!!" -> "!!"
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_informative(text: str, min_chars: int = 3) -> bool:
    """Descarta textos sem conteúdo semântico (vazios, só pontuação ou números)."""
    return len(text) >= min_chars and bool(_LETTER_RE.search(text))


def normalize_for_dedup(text: str) -> str:
    """Chave para identificar comentários idênticos (ex.: 'Recomendo!' e 'recomendo')."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", text).strip()


def sentiment_from_score(score: int) -> str:
    if score <= 2:
        return "negativo"
    if score == 3:
        return "neutro"
    return "positivo"


def _order_features(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Metadados por pedido: status, datas, atraso na entrega, categoria principal, valores e UF."""
    orders = dfs["orders"].copy()
    for col in ["order_purchase_timestamp", "order_delivered_customer_date", "order_estimated_delivery_date"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    orders["delivery_days"] = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days
    # Atraso positivo = entregue depois da data estimada; negativo = entregue antes.
    orders["delivery_delay_days"] = (
        orders["order_delivered_customer_date"] - orders["order_estimated_delivery_date"]
    ).dt.days

    items = dfs["items"].merge(dfs["products"][["product_id", "product_category_name"]], on="product_id", how="left")
    items["product_category_name"] = items["product_category_name"].fillna("sem_categoria")
    # Pedidos com vários itens: a categoria principal é a do item de maior valor.
    main_cat = (
        items.sort_values("price", ascending=False)
        .drop_duplicates("order_id")[["order_id", "product_category_name"]]
        .rename(columns={"product_category_name": "product_category"})
    )
    agg = items.groupby("order_id").agg(
        n_items=("order_item_id", "count"),
        order_value=("price", "sum"),
        freight_value=("freight_value", "sum"),
    )

    orders = (
        orders.merge(main_cat, on="order_id", how="left")
        .merge(agg, on="order_id", how="left")
        .merge(dfs["customers"][["customer_id", "customer_state"]], on="customer_id", how="left")
    )
    return orders[
        [
            "order_id", "order_status", "order_purchase_timestamp", "delivery_days", "delivery_delay_days",
            "product_category", "n_items", "order_value", "freight_value", "customer_state",
        ]
    ]


def build_documents(dfs: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Gera um documento por avaliação com texto. Retorna o DataFrame e estatísticas de cada etapa."""
    stats = {}
    reviews = dfs["reviews"].copy()
    stats["avaliacoes_totais"] = len(reviews)

    # 1) Seleção: só interessam avaliações com algum texto (título ou comentário).
    reviews["title"] = reviews["review_comment_title"].map(clean_text)
    reviews["message"] = reviews["review_comment_message"].map(clean_text)
    stats["sem_titulo_nem_comentario"] = int(((reviews["title"] == "") & (reviews["message"] == "")).sum())

    # Títulos curtos só são úteis quando agregam algo ao comentário (ex.: título "Péssimo").
    reviews.loc[~reviews["title"].map(is_informative), "title"] = ""
    reviews.loc[~reviews["message"].map(is_informative), "message"] = ""
    reviews = reviews[(reviews["title"] != "") | (reviews["message"] != "")]
    stats["com_texto_informativo"] = len(reviews)

    # 2) Duplicidade: o mesmo review_id aparece vinculado a mais de um pedido; mantemos um documento por avaliação.
    before = len(reviews)
    reviews = reviews.sort_values("review_creation_date").drop_duplicates("review_id", keep="first")
    stats["review_id_duplicados_removidos"] = before - len(reviews)

    # 3) Texto final do documento.
    def compose(row):
        if row.title and row.message and normalize_for_dedup(row.title) not in normalize_for_dedup(row.message):
            return f"{row.title}. {row.message}" if not row.title.endswith((".", "!", "?")) else f"{row.title} {row.message}"
        return row.message or row.title

    reviews["text"] = reviews.apply(compose, axis=1)
    reviews["text_key"] = reviews["text"].map(normalize_for_dedup)

    # 4) Enriquecimento com metadados do pedido.
    docs = reviews.merge(_order_features(dfs), on="order_id", how="left")
    docs["review_creation_date"] = pd.to_datetime(docs["review_creation_date"], errors="coerce")
    docs["review_score"] = docs["review_score"].astype(int)
    docs["sentiment"] = docs["review_score"].map(sentiment_from_score)
    docs["year_month"] = docs["review_creation_date"].dt.strftime("%Y-%m")
    docs["year"] = docs["review_creation_date"].dt.year
    docs["delivered_late"] = np.where(
        docs["delivery_delay_days"].isna(), None, docs["delivery_delay_days"] > 0
    )

    # 5) Valores ausentes nos metadados categóricos recebem rótulos explícitos (não são descartados).
    docs["product_category"] = docs["product_category"].fillna("sem_categoria")
    docs["customer_state"] = docs["customer_state"].fillna("desconhecido")
    docs["order_status"] = docs["order_status"].fillna("desconhecido")

    stats["documentos_finais"] = len(docs)
    stats["textos_unicos"] = docs["text_key"].nunique()

    cols = [
        "review_id", "order_id", "text", "text_key", "title", "message", "review_score", "sentiment",
        "review_creation_date", "year_month", "year", "order_status", "order_purchase_timestamp",
        "delivery_days", "delivery_delay_days", "delivered_late", "product_category", "n_items",
        "order_value", "freight_value", "customer_state",
    ]
    return docs[cols].reset_index(drop=True), stats


def metadata_for_index(row: pd.Series) -> dict:
    """Metadados armazenados junto ao vetor (o Chroma só aceita escalares e não aceita None)."""
    meta = {
        "review_id": row.review_id,
        "order_id": row.order_id,
        "review_score": int(row.review_score),
        "sentiment": row.sentiment,
        "year_month": row.year_month,
        "year": int(row.year),
        "order_status": row.order_status,
        "product_category": row.product_category,
        "customer_state": row.customer_state,
        "text_key": row.text_key,
    }
    if pd.notna(row.delivery_delay_days):
        meta["delivery_delay_days"] = int(row.delivery_delay_days)
        meta["delivered_late"] = bool(row.delivery_delay_days > 0)
    if pd.notna(row.delivery_days):
        meta["delivery_days"] = int(row.delivery_days)
    return meta


def run(save: bool = True) -> tuple[pd.DataFrame, dict]:
    docs, stats = build_documents(load_raw())
    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        docs.to_parquet(config.DOCS_PATH, index=False)
    return docs, stats
