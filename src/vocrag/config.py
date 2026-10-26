"""Configurações centrais do projeto (caminhos, modelos e hiperparâmetros do RAG)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# Silencia avisos e barras de progresso do HuggingFace (mantém os notebooks legíveis).
for _var, _val in {
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_VERBOSITY": "error",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ.setdefault(_var, _val)

# Caminhos
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma"
DOCS_PATH = PROCESSED_DIR / "reviews_docs.parquet"

# Base vetorial
COLLECTION_NAME = "olist_reviews"

# Embeddings: modelo multilíngue, leve e treinado para busca assimétrica (pergunta curta -> passagem).
# O E5 exige os prefixos "query: " e "passage: ".
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "
EMBED_BATCH_SIZE = 128

# LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Modelos de reserva caso o principal esteja sobrecarregado (503) ou sem cota (429).
GEMINI_FALLBACK_MODELS = [m for m in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-3.8-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",") if m]
LLM_MAX_RETRIES = 2          # tentativas por modelo em caso de sobrecarga (503)
LLM_COOLDOWN_SECONDS = 120   # modelo sem cota (429) é pulado por este período
TEMPERATURE = 0.1

# Recuperação
CANDIDATES_PER_QUERY = 60   # candidatos buscados por variação da pergunta
SUPPORT_POOL = 1000         # candidatos usados para estimar o volume de avaliações sobre o tema
FINAL_TOP_K = 15            # avaliações enviadas como contexto ao LLM
MMR_LAMBDA = 0.7            # 1.0 = só relevância, 0.0 = só diversidade
# O E5 gera similaridades "comprimidas" (textos sem relação ficam em ~0,82-0,88). O limiar foi calibrado comparando
# consultas do domínio (top-1 entre 0,92 e 0,96) com consultas fora dele (raramente acima de 0,89) - ver notebook 02.
MIN_SIMILARITY = 0.89       # similaridade mínima (cosseno) para uma avaliação ser considerada evidência
MIN_EVIDENCES = 5           # mínimo de evidências acima do limiar para chamar o LLM
