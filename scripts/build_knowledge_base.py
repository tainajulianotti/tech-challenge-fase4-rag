"""Pipeline offline: prepara as avaliações, gera os embeddings e cria o índice vetorial."""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vocrag import config, data_prep, indexing  # noqa: E402


def main():
    t0 = time.time()
    docs, stats = data_prep.run()
    print("Preparação da base:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Documentos salvos em {config.DOCS_PATH}")

    collection = indexing.build_index(docs)
    print(f"Índice '{config.COLLECTION_NAME}' com {collection.count()} vetores em {config.CHROMA_DIR}")
    print(f"Tempo total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
