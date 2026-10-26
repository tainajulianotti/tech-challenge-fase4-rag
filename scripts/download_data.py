"""Baixa a base 'Brazilian E-Commerce Public Dataset by Olist' do Kaggle para data/raw."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import kagglehub  # noqa: E402

from vocrag import config  # noqa: E402

DATASET = "olistbr/brazilian-ecommerce"


def main():
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(kagglehub.dataset_download(DATASET))
    for csv in path.glob("*.csv"):
        shutil.copy2(csv, config.RAW_DIR / csv.name)
        print(f"  {csv.name}")
    print(f"Arquivos copiados para {config.RAW_DIR}")


if __name__ == "__main__":
    main()
