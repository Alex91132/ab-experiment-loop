"""Скачивание датасета Cookie Cats с проверкой контрольной суммы.

Данные не лежат в репозитории: 2,6 МБ CSV, лицензия исходного датасета —
Kaggle «Mobile Games: A/B Testing». Скрипт берёт открытое зеркало и сверяет
SHA-256, чтобы подмена файла не прошла незамеченной.

    python scripts/fetch_data.py            # скачать в data/cookie_cats.csv
    python scripts/fetch_data.py --check    # только проверить имеющийся файл
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "cookie_cats.csv"

MIRROR = "https://raw.githubusercontent.com/Waseekk/Machine-Learning_Deep-Learning/main/cookie_cats.csv"
SHA256 = "5AB54D761FBDDCD50DE7B88E4EAF7837CBA4569474F50C043A4D17EE342C46BD"
ROWS = 90_189


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def validate(path: Path) -> None:
    """Проверки, которые ловят подмену файла и битую загрузку."""
    if not path.exists():
        sys.exit(f"нет файла {path} — запустите без --check, чтобы скачать")
    actual = sha256_of(path)
    if actual != SHA256:
        sys.exit(f"контрольная сумма не совпала:\n  ожидали {SHA256}\n  получили {actual}")

    with path.open(encoding="utf-8") as fh:
        header = fh.readline().strip()
        rows = sum(1 for _ in fh)
    if header != "userid,version,sum_gamerounds,retention_1,retention_7":
        sys.exit(f"неожиданные колонки: {header}")
    if rows != ROWS:
        sys.exit(f"строк {rows}, ожидали {ROWS}")
    print(f"данные в порядке: {rows:,} строк, sha256 {actual[:16]}…")


def download() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"скачиваю {MIRROR}")
    with urllib.request.urlopen(MIRROR, timeout=90) as response, DEST.open("wb") as out:
        out.write(response.read())
    print(f"сохранено: {DEST} ({DEST.stat().st_size / 1024:.0f} КБ)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="только проверить файл")
    args = parser.parse_args()

    if not args.check:
        download()
    validate(DEST)
    print("если зеркало умерло — скачайте датасет с Kaggle "
          "(Mobile Games: A/B Testing) и положите в data/cookie_cats.csv")


if __name__ == "__main__":
    main()
