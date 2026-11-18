"""Валидация данных и EDA: то, что делается до расчёта эффектов.

Пишет results/eda.json — компактный машинночитаемый отчёт, на который дальше
опираются и тесты, и меморандум.

    python scripts/run_eda.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cookie_cats.csv"
OUT = ROOT / "results" / "eda.json"

EXPECTED_COLUMNS = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7"]
EXPECTED_ROWS = 90_189


def load(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        sys.exit(f"неожиданные колонки: {list(df.columns)}")
    if len(df) != EXPECTED_ROWS:
        sys.exit(f"строк {len(df)}, ожидали {EXPECTED_ROWS}")
    return df


def srm_check(counts: dict[str, int]) -> dict:
    """Sample Ratio Mismatch: χ² на равенство долей групп.

    При 90 тыс. игроков даже доли процента дают «значимость», поэтому кроме
    p-value смотрим на абсолютное отклонение от 50/50.
    """
    observed = np.array(list(counts.values()), dtype=float)
    expected = np.full_like(observed, observed.sum() / len(observed))
    chi2, p = stats.chisquare(observed, expected)
    total = observed.sum()
    max_shift_pp = float(np.max(np.abs(observed - expected)) / total * 100)
    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "chi2": round(float(chi2), 3),
        "p_value": round(float(p), 4),
        "max_shift_pp": round(max_shift_pp, 2),
        # при таком n «значимо» и 0,4 п.п. — решаем по практике, а не по p
        "practically_balanced": bool(max_shift_pp < 1.0),
    }


def group_rates(df: pd.DataFrame) -> dict:
    by = df.groupby("version")
    return {
        "retention_1": {k: round(float(v) * 100, 2) for k, v in by["retention_1"].mean().items()},
        "retention_7": {k: round(float(v) * 100, 2) for k, v in by["retention_7"].mean().items()},
        "zero_rounds_share": {k: round(float(v) * 100, 2)
                              for k, v in by["sum_gamerounds"].apply(lambda s: (s == 0).mean()).items()},
        "heavy_users_share": {k: round(float(v) * 100, 3)
                              for k, v in by["sum_gamerounds"].apply(lambda s: (s >= 1000).mean()).items()},
    }


def rounds_profile(df: pd.DataFrame) -> dict:
    s = df["sum_gamerounds"]
    return {
        "mean": round(float(s.mean()), 2),
        "median": float(s.median()),
        "p99": float(s.quantile(0.99)),
        "max": int(s.max()),
        "zeros": int((s == 0).sum()),
        "above_1000": int((s > 1000).sum()),
        "max_share_of_mean_pp": round(float(s.max() / s.sum() * 100), 2),
    }


def main() -> None:
    df = load()
    counts = df["version"].value_counts().to_dict()

    eda = {
        "rows": int(len(df)),
        "columns": EXPECTED_COLUMNS,
        "missing": int(df.isna().sum().sum()),
        "duplicate_userids": int(df["userid"].duplicated().sum()),
        "srm": srm_check(counts),
        "rates": group_rates(df),
        "rounds": rounds_profile(df),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(eda, ensure_ascii=False, indent=1), encoding="utf-8")

    srm = eda["srm"]
    print(f"строк {eda['rows']:,} | пропусков {eda['missing']} | дублей {eda['duplicate_userids']}")
    print(f"группы: {srm['counts']}")
    print(f"SRM: χ² = {srm['chi2']}, p = {srm['p_value']}, "
          f"макс. отклонение от 50/50 = {srm['max_shift_pp']} п.п. "
          f"→ практически {'сбалансировано' if srm['practically_balanced'] else 'НЕ сбалансировано'}")
    print(f"удержание D1: {eda['rates']['retention_1']}")
    print(f"удержание D7: {eda['rates']['retention_7']}")
    print(f"нулевых раундов: {eda['rates']['zero_rounds_share']}")
    print(f"раунды: среднее {eda['rounds']['mean']}, медиана {eda['rounds']['median']:.0f}, "
          f"максимум {eda['rounds']['max']:,} ({eda['rounds']['max_share_of_mean_pp']} % суммы)")
    print(f"написано: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
