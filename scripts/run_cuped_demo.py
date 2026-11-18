"""CUPED на синтетическом предпериоде: сколько дисперсии снял бы метод.

В этом эксперименте предпериода нет, поэтому показать пользу можно только
честной симуляцией: берём реальные раунды игрока как «после», строим «до»
с заданной корреляцией и смотрим, насколько сузится интервал.

    python scripts/run_cuped_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment.advanced import cuped_adjust, cuped_variance_reduction  # noqa: E402

DATA = ROOT / "data" / "cookie_cats.csv"
OUT = ROOT / "results" / "cuped_demo.json"

RHOS = (0.3, 0.5, 0.7, 0.9)


def build_panel(post: np.ndarray, rho: float, seed: int) -> np.ndarray:
    """Предпериод с заданной корреляцией: pre = ρ·post + шум."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.std(post) * np.sqrt(1 - rho**2), len(post))
    return rho * (post - post.mean()) / np.std(post) * np.std(post) + post.mean() + noise


def main() -> None:
    df = pd.read_csv(DATA)
    df["log_rounds"] = np.log1p(df["sum_gamerounds"])

    rows = []
    for rho in RHOS:
        pre = build_panel(df["log_rounds"].to_numpy(), rho, seed=42)
        adjusted = cuped_adjust(pre, df["log_rounds"].to_numpy())
        reduction = cuped_variance_reduction(pre, df["log_rounds"].to_numpy())

        # насколько сузится стандартная ошибка разницы средних
        se_raw = np.std(df["log_rounds"]) * np.sqrt(2 / len(df))
        se_cuped = np.std(adjusted) * np.sqrt(2 / len(df))
        rows.append({
            "rho": rho,
            "variance_reduction_pct": round(reduction * 100, 1),
            "se_raw": round(float(se_raw), 5),
            "se_cuped": round(float(se_cuped), 5),
            "se_shrink_pct": round(float((1 - se_cuped / se_raw) * 100), 1),
            "equivalent_sample_gain_pct": round(float((1 / (se_cuped / se_raw) ** 2 - 1) * 100), 1),
        })

    OUT.write_text(json.dumps({"rows": rows, "n": int(len(df))}, ensure_ascii=False, indent=1),
                   encoding="utf-8")

    print("CUPED на синтетическом предпериоде (log(1+раунды), n = {:,})".format(len(df)))
    print(f"{'ρ':>4} {'снижение дисперсии':>20} {'SE до':>8} {'SE после':>9} {'выигрыш выборки':>16}")
    for row in rows:
        print(f"{row['rho']:>4} {row['variance_reduction_pct']:>19}% {row['se_raw']:>8} "
              f"{row['se_cuped']:>9} {row['equivalent_sample_gain_pct']:>15}%")
    print(f"\nнаписано: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
