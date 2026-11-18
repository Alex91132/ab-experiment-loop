"""Сегменты: где эффект сильнее и где его нет.

Разбиение нужно, чтобы не делать вывод по среднему там, где половина аудитории
ведёт себя иначе. Поправка Холма — потому что сегментов несколько, и «нашли
значимое в одном из шести» — это классическая ловушка.
"""

from __future__ import annotations

import pandas as pd

from experiment.stats import diff_interval, holm

MIN_SEGMENT = 200          # меньше — сегмент не анализируем, только помечаем

ACTIVITY_BINS = [
    (-1, 0, "0 раундов"),
    (1, 10, "1–10"),
    (11, 50, "11–50"),
    (51, 10**9, "51+"),
]


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет активность и декаду установки — оси, по которым режем."""
    frame = df.copy()

    def bucket(rounds: float) -> str:
        for low, high, label in ACTIVITY_BINS:
            if low < rounds <= high:
                return label
        return "0 раундов"

    frame["activity"] = frame["sum_gamerounds"].map(bucket)
    if "install_day" in frame.columns:      # есть только в ежедневном приросте
        frame["install_decade"] = ((frame["install_day"] - 1) // 10 + 1).map(
            lambda d: f"дни {10 * (d - 1) + 1}–{10 * d}")
    return frame


def segment_effects(df: pd.DataFrame, by: str, metric: str = "retention_1",
                    control: str = "gate_30", treatment: str = "gate_40") -> pd.DataFrame:
    """Эффект по сегментам с поправкой Холма на число сегментов."""
    rows = []
    for name, part in df.groupby(by):
        c = part[part.version == control][metric]
        t = part[part.version == treatment][metric]
        if len(c) < MIN_SEGMENT or len(t) < MIN_SEGMENT:
            rows.append({"segment": name, "n_control": len(c), "n_treatment": len(t),
                         "note": "мало данных"})
            continue
        r = diff_interval(int(c.sum()), len(c), int(t.sum()), len(t))
        rows.append({
            "segment": name,
            "n_control": len(c), "n_treatment": len(t),
            "control_pct": round(r["p1"] * 100, 2),
            "treatment_pct": round(r["p2"] * 100, 2),
            "diff_pp": round(r["diff_pp"], 2),
            "ci_lo": round(r["diff_ci_pp"][0], 2),
            "ci_hi": round(r["diff_ci_pp"][1], 2),
            "p_value": round(1.0 - _share_under_normal(r), 4),
            "note": "",
        })

    table = pd.DataFrame(rows)
    analysed = table[table["note"] == ""]
    if len(analysed):
        adjusted = holm({str(row.segment): float(row.p_value) for row in analysed.itertuples()})
        table["p_holm"] = table["segment"].astype(str).map(
            lambda s: adjusted.get(s, {}).get("p_holm"))
        table["significant"] = table["segment"].astype(str).map(
            lambda s: adjusted.get(s, {}).get("reject"))
    return table


def _share_under_normal(result: dict) -> float:
    """Доля под кривой до |z| — чтобы получить двусторонний p из разницы и её SE."""
    from scipy import stats

    se = result["se_pp"] / 100
    if se == 0:
        return 1.0
    z = abs(result["diff_pp"] / 100) / se
    return float(2 * stats.norm.cdf(z) - 1)


def summary(table: pd.DataFrame) -> dict:
    """Короткая сводка для меморандума: где значимо, где нет."""
    analysed = table[table["note"] == ""]
    sig = analysed[analysed.get("significant") == True] if "significant" in analysed else analysed.iloc[0:0]
    return {
        "segments": int(len(analysed)),
        "significant": int(len(sig)),
        "names": [str(s) for s in sig["segment"].tolist()],
        "max_abs_diff": float(analysed["diff_pp"].abs().max()) if len(analysed) else 0.0,
    }
