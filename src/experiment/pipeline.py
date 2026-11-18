"""Шаги пайплайна, которые вызывает Airflow.

Каждый шаг — обычная функция, которую можно запустить и проверить без Airflow:
`python -m experiment.pipeline validate`. DAG только связывает их в цепочку,
поэтому расчёты тестируются независимо от оркестратора.

Шаги пишут артефакты в results/ и data/, idempotent: повторный запуск даёт те же числа.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment.stats import diff_interval, obf_boundaries, two_proportion_z, wilson_interval

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SNAPSHOT = DATA / "cookie_cats.csv"
ACCRUAL = DATA / "accrual.csv"
DAILY = RESULTS / "daily_metrics.csv"
LOOKS = RESULTS / "sequential_looks.csv"
LOOK_DAYS = (5, 10, 15, 20, 25, 30)   # еженедельные взгляды

DAYS = 30               # на сколько дней разворачиваем срез
CONTROL, TREATMENT = "gate_30", "gate_40"


def _ensure_dirs() -> None:
    DATA.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)


def step_ingest(days: int = DAYS, seed: int = 42) -> dict:
    """Разворачивает статический срез в ежедневный прирост.

    Датасет — срез на 14 дней, а не поток. Чтобы контур было что пересчитывать
    ежедневно, игроки раскладываются по дням установки (детерминированно, по seed).
    Это синтетика, и в README она названа синтетикой: анализ настоящий, прирост — нет.
    """
    _ensure_dirs()
    df = pd.read_csv(SNAPSHOT)
    rng = np.random.default_rng(seed)
    df["install_day"] = rng.integers(1, days + 1, len(df))
    df = df.sort_values(["install_day", "userid"]).reset_index(drop=True)
    df.to_csv(ACCRUAL, index=False)
    return {"rows": int(len(df)), "days": days, "path": str(ACCRUAL.relative_to(ROOT))}


def step_validate() -> dict:
    """Проверки схемы и SRM на входе: битые данные не должны идти в расчёт."""
    df = pd.read_csv(ACCRUAL)
    expected = ["userid", "version", "sum_gamerounds", "retention_1", "retention_7", "install_day"]
    if list(df.columns) != expected:
        raise ValueError(f"неожиданные колонки: {list(df.columns)}")
    if df.isna().any().any():
        raise ValueError("в данных есть пропуски")
    if df["userid"].duplicated().any():
        raise ValueError("дубли userid")

    counts = df["version"].value_counts().to_dict()
    n = sum(counts.values())
    expected_share = n / 2
    chi2 = sum((c - expected_share) ** 2 / expected_share for c in counts.values())
    p = float(1 - _chi2_cdf(chi2))
    shift_pp = max(abs(c - expected_share) for c in counts.values()) / n * 100

    result = {"rows": int(n), "counts": {k: int(v) for k, v in counts.items()},
              "srm_chi2": round(float(chi2), 3), "srm_p": round(p, 4),
              "srm_shift_pp": round(float(shift_pp), 2)}
    (RESULTS / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    if p < 0.001 and shift_pp > 1.0:
        raise ValueError(f"SRM: p={p:.4f}, отклонение {shift_pp:.2f} п.п. — проверьте рандомизацию")
    return result


def _chi2_cdf(x: float) -> float:
    """CDF хи-квадрата с одной степенью свободы (без scipy в этом шаге)."""
    from math import erf, sqrt
    return erf(sqrt(x / 2))


def step_daily_metrics(days: int = DAYS) -> dict:
    """Накопительные метрики по дням + z-статистика для последовательного мониторинга.

    На день d берём всех, кто установил игру до d включительно: так выглядит
    реальный мониторинг, когда данные подходят постепенно.
    """
    df = pd.read_csv(ACCRUAL)
    df["retention_1"] = df["retention_1"].astype(bool)
    df["retention_7"] = df["retention_7"].astype(bool)

    rows = []
    total_n = len(df)
    for day in range(1, days + 1):
        part = df[df.install_day <= day]
        if part.empty:
            continue
        c = part[part.version == CONTROL]
        t = part[part.version == TREATMENT]
        k1, n1 = int(c.retention_1.sum()), len(c)
        k2, n2 = int(t.retention_1.sum()), len(t)
        if n1 == 0 or n2 == 0:
            continue
        r = diff_interval(k1, n1, k2, n2)
        z, _ = two_proportion_z(k1, n1, k2, n2)
        rows.append({
            "day": day, "n": len(part), "info_fraction": len(part) / total_n,
            "n_control": n1, "n_treatment": n2,
            "rate_control_pct": round(r["p1"] * 100, 3),
            "rate_treatment_pct": round(r["p2"] * 100, 3),
            "diff_pp": round(r["diff_pp"], 3),
            "ci_lo_pp": round(r["diff_ci_pp"][0], 3),
            "ci_hi_pp": round(r["diff_ci_pp"][1], 3),
            "z": round(float(z), 3),
        })
    daily = pd.DataFrame(rows)
    daily.to_csv(DAILY, index=False)
    return {"days": int(len(daily)), "path": str(DAILY.relative_to(ROOT))}


def step_monitor() -> dict:
    """Последовательный мониторинг D1: еженедельные взгляды, границы O'Brien–Fleming.

    Смотрим не каждый день (30 взглядов — это почти непрерывный мониторинг и очень
    консервативные границы), а раз в неделю: 6 взглядов за 30 дней.
    """
    daily = pd.read_csv(DAILY)
    looks = daily[daily["day"].isin(LOOK_DAYS)].reset_index(drop=True)
    times = looks["info_fraction"].tolist()
    bounds = obf_boundaries(times)
    looks["boundary"] = bounds
    looks["crossed"] = looks["z"].abs() >= looks["boundary"]
    crossed = looks[looks["crossed"]]

    result = {
        "looks": int(len(looks)),
        "look_days": LOOK_DAYS,
        "final_z": round(float(looks.z.iloc[-1]), 3),
        "final_boundary": round(float(looks.boundary.iloc[-1]), 3),
        "crossed_at_day": int(crossed.day.iloc[0]) if len(crossed) else None,
        "decision": "эффект есть" if len(crossed) else "граница не пересечена",
    }
    looks.to_csv(LOOKS, index=False)
    (RESULTS / "monitor.json").write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return result


def step_effects() -> dict:
    """Итоговые эффекты по всему срезу (то же, что scripts/run_effects.py, но для DAG)."""
    df = pd.read_csv(ACCRUAL)
    for col in ("retention_1", "retention_7"):
        df[col] = df[col].astype(bool)
    df["zero_rounds"] = df["sum_gamerounds"] == 0
    df["heavy_user"] = df["sum_gamerounds"] >= 1000

    out = {}
    for name in ("retention_1", "retention_7", "zero_rounds", "heavy_user"):
        c = df[df.version == CONTROL][name]
        t = df[df.version == TREATMENT][name]
        r = diff_interval(int(c.sum()), len(c), int(t.sum()), len(t))
        out[name] = {"control_pct": round(r["p1"] * 100, 3), "treatment_pct": round(r["p2"] * 100, 3),
                     "diff_pp": round(r["diff_pp"], 3),
                     "ci_pp": [round(v, 3) for v in r["diff_ci_pp"]],
                     "significant": bool(r["significant"])}
    (RESULTS / "dag_effects.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    return out


def step_memo() -> dict:
    """Собирает меморандум из текущих артефактов — как шаг контура, а не вручную."""
    import csv as _csv

    from experiment.decide import decide
    from experiment.memo import MDE, render

    effects = json.loads((RESULTS / "effects.json").read_text(encoding="utf-8"))
    eda = json.loads((RESULTS / "eda.json").read_text(encoding="utf-8"))
    monitor = json.loads((RESULTS / "monitor.json").read_text(encoding="utf-8"))
    looks = list(_csv.DictReader((LOOKS).open(encoding="utf-8")))
    verdict = decide(effects["primary"], effects["guardrails"], MDE)
    text = render(effects, eda, verdict, monitor=monitor, looks=looks)
    (RESULTS / "memo.md").write_text(text, encoding="utf-8")
    return {"path": str((RESULTS / "memo.md").relative_to(ROOT)), "chars": len(text)}


def step_publish(db_url: str | None = None) -> dict:
    """Публикует витрины в Postgres, откуда их читает Metabase.

    Пишем через COPY FROM STDIN: pandas.to_sql в контейнере Airflow не распознаёт
    соединение SQLAlchemy, а COPY и быстрее, и без версионных сюрпризов.
    Без базы шаг не падает: возвращает `skipped`, чтобы DAG работал и локально.
    """
    import io
    import os

    from sqlalchemy import create_engine, text

    url = db_url or os.environ.get("ANALYTICS_DB_URL")
    if not url:
        return {"skipped": True, "reason": "ANALYTICS_DB_URL не задан"}

    daily = pd.read_csv(DAILY)
    looks = pd.read_csv(LOOKS)
    effects = pd.read_json(RESULTS / "dag_effects.json", orient="index").reset_index()
    effects.columns = ["metric"] + list(effects.columns[1:])

    # качество эксперимента — плоская таблица для карточки SRM
    validation = json.loads((RESULTS / "validation.json").read_text(encoding="utf-8"))
    quality = pd.DataFrame([{
        "rows": validation["rows"],
        "gate_30": validation["counts"].get("gate_30"),
        "gate_40": validation["counts"].get("gate_40"),
        "srm_chi2": validation["srm_chi2"],
        "srm_p": validation["srm_p"],
        "srm_shift_pp": validation["srm_shift_pp"],
    }])

    tables = {
        "exp_daily_metrics": daily,
        "exp_sequential_looks": looks,
        "exp_effects": effects,
        "exp_quality": quality,
        "exp_segments": pd.read_csv(RESULTS / "segments.csv"),
    }

    engine = create_engine(url)
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("CREATE SCHEMA IF NOT EXISTS analytics")
        for name, frame in tables.items():
            _publish_table(cursor, name, frame)
        raw.commit()
    finally:
        raw.close()

    return {"schema": "analytics", "tables": {name: int(len(frame)) for name, frame in tables.items()}}


def _sql_type(series: pd.Series) -> str:
    kind = series.dtype.kind
    if kind == "b":
        return "boolean"
    if kind in "iu":
        return "bigint"
    if kind == "f":
        return "double precision"
    return "text"


def _publish_table(cursor, name: str, frame: pd.DataFrame) -> None:
    """Полная перезапись таблицы через COPY: идемпотентно и без pandas.to_sql."""
    import io

    columns = [f'"{c}"' for c in frame.columns]
    definitions = ", ".join(f'"{c}" {_sql_type(frame[c])}' for c in frame.columns)
    cursor.execute(f"DROP TABLE IF EXISTS analytics.{name}")
    cursor.execute(f"CREATE TABLE analytics.{name} ({definitions})")

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, header=False)
    buffer.seek(0)
    cursor.copy_expert(f"COPY analytics.{name} ({', '.join(columns)}) FROM STDIN WITH CSV", buffer)


STEPS = {
    "ingest": step_ingest,
    "validate": step_validate,
    "daily": step_daily_metrics,
    "monitor": step_monitor,
    "effects": step_effects,
    "memo": step_memo,
    "publish": step_publish,
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(f"использование: python -m experiment.pipeline {{{'|'.join(STEPS)}}}")
    result = STEPS[sys.argv[1]]()
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
